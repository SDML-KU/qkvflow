import logging

# GPU performance tips
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

import datasets
import equinox as eqx
import jax
import jax.random as jrandom
import levanter
import optax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.data.sharded_dataset import (
    ShardedDataset,
    TextUrlDataset,
    WrappedHFDataset,
)
from levanter.data.text import CausalLmDataset, LMDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import OptimizerConfig, Trainer, TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths, parameter_count
from optax import GradientTransformation

import wandb
from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.nn.dynamic_llama import LlamaLMHeadModel as LlamaODELMHeadModel


os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)


logger = logging.getLogger(__name__)


class HFDataset(WrappedHFDataset):
    def __init__(self, id, val_ratio=0.0005, *, split, **kwargs):
        self.val_ratio = val_ratio
        super().__init__(id, split=split, **kwargs)

    def _load_dataset(self):
        raw_dataset = datasets.load_dataset(self.id, **self.kwargs)
        if "validation" not in raw_dataset:
            if not self.kwargs["streaming"]:
                # split train into subsets
                assert "train" in raw_dataset
                raw_dataset = raw_dataset["train"].train_test_split(
                    test_size=self.val_ratio,
                    seed=2357,  # same seed like flash-attention
                    shuffle=True,  # Otherwise test will be at the end of the dataset
                )
                raw_dataset["validation"] = raw_dataset["test"]
            else:
                raise NotImplementedError()

        return raw_dataset[self.split]


class DatasetConfig(LMDatasetConfig):

    val_ratio: float = 0.0005

    def get_shard_source(self, split) -> ShardedDataset[str]:
        if self.id is not None:
            hf_dataset = HFDataset(
                self.id,
                split=split,
                val_ratio=self.val_ratio,
                name=self.name,
                streaming=self.stream,
            )
            return hf_dataset.map(lambda x: x[self.text_key])
        else:
            return TextUrlDataset(self.urls_for_split(split), self.text_key)


@dataclass
class OptimizerConfigWithWeightDecay(OptimizerConfig):

    weight_decay_modules: Optional[Union[List[str], str]] = None

    def build(self, num_train_steps: int) -> GradientTransformation:
        """Creates the optimizer"""

        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))

            if self.weight_decay > 0:
                components.append(
                    optax.add_decayed_weights(
                        self.weight_decay, self.build_weight_decay_mask()
                    )
                )

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps)
        )

    def build_weight_decay_mask(self):
        if self.weight_decay_modules is None:
            return None
        else:
            # mask based on regex or module path
            def _apply_on(x, key_path):
                if isinstance(self.weight_decay_modules, str):
                    compiled_regex = re.compile(self.weight_decay_modules)
                    return compiled_regex.match(key_path) is not None
                else:
                    return any(
                        key_path.__contains__(target)
                        for target in self.weight_decay_modules
                    )

            def mask_fn(model):
                return jax.tree_util.tree_map(
                    _apply_on,
                    model,
                    leaf_key_paths(model, is_leaf=eqx.is_array),
                    is_leaf=eqx.is_array,
                )

            return mask_fn


@dataclass
class TrainLmConfig:

    model_choice: str = field(default="gpt2")

    data: DatasetConfig = field(default_factory=DatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=Gpt2Config)
    optimizer: OptimizerConfigWithWeightDecay = field(
        default_factory=OptimizerConfigWithWeightDecay
    )

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    use_hf_model_config: bool = False

    fcm_prob: float = 0.0  # forgetful context masking prob. recommended 0.15

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000

    # additional config for Neural ODE
    time_embed_dim: int = 100
    sinusodial_dim: int = 16
    num_check_points: int = 2
    rank: int = 8
    alpha: float = 1.0
    num_blocks: int = 4
    multiplier: int = 2


def main(config: TrainLmConfig):
    logger.info(f"Training model {config.model_choice}")

    tokenizer = config.data.the_tokenizer

    if isinstance(config.model, HFCompatConfig):
        converter = config.model.default_hf_checkpoint_converter
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    config.trainer.initialize(config)

    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)

    Batch = config.trainer.TrainBatch
    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    def compute_loss(model: LmHeadModel, example: LmExample, key=None):
        return model.compute_loss(example, key=key).scalar()

    if config.model_choice == "gpt2" or config.model_choice == "llama":
        from dataclasses import replace

        new_optimizer = replace(
            config.optimizer,
            weight_decay_modules=r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings",
        )  # noqa
        config = replace(config, optimizer=new_optimizer)
    elif config.model_choice == "neuralode":
        from dataclasses import replace

        new_optimizer = replace(
            config.optimizer,
            weight_decay_modules=r".*time_embedding|.*token_embeddings|.*position_embeddings",
        )  # noqa
        config = replace(config, optimizer=new_optimizer)
    else:
        pass

    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    trainer = Trainer(config.trainer, optimizer, compute_loss)

    eval_dataset = CausalLmDataset(
        config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos
    )
    eval_loader = trainer.replicated_loader(eval_dataset, EvalBatch)
    train_dataset = CausalLmDataset(
        config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos
    )
    train_loader = iter(trainer.sharded_loader(train_dataset, Batch))

    with trainer.device_mesh:

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(
            Axis("vocab", vocab_size), parameter_axis_mapping
        )

        if vocab_size != Vocab.size:
            logger.info(
                f"Round vocab size from {vocab_size} to {Vocab.size} for partitioning"
            )

        if config.model_choice == "gpt2" or config.model_choice == "llama":
            model_init = lambda: config.model.build(Vocab, key=model_key)
        elif config.model_choice == "neuralode":

            def model_init():
                return NeuralOdeLMHeadModel.init(
                    Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    key=model_key,
                )

        elif config.model_choice == "llamaode":

            def model_init():
                return LlamaODELMHeadModel.init(
                    Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    key=model_key,
                )

        else:
            raise ValueError(f"Unknown model_choice {config.model_choice}")

        state = trainer.initial_state(
            training_key=train_key,
            # model_init=model_init,
            model_init=model_init,
        )

        wandb.summary["parameter_count"] = parameter_count(state.model)

        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size),
            every=1,
        )

        if state.step > 0:

            import tqdm

            for _ in tqdm.tqdm(
                range(state.step + 1),
                desc="finding where to resume",
            ):
                next(train_loader)

        trainer.train(state, train_loader)


if __name__ == "__main__":
    levanter.config.main(main)()
