import logging
from dataclasses import dataclass, field

import jax
import jax.random as jrandom
import levanter
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.data.text import CausalLmDataset
from levanter.models.gpt2 import Gpt2Embeddings
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode

import wandb
from qkvflow.lora import is_lora_param, LoraConfig, loraize
from qkvflow.nn.dynamic_cp_v4 import NeuralOdeLMHeadModel
from qkvflow.train_lm import (
    DatasetConfig,
    OptimizerConfigWithWeightDecay,
    TrainLmConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class FinetuneLmConfig:

    mode: str = field(default="pretrain")
    finetune_choice: str = field(default="lora")
    finetune_layer: int = field(default=24)
    pretrain_config: TrainLmConfig = field(default=TrainLmConfig)
    finetune_data: DatasetConfig = field(default_factory=DatasetConfig)
    finetuner: TrainerConfig = field(default_factory=TrainerConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    lora_optimizer: OptimizerConfigWithWeightDecay = field(
        default_factory=OptimizerConfigWithWeightDecay,
    )


def run_or_load_pretrain(config: TrainLmConfig, compute_loss_fn):

    logger.info("\n \t \t \t === PRETRAIN STEP === \n")

    tokenizer = config.data.the_tokenizer
    config.trainer.initialize(config)

    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)

    Batch = config.trainer.TrainBatch
    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    trainer = Trainer(config.trainer, optimizer, compute_loss_fn)

    eval_dataset = CausalLmDataset(
        config.data.token_seq_dataset("test", Pos.size), Pos, KeyPos
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

        if config.model_choice == "neuralode":

            def model_init():
                return NeuralOdeLMHeadModel.init(
                    Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    key=model_key,
                )

        elif config.model_choice == "gpt2":
            model_init = lambda: config.model.build(Vocab, key=model_key)
        else:
            raise ValueError(f"Unknown model_choice {config.model_choice}")

        state = trainer.initial_state(
            training_key=train_key,
            model_init=model_init,
        )

        wandb.summary["parameter_count"] = parameter_count(state.model)

        trainer.add_default_hooks(eval_loader)

        if state.step == config.trainer.num_train_steps - 1:
            trained_state = state
        else:
            if state.step > 0:
                import tqdm

                for _ in tqdm.tqdm(
                    range(state.step + 1),
                    desc="finding where to resume",
                ):
                    next(train_loader)

            trained_state = trainer.train(state, train_loader)

    return trained_state.model


def main(config: FinetuneLmConfig):

    pretrain_config = config.pretrain_config

    def compute_loss(model: LmHeadModel, example: LmExample, key=None):
        return model.compute_loss(example, key=key).scalar()

    if config.mode == "pretrain":
        model = run_or_load_pretrain(
            config=pretrain_config, compute_loss_fn=compute_loss
        )
        return

    logger.info("\n \t \t \t === FINETUNE STEP === \n")
    tokenizer = config.finetune_data.the_tokenizer
    config.finetuner.initialize(config)

    seed = config.finetuner.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)

    Batch = config.finetuner.TrainBatch
    EvalBatch = config.finetuner.EvalBatch
    Pos = pretrain_config.model.Pos
    KeyPos = pretrain_config.model.KeyPos

    parameter_axis_mapping = config.finetuner.parameter_axis_mapping

    lora_optimizer = config.lora_optimizer.build(config.finetuner.num_train_steps)
    finetuner = Trainer(config.finetuner, lora_optimizer, compute_loss)

    finetune_eval_dataset = CausalLmDataset(
        config.finetune_data.token_seq_dataset("test", Pos.size),
        Pos,
        KeyPos,
    )
    finetune_eval_loader = finetuner.replicated_loader(finetune_eval_dataset, EvalBatch)
    finetune_train_dataset = CausalLmDataset(
        config.finetune_data.token_seq_dataset("train", Pos.size),
        Pos,
        KeyPos,
    )
    finetune_train_loader = iter(
        finetuner.sharded_loader(
            finetune_train_dataset,
            Batch,
        )
    )

    with finetuner.device_mesh:

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(
            Axis("vocab", vocab_size), parameter_axis_mapping
        )

        if vocab_size != Vocab.size:
            logger.info(
                f"Round vocab size from {vocab_size} to {Vocab.size} for partitioning"
            )

        def model_init():
            if pretrain_config.model_choice == "neuralode":
                return NeuralOdeLMHeadModel.init(
                    Vocab,
                    config=pretrain_config.model,
                    time_embed_dim=pretrain_config.time_embed_dim,
                    sinusodial_dim=pretrain_config.sinusodial_dim,
                    key=model_key,
                )
            elif pretrain_config.model_choice == "gpt2":
                return pretrain_config.model.build(Vocab, key=model_key)
            else:
                raise ValueError(
                    f"Unknown model choice : {pretrain_config.model_choice}"
                )

        logger.info(f" ---> Selected model: \t {pretrain_config.model_choice}")
        model = model_init()

        model, *_ = load_checkpoint(
            model, None, f"checkpoints/{pretrain_config.trainer.id}"
        )

        if pretrain_config.model_choice == "neuralode":
            # convert model to pure discrete
            logger.info(
                f"NeuralODE model will be finetuned with {config.finetune_layer} layers"
            )
            model = model.evaluate_at(dt=1.0 / float(config.finetune_layer))

        logger.info(f"Finetuning style: {config.finetune_choice}")
        if config.finetune_choice == "lora":
            # apply_lora
            model = loraize(model, config=config.lora, key=model_key)

            def filter_spec(node):
                # finetune both GPT2 embedding and LoRA
                return isinstance(node, Gpt2Embeddings) or is_lora_param(node)

            param_filter = jax.tree_util.tree_map(
                filter_spec, model, is_leaf=filter_spec
            )
        elif config.finetune_choice == "full":
            param_filter = None

        # correct `inference` in model after load
        model = inference_mode(model, False, "replace")

        if param_filter is not None:
            finetuner.is_trainable_param = param_filter

        state = finetuner.initial_state(
            training_key=train_key,
            model=model,
        )

        all_param_count = parameter_count(state.model)
        finetune_param_count = parameter_count(
            finetuner.trainable_params_only(state.model)
        )
        param_fraction = finetune_param_count * 1.0 / all_param_count

        wandb.summary["parameter_count"] = all_param_count
        wandb.summary["trainable_parameter_count"] = finetune_param_count
        logger.info(f"Total parameter count: {all_param_count}")
        logger.info(f"Trainable parameter count: {finetune_param_count}")
        logger.info(f"Fraction of parameters that are trainable: {param_fraction:.3f}")

        finetuner.add_default_hooks(finetune_eval_loader)
        finetuner.add_hook(
            callbacks.log_performance_stats(
                Pos.size, finetuner.config.train_batch_size
            ),
            every=1,
        )

        finetuner.train(state, train_loader=finetune_train_loader)


if __name__ == "__main__":
    levanter.config.main(main)()
