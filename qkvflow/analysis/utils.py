import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.models.gpt2 import Gpt2LMHeadModel
from levanter.trainer import Trainer

from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.train_lm import TrainLmConfig


def get_model(config: TrainLmConfig):
    tokenizer = config.data.the_tokenizer

    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)

    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    vocab_size = len(tokenizer)
    Vocab = round_axis_for_partitioning(
        Axis("vocab", vocab_size), parameter_axis_mapping
    )

    # EvalBatch = config.trainer.EvalBatch
    # Pos = config.model.Pos
    # KeyPos = config.model.KeyPos

    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    config.trainer.initialize(config)

    def compute_loss(model, example, key=None):
        return model.compute_loss(example, key=key).scalar()

    trainer = Trainer(config.trainer, optimizer, compute_loss)
    # eval_dataset = CausalLmDataset(
    #     config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos
    # )
    # eval_loader = trainer.replicated_loader(eval_dataset, EvalBatch)
    eval_loader = None
    trainer.add_default_hooks(None)
    if config.model_choice == "gpt2":
        model = Gpt2LMHeadModel.init(Vocab, config=config.model, key=model_key)
    elif config.model_choice == "neuralode":
        model = NeuralOdeLMHeadModel.init(
            Vocab,
            config=config.model,
            time_embed_dim=config.time_embed_dim,
            sinusodial_dim=config.sinusodial_dim,
            multiplier=config.multiplier,
            key=model_key,
        )
    else:
        raise NotImplementedError()

    with trainer.device_mesh:
        state = trainer.initial_state(
            train_key,
            model=model,
        )

    return trainer, state.model, eval_loader, tokenizer
