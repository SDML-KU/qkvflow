import einops
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import levanter
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.data.text import CausalLmDataset
from levanter.models.gpt2 import Gpt2LMHeadModel
from levanter.trainer import Trainer
from levanter.utils.tree_utils import inference_mode
from matplotlib import colors

from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.train_lm import TrainLmConfig


def _get_weight(
    model: NeuralOdeLMHeadModel,
    t,
):
    """Get all weights given a time
    Return:
        - weights in order: query, key, value, output, mlp linear 1, mlp linear 2
    """
    if isinstance(model, NeuralOdeLMHeadModel):
        time_embed = model.transformer.block.time_embeding(t)
        qkv = model.transformer.block.attn.c_attn.f_W(time_embed)
        q, k, v = qkv.unbind("qkv")
        o = model.transformer.block.attn.c_proj.f_W(time_embed)
        ff1 = model.transformer.block.mlp.c_fc.f_W(time_embed)
        ff2 = model.transformer.block.mlp.c_proj.f_W(time_embed)
        return q, k, v, o, ff1, ff2
    elif isinstance(model, Gpt2LMHeadModel):
        qkv = model.transformer.blocks.stacked.attn.c_attn.weight
        q, k, v = qkv.unbind("qkv")
        o = model.transformer.blocks.stacked.attn.c_proj.weight
        ff1 = model.transformer.blocks.stacked.mlp.c_fc.weight
        ff2 = model.transformer.blocks.stacked.mlp.c_proj.weight

        def take(x):
            return x.take("layers", t)

        q, k, v, o, ff1, ff2 = map(take, (q, k, v, o, ff1, ff2))
        return q, k, v, o, ff1, ff2
    else:
        raise NotImplementedError()


def _get_all_weights(model):
    for t in range(model.transformer.config.num_layers):
        if isinstance(model, NeuralOdeLMHeadModel):
            yield _get_weight(model, jnp.asarray(t * model.transformer.dt))
        elif isinstance(model, Gpt2LMHeadModel):
            yield _get_weight(model, t)


def examine_QK_spectra_info(model: NeuralOdeLMHeadModel):
    """Get spectra info of QK

    Return:
        - Array: n_layers x n_heads x embed_size
    """
    n_layers = model.config.num_layers
    n_heads = model.config.num_heads
    embed_size = model.config.Embed.size

    eigen_values = jnp.empty((n_layers, n_heads, embed_size))

    layer_index = 0
    for q, k, _, _, _, _ in _get_all_weights(model):

        k = k.rename({"embed": "embed_key"})
        qk = hax.dot("head_size", q, k)

        for head_index in range(n_heads):
            qk_head = jax.device_put(qk.array[:, head_index, :], jax.devices("cpu")[0])
            eigen_v, _ = jnp.linalg.eig(qk_head)
            eigen_v = jnp.sort(eigen_v, axis=-1)
            # note that this will remove the imagine part of eigenvalue
            eigen_values = eigen_values.at[layer_index, head_index].set(eigen_v)

        layer_index += 1

    return eigen_values


def examine_OV_spectra_info(model: NeuralOdeLMHeadModel):

    n_layers = model.config.num_layers
    embed_size = model.config.Embed.size
    head = model.config.num_heads

    eigen_values = jnp.empty((n_layers, head, embed_size))

    layer_index = 0
    for _, _, v, o, _, _ in _get_all_weights(model):

        # w_o = einops.rearrange(
        #     o.array, "head head_size embed -> (head head_size) embed"
        # )
        w_o = o.array
        w_v = einops.rearrange(v.array, "embed head head_size -> head embed head_size")

        w_ov = jax.vmap(jnp.matmul)(w_v, w_o)
        w_ov = jax.device_put(w_ov, jax.devices("cpu")[0])
        eigen_val, _ = jnp.linalg.eig(w_ov)
        eigen_val = jnp.sort(eigen_val, axis=-1)
        # note that this will remove the imagine part of eigenvalue
        eigen_values = eigen_values.at[layer_index].set(eigen_val)

        layer_index += 1

    return eigen_values


def examine_FF_spectra_info(model: NeuralOdeLMHeadModel, rank: int = None, *, key):

    n_layers = model.config.num_layers
    embed_size = model.config.Embed.size

    eigen_values = jnp.empty((n_layers, embed_size))

    layer_index = 0

    if rank is not None:
        assert rank < model.config.mlp_scale * embed_size
        selected_index = jrandom.permutation(key=key, x=jnp.arange(embed_size))[:rank]

    for _, _, _, _, ff1, ff2 in _get_all_weights(model):

        ff1 = ff1.array
        ff2 = ff2.array

        if rank is None:
            m = ff1 @ ff2
        else:
            m = ff1[:, selected_index] @ ff2[selected_index, :]

        m = jax.device_put(m, jax.devices("cpu")[0])
        eigen_val, _ = jnp.linalg.eig(m)
        eigen_val = jnp.sort(eigen_val, axis=-1)
        # note that this will remove the imagine part of eigenvalue
        eigen_values = eigen_values.at[layer_index].set(eigen_val)

        layer_index += 1

    return eigen_values


def get_model(config: TrainLmConfig):
    tokenizer = config.data.the_tokenizer

    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)

    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    vocab_size = len(tokenizer)
    Vocab = round_axis_for_partitioning(
        Axis("vocab", vocab_size), parameter_axis_mapping
    )

    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    config.trainer.initialize(config)

    def compute_loss(model, example, key=None):
        return model.compute_loss(example, key=key).scalar()

    trainer = Trainer(config.trainer, optimizer, compute_loss)
    eval_dataset = CausalLmDataset(
        config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos
    )
    eval_loader = trainer.replicated_loader(eval_dataset, EvalBatch)
    trainer.add_default_hooks(eval_loader)
    if config.model_choice == "gpt2":
        model = Gpt2LMHeadModel.init(Vocab, config=config.model, key=model_key)
    elif config.model_choice == "neuralode":
        model = NeuralOdeLMHeadModel.init(Vocab, config=config.model, key=model_key)
    else:
        raise NotImplementedError()

    with trainer.device_mesh:
        state = trainer.initial_state(
            train_key,
            model=model,
        )

    return trainer, state.model, eval_loader, tokenizer


def create_red_colormap(num_values=50):
    """
    Creates a colormap with 50 shades of red, ranging from light to dark.

    Returns:
        A list of 50 hex color codes representing the red colormap.
    """

    # Create a linear colormap with red as the only color
    cmap = colors.LinearSegmentedColormap.from_list("", ["white", "red"])

    # Generate 50 evenly spaced values within the colormap
    norm = colors.Normalize(vmin=0, vmax=num_values - 1)
    colors_rgb = cmap(norm(range(num_values)))

    # Convert RGB values to hex color codes
    colors_hex = [colors.rgb2hex(rgb) for rgb in colors_rgb]

    return colors_hex


def visualize_lypunov_exponent(token_ids, lambds, tokenizer, n_bins=50):
    def find_historgram_bin(value, bins):
        if value < bins[0] or value >= bins[-1]:
            return None

        bin_index = np.digitize(value, bins) - 1
        return bin_index

    lambds = lambds.max(-1)

    _, bins = np.histogram(lambds, bins=n_bins)
    red_colors = create_red_colormap(num_values=n_bins)

    html = ""
    for token_id, lambd in zip(token_ids, lambds):
        word = tokenizer.decode(token_id)
        bin_index = find_historgram_bin(lambd, bins)
        if bin_index is None:
            bin_index = 0

        html += f"<span style='background-color: {red_colors[bin_index]}; color: black'>{word}</span>"  # noqa

    return html


if __name__ == "__main__":

    import numpy as np

    # -------------------------------------------------------------------
    config_path = "config/wikitext_103/small.yaml"
    checkpoint_path = "/home/anhth/project/qkvflow/checkpoints/r2xr231d"
    base_dir = "output/"
    save_folder = "wikitext_small_new"
    model_choice = "neuralode"
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Hinton server
    config_path = "config/wikitext_103/small.yaml"
    checkpoint_path = "/home/anhth/project/qkvflow/checkpoints/0hy66tsz"
    base_dir = "output/"
    save_folder = "wikitext_small_gpt"
    model_choice = "gpt2"
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Azure server
    config_path = "config/owt/medium.yaml"
    checkpoint_path = "checkpoints/yjt2bk0y"
    base_dir = "output/"
    save_folder = "owt_medium_gpt"
    model_choice = "gpt2"
    # -------------------------------------------------------------------

    # Hinton server
    # dataset does not matter when using wikitext instead of owt
    config_path = "config/wikitext_103/small.yaml"
    checkpoint_path = "checkpoints/0hy66tsz"
    base_dir = "output/"
    save_folder = "owt_small_gpt"
    model_choice = "gpt2"
    # -------------------------------------------------------------------

    # N = 50
    # n_examples = 4

    trainer, model, eval_loader, tokenizer = levanter.config.main(
        get_model,
        args=[
            "--config_path",
            config_path,
            "--model_choice",
            model_choice,
            "--trainer.load_checkpoint_path",
            checkpoint_path,
            "--trainer.wandb.mode",
            "disabled",
        ],
    )()

    # inferrence
    model = inference_mode(model, value=True)

    qk_eigen = examine_QK_spectra_info(model)
    ov_eigen = examine_OV_spectra_info(model)

    full_rank_mlp_eigen = examine_FF_spectra_info(model, key=None)

    rank_100_mlp_eigen = examine_FF_spectra_info(
        model, rank=100, key=jrandom.PRNGKey(0)
    )

    rank_500_mlp_eigen = examine_FF_spectra_info(
        model, rank=500, key=jrandom.PRNGKey(0)
    )

    qk_eigen = np.array(qk_eigen)
    ov_eigen = np.array(ov_eigen)
    full_rank_mlp_eigen = np.array(full_rank_mlp_eigen)
    rank_100_mlp_eigen = np.array(rank_100_mlp_eigen)
    rank_500_mlp_eigen = np.array(rank_500_mlp_eigen)

    np.savez_compressed(
        "output/spectra/owt_small_gpt.npz",
        qk_eigen=qk_eigen,
        ov_eigen=ov_eigen,
        full_rank_mlp_eigen=full_rank_mlp_eigen,
        rank_100_mlp_eigen=rank_100_mlp_eigen,
        rank_500_mlp_eigen=rank_500_mlp_eigen,
    )

    # computation bottle next is from the computing the whole sequence 1024
    # let's make a shorter sentence

    inputs = [
        "Robert Boulter is an English film, television and theatre actor. He had a guest-starring role on the television series",  # noqa
        "In 2000 Boulter had a guest-starring role on the television series",
        "He had a recurring role in 2003 on two episodes of The Bill, as character",
        "In 2006 Boulter starred in the play Citizenship written by",
    ]

    # for i, input in enumerate(inputs):

    #     tokens, lambds = compute_lyapunov_exponent_for_sentence(input, model,
    # tokenizer)

    #     tokens = np.array(tokens.array, dtype=tokens.dtype)
    #     lambds = np.array(lambds)

    #     try:
    #         # produce highlight text as html form
    #         html = visualize_lypunov_exponent(tokens, lambds, tokenizer)
    #     except Exception:
    #         print("Exception occurs!!!")
    #         html = " "

    #     save_dir = os.path.join(base_dir, save_folder)

    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     save_file = os.path.join(save_dir, f"lypunov_exponent_{i}.npz")

    #     # save data
    #     np.savez_compressed(
    #         save_file,
    #         tokens=tokens,
    #         lambds=lambds,
    #     )
    #     # save highlight text
    #     with open(os.path.join(save_dir, f"visualize_{i}.txt"), "w") as f:
    #         f.write(html)
    #         os.mkdir(save_dir)
    #     save_file = os.path.join(save_dir, f"lypunov_exponent_{i}.npz")

    #     # save data
    #     np.savez_compressed(
    #         save_file,
    #         tokens=tokens,
    #         lambds=lambds,
    #     )
    #     # save highlight text
    #     with open(os.path.join(save_dir, f"visualize_{i}.txt"), "w") as f:
    #         f.write(html)
