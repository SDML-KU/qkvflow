import einops
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.utils.tree_utils import inference_mode
from matplotlib import colors

from qkvflow.nn.dynamic import NeuralOdeLMHeadModel


def compute_jacobian(
    attn,
    time_embed: hax.NamedArray,
    x: hax.NamedArray,
    mask,
    j,
    Pos,
):
    axes = x.axes

    def fn(x):
        x = hax.NamedArray(x, axes=axes)
        f = attn(time_embed, x, mask, layer_idx=None, key=None)
        return f.take(axis=Pos, index=j).array

    output = jax.jacrev(fn)(x.array)
    return output


@eqx.filter_jit
def compute_lyapunov_exponent(
    model: NeuralOdeLMHeadModel,
    trajectory: hax.NamedArray,
    mask,
    output_index: int = -1,
):
    """
    Compute Lyapunov exponent that influent the output with `output_index`

    Returns:
        - N_steps x N_tokens x EmbedDim: trace of
    """

    Embed = model.config.Embed
    Pos = trajectory.resolve_axis("position")
    dt = model.transformer.dt

    # compute time embed
    t = hax.arange(model.config.Layers) * dt
    time_embed = model.transformer.time_embedding(t)

    q0 = jnp.eye(Embed.size)
    q0 = einops.repeat(q0, "n m -> k n m", k=Pos.size)
    q = q0

    lambdas = []
    for t in range(model.config.num_layers):

        x = trajectory.take(axis="layers", index=t)
        tembed = time_embed.take(axis="layers", index=t)

        J = compute_jacobian(
            model.transformer.block,
            tembed,
            x,
            mask=mask,
            j=output_index,
            Pos=model.config.Pos,
        )

        J = einops.rearrange(J, "n k m -> k n m")
        assert J.shape[0] == Pos.size

        # compute tangent matrix
        tangent = jax.vmap(jnp.matmul)(J, q)
        q = q + tangent * dt

        # QR decomposition and reassign `q`
        q, r = jnp.linalg.qr(q)

        # lyapunov will be the log diagonal part of `r`
        lambdas += [jnp.log(jnp.abs(jax.vmap(jnp.diag)(r)))]

    lambdas = jnp.stack(lambdas, axis=0)
    return lambdas


def compute_lyapunov_exponent_for_sentence(
    input,
    model,
    tokenizer,
):
    from levanter.data.text import AttentionMask

    tokens = tokenizer.encode(input)
    N = len(tokens)
    # wrap with NamedArray
    tokens = hax.NamedArray(
        jnp.array(tokens), axes=(model.config.Pos.resize(len(tokens)),)
    )

    # make sure inference mode
    model = inference_mode(model, True)
    token_embeddings = model.embeddings.embed(tokens, key=None)
    attn_mask = AttentionMask(is_causal=True)

    trajectory = model.transformer.compute_trajectory(token_embeddings, attn_mask)

    lambds = compute_lyapunov_exponent(
        model,
        trajectory,
        mask=attn_mask,
        output_index=N - 1,  # predict next word
    )

    return tokens, lambds


def kaplan_yorke_dimension(spectrum):
    """
    taken from:
    https://github.com/williamgilpin/dysts/blob/master/dysts/analysis.py
    """
    spectrum = np.sort(spectrum)[::-1]
    d = len(spectrum)
    cspec = np.cumsum(spectrum)
    j = np.max(np.where(cspec >= 0))
    if j > d - 2:
        j = d - 2
        print(
            "Cumulative sum of Lyapunov exponents never crosses zero."
            + "System may be ill-posed or undersampled."
        )
    dky = 1 + j + cspec[j] / np.abs(spectrum[j + 1])

    return dky


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
    # # a small test
    # config = Gpt2Config()
    # TembedDim = hax.Axis("TembedDim", 20)
    # x = hax.random.normal(key=jrandom.key(0), shape=(config.Pos, config.Embed))
    # time_embed = hax.random.normal(key=jrandom.key(0), shape=(TembedDim,))
    # Vocab = hax.Axis("Vocab", 5000)

    # model = NeuralOdeLMHeadModel.init(Vocab=Vocab,
    #                                 config=config,
    #                                 key=jrandom.key(0))
    # trajectory = model.transformer.compute_trajectory(x, None)

    # compute_lyapunov_exponent(
    #     model=model,
    #     trajectory=trajectory,
    #     mask=None
    # )
    import levanter

    from qkvflow.analysis.utils import get_model

    config_path = "config/wikitext_103/medium.yaml"
    checkpoint_path = "checkpoints/d1v5xney"
    base_dir = "output/"
    save_folder = "wikitext_small_new"
    model_choice = "neuralode"
    additional_args = [
        "--time_embed_dim",
        "48",
        "--sinusodial_dim",
        "512",
    ]

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
        ]
        + additional_args,
    )()

    input = "Robert Boulter is an English film, television and theatre actor. He had a guest-starring role on the television series"  # noqa

    tokens, lambdas = compute_lyapunov_exponent_for_sentence(
        input=input,
        model=model,
        tokenizer=tokenizer,
    )
    tokens = np.array(tokens.array, dtype=tokens.dtype)
    lambdas = np.array(lambdas)

    lambdas = jnp.sum(lambdas, axis=0)
    html = visualize_lypunov_exponent(tokens, lambdas, tokenizer)
    print(html)
