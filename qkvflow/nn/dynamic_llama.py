"""
Adapted from:

https://github.com/stanford-crfm/levanter/blob/main/src/levanter/models/llama.py
"""

import dataclasses
from typing import Callable, Dict, Optional, Tuple, Union

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from jaxtyping import PRNGKeyArray
from levanter.compat.torch_serialization import (
    apply_prefix,
    flatten_linear_layers,
    StateDict,
    StateDictSerializationMixin,
    unflatten_linear_layers,
)
from levanter.models.attention import AttentionMask, dot_product_attention
from levanter.models.gpt2 import ACT2FN
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmHeadModel

from qkvflow.nn.dynamic import TemporalLinear
from qkvflow.nn.time_embed import SinusoidalPosEmb


class LlamaMlp(eqx.Module):
    """Multi-layer Perceptron
    In comparison with GPT2, LlamaMlp adds an up-proj
    that multiplies with activated gate_proj,
    before down-proj.
    """

    # gate_proj: hnn.Linear  # projection from Embed to Mlp
    # up_proj: hnn.Linear  # projection from Embed to Mlp
    # down_proj: hnn.Linear  # projection from Mlp to Embed
    gate_proj: TemporalLinear
    up_proj: TemporalLinear
    down_proj: TemporalLinear
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        SinusoidalDim,
        TembedDim,
        Embed: Axis,
        Mlp: Axis,
        activation_fn: Union[str, Callable],
        *,
        key,
        use_bias: bool = False,
    ) -> "LlamaMlp":
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = TemporalLinear.init(
            SinusodialDim=SinusoidalDim,
            TembedDim=TembedDim,
            Out=Mlp,
            In=Embed,
            key=k_fc,
            use_bias=use_bias,
        )
        up_proj = TemporalLinear.init(
            SinusodialDim=SinusoidalDim,
            TembedDim=TembedDim,
            Out=Mlp,
            In=Embed,
            key=k_up_proj,
            use_bias=use_bias,
        )
        down_proj = TemporalLinear.init(
            SinusodialDim=SinusoidalDim,
            TembedDim=TembedDim,
            Out=Embed,
            In=Mlp,
            key=k_down_proj,
            use_bias=use_bias,
        )
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore
        return LlamaMlp(gate_proj, up_proj, down_proj, act)

    @named_call
    def __call__(self, time_embed, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.gate_proj(time_embed, x, key=k_gate)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(time_embed, x, key=k_up)
        outputs = self.down_proj(time_embed, hidden_states, key=k_down)
        return outputs


class LlamaRotaryEmbedding(eqx.Module, StateDictSerializationMixin):
    Pos: Axis = eqx.field(static=True)
    cos_cached: NamedArray
    sin_cached: NamedArray

    def __init__(self, HeadSize: Axis, Pos: Axis, base: int = 10000):
        self.Pos = Pos
        # this must be compile-time b/c we want to store them in a static field
        with jax.ensure_compile_time_eval():
            self.cos_cached, self.sin_cached = self._get_cos_sin_cache(
                Pos=Pos, HeadSize=HeadSize, base=base
            )

    @staticmethod
    def _get_cos_sin_cache(
        HeadSize: hax.Axis, Pos: hax.Axis, base: float
    ) -> Tuple[NamedArray, NamedArray]:
        HeadHalfSize = HeadSize.resize(HeadSize.size // 2)
        inv_freq: NamedArray = 1.0 / (
            base ** (hax.arange(HeadHalfSize, step=2) / HeadSize.size)
        )

        position_ids: NamedArray = hax.arange(Pos)

        freqs = position_ids * inv_freq.broadcast_axis(Pos)
        # This is different from the paper but aligns with HF implementation:
        # It uses a different permutation in order to obtain the same calculation
        emb = hax.concatenate(HeadSize, (freqs, freqs))
        cos_cached = hax.cos(emb)
        sin_cached = hax.sin(emb)
        # This is different from the paper but aligns with HF implementation:
        return cos_cached, sin_cached

    def __call__(self, seq_len: int) -> Tuple[NamedArray, NamedArray]:
        return jax.lax.stop_gradient(
            (
                self.cos_cached[self.Pos, :seq_len],
                self.sin_cached[self.Pos, :seq_len],
            )
        )

    # if we do that, consider moving the key remapping stuff there too?
    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        return self

    def update_state_dict(
        self, state_dict: StateDict, prefix: Optional[str] = None
    ) -> StateDict:
        return state_dict


class LlamaAttention(eqx.Module):
    config: LlamaConfig = eqx.static_field()
    q_proj: hnn.Linear  # projection from Embed to query
    k_proj: hnn.Linear  # projection from Embed to key
    v_proj: hnn.Linear  # projection from Embed to value
    o_proj: hnn.Linear  # projection from Heads to output
    rotary_emb: LlamaRotaryEmbedding  # rotary embedding

    @staticmethod
    def init(config: LlamaConfig, SinusoidalDim, TembedDim, *, key) -> "LlamaAttention":
        use_bias = config.use_bias
        Embed = config.Embed

        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        q_proj = TemporalLinear.init(
            SinusodialDim=SinusoidalDim,
            TembedDim=TembedDim,
            In=Embed,
            Out=(config.Heads, config.HeadSize),
            key=k_q,
            use_bias=use_bias,
        )
        k_proj = TemporalLinear.init(
            SinusodialDim=SinusoidalDim,
            TembedDim=TembedDim,
            In=Embed,
            Out=(config.Heads, config.HeadSize),
            key=k_k,
            use_bias=use_bias,
        )
        v_proj = TemporalLinear.init(
            SinusodialDim=SinusoidalDim,
            TembedDim=TembedDim,
            In=Embed,
            Out=(config.Heads, config.HeadSize),
            key=k_v,
            use_bias=use_bias,
        )
        o_proj = TemporalLinear.init(
            SinusodialDim=SinusoidalDim,
            TembedDim=TembedDim,
            In=(config.Heads, config.HeadSize),
            Out=Embed,
            key=k_o,
            use_bias=use_bias,
        )
        rotary_emb = LlamaRotaryEmbedding(config.HeadSize, config.Pos)
        return LlamaAttention(config, q_proj, k_proj, v_proj, o_proj, rotary_emb)

    @named_call
    def __call__(
        self,
        time_embed,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        *,
        key=None,
    ) -> NamedArray:
        key_q, key_k, key_v, key_o = maybe_rng_split(key, 4)

        # reorder heads and position for better training throughput
        q = self.q_proj(time_embed, x, key=key_q).rearrange(
            (..., "heads", "position", "head_size")
        )
        k = self.k_proj(time_embed, x, key=key_k).rearrange(
            (..., "heads", "position", "head_size")
        )
        v = self.v_proj(time_embed, x, key=key_v).rearrange(
            (..., "heads", "position", "head_size")
        )

        cos, sin = self.rotary_emb(seq_len=x.axis_size("position"))

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if self.config.upcast_attn:
            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        c = self.config
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            use_flash=c.use_flash_attention,
            flash_block_size=c.flash_attention_block_size,
        )

        if self.config.upcast_attn:
            attn_output = attn_output.astype(x.dtype)

        attn_output = self.o_proj(time_embed, attn_output, key=key_o)
        return attn_output


class LlamaRMSNorm(eqx.Module):
    """
    Similar to LayerNorm, but uses the RMS of the input along
    the specified axis (or axes) instead of variance.
    """

    lin1: hnn.Linear
    lin2: hnn.Linear

    axis: AxisSpec = eqx.static_field()
    f_weight: Optional[NamedArray]
    f_bias: Optional[NamedArray]

    time_embed_axis: AxisSpec = eqx.field(static=True)
    eps: float = eqx.static_field(default=1e-5)
    dtype: Optional[jnp.dtype] = eqx.static_field(default=jnp.float32)

    @staticmethod
    def init(
        axis: AxisSpec,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = True,
        dtype=jnp.float32,
        *,
        SinusodialDim: Axis,
        TembedDim: Axis,
        key,
    ):
        k_lin1, k_lin2 = jrandom.split(key)
        TembedDim_alias = TembedDim.alias("TembedDim_alias")
        lin1 = hnn.Linear.init(SinusodialDim, TembedDim_alias, key=k_lin1)
        lin2 = hnn.Linear.init(TembedDim_alias, TembedDim, key=k_lin2)

        if use_weight:
            f_weight = hax.zeros(hax.concat_axes(axis, TembedDim))
        else:
            f_weight = None
        if use_bias:
            f_bias = hax.zeros(hax.concat_axes(axis, TembedDim))
        else:
            f_bias = None

        return LlamaRMSNorm(lin1, lin2, axis, f_weight, f_bias, TembedDim, eps, dtype)

    def __call__(self, time_embed, x: NamedArray) -> NamedArray:
        # This gives a different result than jnp.var(), which is
        # defined as the average of the squared deviations from the mean

        # MLP block
        time_embed = self.lin1(time_embed)
        time_embed = hnn.silu(time_embed)
        time_embed = self.lin2(time_embed)

        in_dtype = x.dtype
        x = x.astype(self.dtype)
        var = hax.mean(hax.square(x), axis=self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = x * inv
        out = out.astype(in_dtype)

        if self.f_weight is not None:
            weight = hax.dot(self.time_embed_axis, time_embed, self.f_weight) + 1.0
            out = weight * out
        if self.f_bias is not None:
            bias = hax.dot(self.time_embed_axis, time_embed, self.f_bias)
            out = out + bias

        # second cast in case params are in float32
        return out.astype(in_dtype)


class LlamaDecoderLayer(eqx.Module):
    config: LlamaConfig = eqx.static_field()
    self_attn: LlamaAttention
    mlp: LlamaMlp
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm

    @staticmethod
    def init(
        config: LlamaConfig, SinusoidalDim, TembedDim, *, key
    ) -> "LlamaDecoderLayer":
        k_attn, k_mlp, ln_1_key, ln_2_key = jrandom.split(key, 4)

        attn = LlamaAttention.init(config, SinusoidalDim, TembedDim, key=k_attn)
        mlp = LlamaMlp.init(
            SinusoidalDim,
            TembedDim,
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = LlamaRMSNorm.init(
            config.Embed, SinusodialDim=SinusoidalDim, TembedDim=TembedDim, key=ln_1_key
        )
        ln_2 = LlamaRMSNorm.init(
            config.Embed, SinusodialDim=SinusoidalDim, TembedDim=TembedDim, key=ln_2_key
        )

        return LlamaDecoderLayer(config, attn, mlp, ln_1, ln_2)

    @named_call
    def __call__(
        self,
        time_embed,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        *,
        key=None,
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        x = self.input_layernorm(time_embed, x)
        attn_output = self.self_attn(time_embed, x=x, mask=mask, key=k_attn)
        x = self.post_attention_layernorm(time_embed, x)
        mlp_output = self.mlp(time_embed, x, key=k_mlp)
        output = attn_output + mlp_output
        return output


class LlamaRMSNorm_(hnn.LayerNorm):
    """It is a modified version of LayerNorm.
    The main changes are:
    1. The variance is defined as the average of square, versus the original
    definition as the average of the squared deviations from the mean.
    2. The output is defined as x * inv, without minusing the mean.
    3. The default value of eps is set to 1e-6 and use_bias to False.
    """

    @staticmethod
    def init(
        axis: AxisSpec,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
    ):
        if use_weight:
            weight = hax.ones(axis)
        else:
            weight = None
        if use_bias:
            bias = hax.zeros(axis)
        else:
            bias = None

        return LlamaRMSNorm_(axis, weight, bias, eps)

    def __call__(self, x: NamedArray) -> NamedArray:
        # This gives a different result than jnp.var(), which is
        # defined as the average of the squared deviations from the mean
        var = hax.mean(hax.square(x), axis=self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = x * inv

        if self.weight is not None:
            out = self.weight * out
        if self.bias is not None:
            out = out + self.bias
        return out


class LlamaTransformer(StateDictSerializationMixin, eqx.Module):
    config: LlamaConfig = eqx.static_field()
    time_embedding: SinusoidalPosEmb
    block: LlamaDecoderLayer
    norm: LlamaRMSNorm

    dt: float = eqx.field(static=True)

    @staticmethod
    def init(
        config: LlamaConfig,
        time_embed_dim,
        sinusodial_dim,
        *,
        key,
    ) -> "LlamaTransformer":
        k_tembed, k_block = jrandom.split(key)
        TembedDim = hax.Axis("TembedDim", time_embed_dim)
        SinusodialDim = hax.Axis("SinusodialDim", sinusodial_dim)
        time_embeding = SinusoidalPosEmb.init(SinusodialDim, key=k_tembed)
        SinusodialDim = SinusodialDim.resize(sinusodial_dim * 2 + 1)

        block = LlamaDecoderLayer.init(config, SinusodialDim, TembedDim, key=k_block)
        ln_f = LlamaRMSNorm_.init(
            config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias
        )
        dt = 1.0 / config.num_layers
        return LlamaTransformer(config, time_embeding, block, ln_f, dt)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray], *, key
    ) -> NamedArray:

        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        dts = hax.ones((self.config.Layers,), dtype=x.dtype) * self.dt

        # key, t, dts = generate_t(self.config.Layers, self.dt, x.dtype, key)

        time_embed = self.time_embedding(t)

        if key is not None:
            keys = maybe_rng_split(key, self.config.num_layers)
        else:
            keys = None

        def do_block(x, time_embed, dt, key=None):
            output = self.block(time_embed, x, attn_mask, key=key)
            return x + output * dt

        # for scan operator, it is recommended to `prevent_cse=False`
        do_block = jax.checkpoint(do_block, prevent_cse=False)

        x = hax.fold(do_block, axis=self.config.Layers)(x, time_embed, dts, key=keys)

        x = self.norm(x)

        return x


class LlamaEmbedding(StateDictSerializationMixin, eqx.Module):
    """Similar to GPT2 Embedding, except that:
    - Llama doesn't have position embedding in the Embedding layer.
    - Llama doesn't use dropout.
    """

    Vocab: Axis = eqx.static_field()
    config: LlamaConfig = eqx.static_field()
    token_embeddings: NamedArray

    @staticmethod
    def init(Vocab: Axis, config: LlamaConfig, *, key) -> "LlamaEmbedding":
        k_wte = jrandom.split(key, 1)

        token_embeddings = hax.random.normal(k_wte, (Vocab, config.Embed))
        return LlamaEmbedding(Vocab, config, token_embeddings)

    @named_call
    def embed(self, input_ids, *args):
        input_embeds = self.token_embeddings.take("vocab", input_ids)
        x = input_embeds
        return x

    def unembed(self, x: NamedArray):
        return hax.dot("embed", x, self.token_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "model.embed_tokens.weight"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = hax.tree_util.resize_axis(
            self.token_embeddings, self.Vocab, new_size, key=key
        )
        return dataclasses.replace(
            self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_weights
        )


class LlamaLMHeadModel(
    eqx.Module, LmHeadModel[LlamaConfig], StateDictSerializationMixin
):
    transformer: LlamaTransformer
    embeddings: LlamaEmbedding
    lm_head: hnn.Linear

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: LlamaConfig,
        time_embed_dim=100,
        sinusodial_dim=16,
        *,
        key,
    ) -> "LlamaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = LlamaTransformer.init(
            config,
            time_embed_dim=time_embed_dim,
            sinusodial_dim=sinusodial_dim,
            key=k_t,
        )
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False)
        return LlamaLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        *,
        key=None,
    ) -> NamedArray:
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t)
        lm_logits = self.lm_head(x, key=k_head)
        return lm_logits

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[LlamaConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        new_lm_matrix = hax.tree_util.resize_axis(
            self.lm_head.weight, self.Vocab, new_size, key=k2
        )
        new_lm_head = dataclasses.replace(
            self.lm_head, Out=new_Vocab, weight=new_lm_matrix
        )

        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # unflatten the linear layers of HF state_dict to match the shape of LlamaMlp
        d = state_dict.copy()
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "lm_head"),
                state_dict,
                self.lm_head,
                out_dims_first_in_dict=True,
            )
        )
        return super().from_state_dict(d, prefix)

    def update_state_dict(
        self, state_dict: StateDict, prefix: Optional[str] = None
    ) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix=prefix)

        my_dict.update(
            flatten_linear_layers(
                apply_prefix(prefix, "lm_head"),
                self.lm_head,
                out_dims_first_in_dict=True,
            )
        )

        state_dict.update(my_dict)
        return state_dict


def _rotate_half(x: NamedArray) -> NamedArray:
    """Rotates half of the hidden dims of
    the input and concatenates them.
    """
    HeadSize = x.axes[-1]
    x1 = x[HeadSize, : HeadSize.size // 2]
    x2 = x[HeadSize, HeadSize.size // 2 :]  # noqa
    out = hax.concatenate(HeadSize, (-x2, x1))
    return out


def _apply_rotary_pos_emb(
    q: NamedArray,  # [batch, position, heads, head_size]
    k: NamedArray,  # [batch, position, kv_heads, head_size]
    cos: NamedArray,  # [position, head_size]
    sin: NamedArray,  # [position, head_size]
) -> Tuple[NamedArray, NamedArray]:
    """Applies rotary position embedding to q and k."""
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


if __name__ == "__main__":

    config = LlamaConfig()
    Vocab = hax.Axis("vocab", 1000)
    Batch = hax.Axis("batch", 8)

    model = LlamaLMHeadModel.init(
        Vocab=Vocab,
        config=config,
        time_embed_dim=20,
        sinusodial_dim=16,
        key=jrandom.PRNGKey(0),
    )

    x = hax.random.randint(
        jrandom.PRNGKey(0), (Batch, config.Pos), minval=0, maxval=Vocab.size
    )
    model(x)
