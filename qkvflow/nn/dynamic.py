# this one we randomly pick time step

import dataclasses
from collections.abc import Sequence
from typing import Callable, Dict, Optional

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from levanter.models.gpt2 import (
    ACT2FN,
    dot_product_attention,
    Gpt2Attention,
    Gpt2Config,
    Gpt2Embeddings,
    Gpt2Mlp,
)

from qkvflow.nn.time_embed import SinusoidalPosEmb, TimeEmbeding


class TemporalLinear(eqx.Module):
    """
    Time-dependent weights implementation

    This returns W(t)x
    """

    # MLP block contains two linear layers
    lin1: hnn.Linear
    lin2: hnn.Linear

    f_W: hnn.Linear
    f_b: Optional[NamedArray]

    In: hax.AxisSpec = eqx.field(static=True)
    Out: hax.AxisSpec = eqx.field(static=True)
    TembedDim: hax.AxisSpec = eqx.field(static=True)

    @staticmethod
    def init(
        SinusodialDim: hax.Axis,
        TembedDim: hax.Axis,
        In: hax.AxisSpec,
        Out: hax.AxisSpec,
        *,
        key,
        use_bias=True,
    ):
        k_lin1, k_lin2, key = jrandom.split(key, 3)
        if not isinstance(In, Sequence):
            In = (In,)
        if not isinstance(Out, Sequence):
            Out = (Out,)

        TembedDim_alias = TembedDim.alias("TembedDim_alias")
        lin1 = hnn.Linear.init(SinusodialDim, TembedDim_alias, key=k_lin1)
        lin2 = hnn.Linear.init(
            TembedDim_alias,
            TembedDim,
            key=k_lin2,
        )
        f_W = hnn.Linear.init(
            In=TembedDim,
            Out=In + Out,
            key=key,
            use_bias=True,
        )

        if use_bias:
            # zero init for bias
            # f_b = hax.zeros(shape=(TembedDim,) + Out)
            f_b = hax.random.normal(key=key, shape=(TembedDim,) + Out) * 1e-4
        else:
            f_b = None

        return TemporalLinear(lin1, lin2, f_W, f_b, In, Out, TembedDim)

    def __call__(
        self,
        time_embed: NamedArray,
        x: NamedArray,
        *,
        key=None,
    ):

        # MLP block
        time_embed = self.lin1(time_embed)
        time_embed = hnn.silu(time_embed)
        time_embed = self.lin2(time_embed)

        # Projection, `haliax` does not require to reshape
        W = self.f_W(time_embed)

        output = x.dot(self.In, W)
        output = hax.auto_sharded(output)

        if self.f_b is not None:
            # allow time-dependent bias
            b = time_embed.dot(self.TembedDim, self.f_b)
            output = output + b
            output = hax.auto_sharded(output)

        return output

    def evaluate_at(self, time_embed: NamedArray):

        # MLP block
        time_embed = self.lin1(time_embed)
        time_embed = hnn.silu(time_embed)
        time_embed = self.lin2(time_embed)

        # Projection
        W = self.f_W(time_embed)
        if self.f_b is not None:
            b = time_embed.dot(self.TembedDim, self.f_b)
        else:
            b = None

        return hnn.Linear(weight=W, bias=b, In=self.In, Out=self.Out)


class TemporalLayerNorm(eqx.Module):

    # MLP block
    lin1: hnn.Linear
    lin2: hnn.Linear
    axis: AxisSpec = eqx.field(static=True)
    f_weight: Optional[NamedArray]
    f_bias: Optional[NamedArray]

    time_embed_axis: AxisSpec = eqx.field(static=True)
    eps: float = eqx.field(static=True, default=1e-5)

    @staticmethod
    def init(
        axis: AxisSpec,
        eps: float = 1e-5,
        use_weight: bool = True,
        use_bias: bool = True,
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

        return TemporalLayerNorm(lin1, lin2, axis, f_weight, f_bias, TembedDim, eps)

    def __call__(self, time_embed, x):

        # MLP block
        time_embed = self.lin1(time_embed)
        time_embed = hnn.silu(time_embed)
        time_embed = self.lin2(time_embed)

        # time-dependent layer norm
        mean = x.mean(self.axis)
        var = x.var(self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = (x - mean) * inv

        if self.f_weight is not None:
            weight = hax.dot(self.time_embed_axis, time_embed, self.f_weight) + 1.0
            out = weight * out
        if self.f_bias is not None:
            bias = hax.dot(self.time_embed_axis, time_embed, self.f_bias)
            out = out + bias
        return out

    def evaluate_at(self, time_embed):

        weight, bias = None, None
        if self.f_weight is not None:
            weight = hax.dot(self.time_embed_axis, time_embed, self.f_weight) + 1.0
        if self.f_bias is not None:
            bias = hax.dot(self.time_embed_axis, time_embed, self.f_bias)

        return hnn.LayerNorm(axis=self.axis, weight=weight, bias=bias, eps=self.eps)


class Attention(eqx.Module):

    config: Gpt2Config = eqx.field(static=True)
    inference: bool = eqx.field(static=True)

    c_attn: TemporalLinear
    c_proj: TemporalLinear

    @staticmethod
    def init(config: Gpt2Config, SinusodialDim, TembedDim: hax.Axis, *, key):

        Qkv = hax.Axis("qkv", size=3)
        use_bias = config.use_bias
        Embed = config.Embed

        k_c, k_proj = jrandom.split(key, 2)
        c_attn = TemporalLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=Embed,
            Out=(Qkv, config.Heads, config.HeadSize),
            key=k_c,
            use_bias=use_bias,
        )
        c_proj = TemporalLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=(config.Heads, config.HeadSize),
            Out=Embed,
            key=k_proj,
            use_bias=use_bias,
        )

        return Attention(config=config, inference=False, c_attn=c_attn, c_proj=c_proj)

    @named_call
    def __call__(self, time_embed: NamedArray, x: NamedArray, mask, layer_idx, *, key):

        qkv_out = self.c_attn(time_embed, x).rearrange(
            (..., "qkv", "heads", "position", "head_size")
        )

        q, k, v = qkv_out.unbind("qkv")

        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        attn_output = dot_product_attention(
            QPos="position",
            KPos="key_position",
            Key="head_size",
            query=q,
            key=k,
            value=v,
            mask=mask,
            inference=self.inference,
            use_flash=self.config.use_flash_attention,
            flash_block_size=self.config.flash_attention_block_size,
            prng=key,
            attention_dtype=jnp.float32 if self.config.upcast_attn else None,
        )

        attn_output = self.c_proj(time_embed, attn_output)

        if self.config.upcast_attn:
            attn_output = attn_output.astype(x.dtype)

        return attn_output

    def evaluate_at(self, time_embed):

        c_attn = self.c_attn.evaluate_at(time_embed)
        c_proj = self.c_proj.evaluate_at(time_embed)

        return Gpt2Attention(
            config=self.config,
            c_attn=c_attn,
            c_proj=c_proj,
            inference=self.inference,
        )


class MLP(eqx.Module):

    config: Gpt2Config = eqx.field(static=True)

    c_fc: TemporalLinear
    c_proj: TemporalLinear
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(
        config: Gpt2Config,
        SinusodialDim,
        TembedDim: hax.Axis,
        *,
        key,
        use_bias: bool = True,
    ):

        k_fc, k_proj = jrandom.split(key, 2)
        Embed, Mlp, activation_fn = config.Embed, config.Mlp, config.activation_function
        c_fc = TemporalLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=Embed,
            Out=Mlp,
            key=k_fc,
            use_bias=use_bias,
        )
        c_proj = TemporalLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=Mlp,
            Out=Embed,
            key=k_proj,
            use_bias=use_bias,
        )
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn

        return MLP(config, c_fc, c_proj, act)

    @named_call
    def __call__(self, time_embed: NamedArray, x: NamedArray, *, key=None):
        del key

        x = self.c_fc(time_embed, x)
        x = self.act(x)

        x = self.c_proj(time_embed, x)
        return x

    def evaluate_at(self, time_embed):

        c_fc = self.c_fc.evaluate_at(time_embed)
        c_proj = self.c_proj.evaluate_at(time_embed)

        return Gpt2Mlp(c_fc=c_fc, c_proj=c_proj, act=self.act)


class _Block(eqx.Module):

    attn_ln: hnn.LayerNorm
    attn: Gpt2Attention
    mlp_ln: hnn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: hnn.Dropout

    def __call__(self, x, mask, layer_idx, *, key):
        k1, k2, k3, k4 = maybe_rng_split(key, 4)

        attn_output = self.attn(
            x=self.attn_ln(x),
            mask=mask,
            layer_idx=layer_idx,
            key=k1,
        )
        attn_output = self.resid_dropout(attn_output, key=k2)

        ff_output = self.mlp(
            x=self.mlp_ln(x),
            key=k3,
        )
        ff_output = self.resid_dropout(ff_output, key=k4)

        return attn_output + ff_output


class Block(eqx.Module):

    config: Gpt2Config = eqx.field(static=True)

    attn_ln: TemporalLayerNorm
    attn: Attention
    mlp_ln: TemporalLayerNorm
    mlp: MLP
    resid_dropout: hnn.Dropout

    @staticmethod
    def init(config: Gpt2Config, SinusodialDim, TembedDim, *, key):
        k_attn, k_mlp = jrandom.split(key)

        attn_ln = TemporalLayerNorm.init(
            config.Embed,
            eps=config.layer_norm_epsilon,
            use_bias=config.use_bias,
            TembedDim=TembedDim,
            SinusodialDim=SinusodialDim,
            key=key,
        )
        attn = Attention.init(config, SinusodialDim, TembedDim, key=k_attn)
        mlp_ln = TemporalLayerNorm.init(
            config.Embed,
            eps=config.layer_norm_epsilon,
            use_bias=config.use_bias,
            TembedDim=TembedDim,
            SinusodialDim=SinusodialDim,
            key=key,
        )
        mlp = MLP.init(
            config, SinusodialDim, TembedDim, key=k_mlp, use_bias=config.use_bias
        )
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)

        return Block(config, attn_ln, attn, mlp_ln, mlp, resid_dropout)

    def __call__(self, time_embed, x: NamedArray, mask, layer_idx, *, key):

        k1, k2, k3, k4 = maybe_rng_split(key, 4)

        attn_output = self.attn(
            time_embed=time_embed,
            x=self.attn_ln(time_embed, x),
            mask=mask,
            layer_idx=layer_idx,
            key=k1,
        )
        attn_output = self.resid_dropout(attn_output, key=k2)

        ff_output = self.mlp(
            time_embed=time_embed,
            x=self.mlp_ln(time_embed, x),
            key=k3,
        )
        ff_output = self.resid_dropout(ff_output, key=k4)

        return attn_output + ff_output

    def evaluate_at(self, time_embed):

        attn_ln = self.attn_ln.evaluate_at(time_embed)
        attn = self.attn.evaluate_at(time_embed)
        mlp_ln = self.mlp_ln.evaluate_at(time_embed)
        mlp = self.mlp.evaluate_at(time_embed)

        return _Block(
            attn_ln=attn_ln,
            attn=attn,
            mlp_ln=mlp_ln,
            mlp=mlp,
            resid_dropout=self.resid_dropout,
        )


def generate_t(axis, dt, dtype, key=None):
    if key is None:
        # inference
        # NOTE: old version index start from 0
        t = (hax.arange(axis, dtype=dtype) + 1) * dt
        dts = hax.ones((axis,), dtype=dtype) * dt
    else:
        # training
        key, key_t = jrandom.split(key)

        # randomize the whole interval
        # t = jrandom.uniform(key_t, (axis.size,), dtype)
        # t = jnp.sort(t)
        # dts = jnp.diff(t)
        # dts = jnp.concatenate([t[0][None], dts])
        # t = NamedArray(t, axes=(axis,))
        # dts = NamedArray(dts, axes=(axis,))

        # randomize dt0, dt1 is adapted to dt0 to make sure the interval will be [0, 1]
        dt0 = jrandom.uniform(key_t, shape=(), dtype=dtype) * dt
        t = dt0 + jnp.arange(axis.size - 1, dtype=dtype) * dt
        t = jnp.concatenate([t, jnp.ones((1,), dtype=dtype)])
        dts = jnp.diff(t)
        dts = jnp.concatenate([dt0[None], dts])
        t = NamedArray(t, axes=(axis,))
        dts = NamedArray(dts, axes=(axis,))

    return key, hax.auto_sharded(t), hax.auto_sharded(dts)


class _NeuralOdeTransformer(eqx.Module):

    config: Gpt2Config = eqx.field(static=True)

    blocks: Sequence[_Block]
    ln_f: hnn.LayerNorm

    dts: Sequence[float] = eqx.field(static=True)

    def __call__(self, x, mask, *, key):

        dts = self.dts.astype(x.dtype)

        for i, (block, dt) in enumerate(zip(self.blocks, dts)):
            key_i = jrandom.fold_in(key, i) if key is not None else None

            def do_block(x):
                output = block(x, mask, None, key=key_i)
                return x + output * dt

            x = jax.checkpoint(do_block, prevent_cse=False)(x)

        x = self.ln_f(x)

        return x


class NeuralOdeTransformer(eqx.Module):

    config: Gpt2Config = eqx.field(static=True)
    time_embedding: TimeEmbeding
    block: Block
    ln_f: hnn.LayerNorm

    dt: float = eqx.field(static=True)

    @staticmethod
    def init(
        config: Gpt2Config,
        time_embed_dim,
        sinusodial_dim,
        *,
        key,
    ):
        k_tembed, k_block = jrandom.split(key)
        TembedDim = hax.Axis("TembedDim", time_embed_dim)
        SinusodialDim = hax.Axis("SinusodialDim", sinusodial_dim)
        time_embeding = SinusoidalPosEmb.init(SinusodialDim, key=k_tembed)
        SinusodialDim = SinusodialDim.resize(sinusodial_dim * 2 + 1)

        block = Block.init(config, SinusodialDim, TembedDim, key=k_block)
        ln_f = hnn.LayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias
        )
        dt = 1.0 / config.num_layers

        return NeuralOdeTransformer(config, time_embeding, block, ln_f, dt)

    def __call__(self, x: NamedArray, attn_mask, *, key=None) -> NamedArray:

        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        dts = hax.ones((self.config.Layers,), dtype=x.dtype) * self.dt

        # key, t, dts = generate_t(self.config.Layers, self.dt, x.dtype, key)

        time_embed = self.time_embedding(t)

        if key is not None:
            keys = maybe_rng_split(key, self.config.num_layers)
        else:
            keys = None

        def do_block(x, time_embed, dt, key=None):
            output = self.block(time_embed, x, attn_mask, None, key=key)
            return x + output * dt

        # for scan operator, it is recommended to `prevent_cse=False`
        do_block = jax.checkpoint(do_block, prevent_cse=False)

        x = hax.fold(do_block, axis=self.config.Layers)(x, time_embed, dts, key=keys)
        x = self.ln_f(x)
        return x

    def compute_trajectory(self, x, attn_mask):

        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        time_embed = self.time_embedding(t)

        def do_block(x, time_embed):
            output = self.block(time_embed, x, attn_mask, None, key=None)
            ret = x + output * self.dt
            return ret, ret

        do_block = jax.checkpoint(do_block)

        _, trajectory = hax.scan(do_block, axis=self.config.Layers)(x, time_embed)
        return trajectory

    def evaluate_at(self, dt):
        """Here we evaluate on [0, 1] interval with step size `dt`"""

        dtype = self.ln_f.weight.dtype
        new_axis = self.config.Layers.resize(int(1.0 / dt))
        t = (hax.arange(new_axis, dtype=dtype) + 1) * dt
        dts = hax.ones((new_axis,), dtype=dtype) * dt
        time_embed: NamedArray = self.time_embedding(t)

        blocks = []
        for i in range(time_embed.axis_size("layers")):
            tb = time_embed.take("layers", i)
            block = self.block.evaluate_at(tb)
            blocks.append(block)

        return _NeuralOdeTransformer(
            config=self.config, blocks=blocks, ln_f=self.ln_f, dts=np.array(dts.array)
        )


class NeuralOdeLMHeadModel(eqx.Module):
    transformer: NeuralOdeTransformer
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.transformer.config

    @property
    def Vocab(self) -> hax.Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> hax.Axis:
        return self.config.Pos

    @classmethod
    def init(
        cls,
        Vocab: hax.Axis,
        config: Gpt2Config,
        time_embed_dim=100,
        sinusodial_dim=16,
        *,
        key,
    ) -> "NeuralOdeLMHeadModel":
        k_t, k_embeddings = jrandom.split(key, 2)
        transformer = NeuralOdeTransformer.init(
            config,
            time_embed_dim=time_embed_dim,
            sinusodial_dim=sinusodial_dim,
            key=k_t,
        )
        embeddings = Gpt2Embeddings.init(Vocab, config, key=k_embeddings)

        return NeuralOdeLMHeadModel(transformer, embeddings)

    def __call__(
        self, input_ids: NamedArray, attn_mask=None, *, key=None
    ) -> NamedArray:
        k_embed, k_transformer = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids, key=k_embed)
        x = self.transformer(x, attn_mask, key=k_transformer)
        lm_logits = self.embeddings.unembed(x)

        return lm_logits

    def compute_loss(
        self,
        example,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> NamedArray:
        logits = self(example.tokens, example.attn_mask, key=key)
        targets = hax.roll(example.tokens, -1, axis=self.Pos.name)
        target_y = hax.nn.one_hot(targets, self.Vocab, dtype=logits.dtype)
        return hnn.cross_entropy_loss(
            logits,
            self.Vocab,
            target_y,
            reduction,
            reduction_axis=reduction_axis,
            where=example.loss_mask,
        )

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    def resize_vocab(self, new_size: int, key=None) -> "NeuralOdeLMHeadModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": None, "embeddings": None}

    def evaluate_at(self, dt):

        transformer = self.transformer.evaluate_at(dt)

        return NeuralOdeLMHeadModel(
            transformer=transformer,
            embeddings=self.embeddings,
        )


if __name__ == "__main__":

    # TembedDim = hax.Axis("tb", 3)
    # In = hax.Axis("In", 4)
    # Out = hax.Axis("Out", 10)

    # lin = TemporalLinear.init(TembedDim, In, Out, key=jrandom.PRNGKey(0))
    # output = lin(
    #     hax.random.normal(key=jrandom.PRNGKey(1), shape=(TembedDim,)),
    #     hax.random.normal(key=jrandom.PRNGKey(2), shape=(In,)),
    # )
    # print(output)

    # config = Gpt2Config(seq_len=10, hidden_dim=4, num_heads=4)
    # time_embed = hax.random.normal(key=jrandom.PRNGKey(1), shape=(TembedDim,))
    # x = hax.random.normal(key=jrandom.PRNGKey(2), shape=(config.Pos, config.Embed))
    # attn = Attention.init(config, TembedDim, key=jrandom.PRNGKey(0))
    # attn(time_embed, x, None, None, key=jrandom.PRNGKey(3))

    # mlp = MLP.init(config, TembedDim, key=jrandom.PRNGKey(0))
    # mlp(time_embed, x, key=jrandom.PRNGKey(1))

    # block = Block.init(config, key=jrandom.PRNGKey(0))
    # block(jnp.array(0.1), x, None, None, key=jrandom.PRNGKey(1))

    # T = hax.Axis("T", 10)
    # key = None
    # key, t, dts = generate_t(T, 1.0 / T.size, key=key)
    # print(t)
    # print(dts)

    from levanter.utils.tree_utils import inference_mode

    config = Gpt2Config()
    Vocab = hax.Axis("vocab", 1000)
    model = NeuralOdeTransformer.init(
        config,
        time_embed_dim=100,
        sinusodial_dim=16,
        key=jrandom.key(0),
    )
    model = inference_mode(model, True)
    x = hax.random.normal(key=jrandom.key(0), shape=(config.Pos, config.Embed))
    output1 = model(x, None, key=None)

    # when the time step is unchanged
    model_2 = model.evaluate_at(dt=model.dt)
    output2 = model_2(x, None, key=None)

    assert jnp.allclose(output1.array, output2.array, rtol=1e-2, atol=1e-2)

    # when the time step is changed
    model_3 = model.evaluate_at(dt=0.1)
    output3 = model_3(x, None, key=None)

    assert not jnp.allclose(output1.array, output3.array, rtol=1e-2, atol=1e-2)

    Vocab = hax.Axis("vocab", 1000)
    model = NeuralOdeLMHeadModel.init(Vocab, config, key=jrandom.key(0))
    model = inference_mode(model, True)

    x = hax.random.randint(
        key=jrandom.key(0), shape=(config.Pos,), minval=0, maxval=999
    )

    output1 = model(x, None, key=None)

    # the case the does not change time step
    model_2 = model.evaluate_at(dt=1.0 / config.num_layers)
    output2 = model_2(x, None, key=None)

    assert jnp.allclose(output1.array, output2.array, rtol=1e-2, atol=1e-2)

    # time step is changed
    model_3 = model.evaluate_at(dt=0.1)
    output3 = model_3(x, None, key=None)

    assert not jnp.allclose(output1.array, output3.array, rtol=1e-2, atol=1e-2)

    from levanter.lora import LoraConfig, loraize

    lora_config = LoraConfig(target_modules="")
    lora_model = loraize(model=model_3, config=lora_config, key=jrandom.key(0))

    print(lora_model)

    # print(lora_trainable_params_filter(lora_model))
