import math

import equinox as eqx
import haliax as hax
import jax.random as jrandom
from haliax import NamedArray


class SinusoidalPosEmb(eqx.Module):

    SinusodialDim: hax.Axis = eqx.static_field()

    @staticmethod
    def init(SinusodialDim: hax.Axis, *, key):
        assert SinusodialDim.size % 2 == 0
        return SinusoidalPosEmb(SinusodialDim)

    def __call__(self, x: NamedArray, max_period=10000, scale=1000, *, key=None):
        """Return [x, sin(w*x), cos(w*x)]"""

        # time interval solving differential is small
        # rescale it

        x = x * scale

        freqs = hax.exp(
            -math.log(max_period)
            * hax.arange(self.SinusodialDim, dtype=x.dtype)
            / self.SinusodialDim.size
        )

        x_axes = x.axes
        x_broadcasted = hax.broadcast_axis(x, axis=x_axes + (self.SinusodialDim,))
        freqs = hax.broadcast_axis(freqs, axis=x_axes + (self.SinusodialDim,))
        args = x_broadcasted * freqs  # * 2 * math.pi
        x = hax.broadcast_axis(x, axis=x_axes + (self.SinusodialDim.resize(1),))
        fouriered = hax.concatenate(
            axis=self.SinusodialDim.resize(self.SinusodialDim.size * 2 + 1),
            arrays=[x / scale, hax.sin(args), hax.cos(args)],
        )
        return fouriered


class LearnedSinusoidalPosEmb(eqx.Module):

    weights: NamedArray

    @staticmethod
    def init(SinusodialDim: hax.Axis, *, key):
        assert SinusodialDim.size % 2 == 0
        weights = hax.random.normal(key, shape=(SinusodialDim,))
        return LearnedSinusoidalPosEmb(weights)

    def __call__(self, x: NamedArray, *, key=None):
        """Return [x, sin(w*x), cos(w*x)]"""
        x_axes = x.axes
        weight_axis = self.weights.axes[0]
        x_broadcasted = hax.broadcast_axis(x, axis=x_axes + (weight_axis,))
        weights = hax.broadcast_axis(self.weights, axis=x_axes + (weight_axis,))
        freqs = x_broadcasted * weights * 2 * math.pi
        x = hax.broadcast_axis(x, axis=x_axes + (weight_axis.resize(1),))
        fouriered = hax.concatenate(
            axis=weight_axis.resize(weight_axis.size * 2 + 1),
            arrays=[x, hax.sin(freqs), hax.cos(freqs)],
        )
        return fouriered


class AlternativeTimeEmbeding(eqx.Module):
    """Alternative architecture that share Sinusoidal and MLP"""

    sinusodial: LearnedSinusoidalPosEmb
    lin1: hax.nn.Linear
    lin2: hax.nn.Linear
    dropout: hax.nn.Dropout

    @staticmethod
    def init(
        SinusodialDim: hax.Axis,
        TembedDim: hax.Axis,
        multiplier=1,
        dropout=0.1,
        learnable_sinusodial=False,
        *,
        key,
    ):
        sinusodial_key, lin1_key, lin2_key = jrandom.split(key, 3)
        if learnable_sinusodial:
            sinusodial = LearnedSinusoidalPosEmb.init(SinusodialDim, key=sinusodial_key)
        else:
            sinusodial = SinusoidalPosEmb.init(SinusodialDim, key=sinusodial_key)
        SinusodialDim = SinusodialDim.resize(SinusodialDim.size * 2 + 1)
        IntermediateDim = hax.Axis("IntermediateDim", TembedDim.size * multiplier)
        lin1 = hax.nn.Linear.init(
            In=SinusodialDim,
            Out=IntermediateDim,
            key=lin1_key,
        )

        lin2 = hax.nn.Linear.init(
            In=IntermediateDim,
            Out=TembedDim,
            key=lin2_key,
        )
        dropout = hax.nn.Dropout(dropout)

        return AlternativeTimeEmbeding(sinusodial, lin1, lin2, dropout)

    def __call__(self, t: NamedArray, *, key=None):

        # let's transform `t` into step function
        x = self.sinusodial(t, key=key)
        x = self.lin1(x)
        x = hax.nn.silu(x)
        x = self.lin2(x)
        return x


class ScaleShift(eqx.Module):

    lin: hax.nn.Linear

    @staticmethod
    def init(TembedDim: hax.Axis, Out: hax.Axis, *, key):
        ss = hax.Axis("ss", size=2)
        lin = hax.nn.Linear.init(
            In=TembedDim,
            Out=(Out, ss),
            key=key,
        )

        return ScaleShift(lin=lin)

    def __call__(self, x: NamedArray):
        x = hax.nn.silu(x)
        x = self.lin(x)
        scale, shift = x.unbind("ss")
        return scale, shift
