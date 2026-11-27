"""
Neural network building blocks (Flax Linen) shared by the policies.

The layers reproduce the computations and default initializers of the Haiku layers used by
earlier versions of social-jym, so that parameters trained with those versions give the same
outputs. Parameters saved by the Haiku-based versions are converted automatically by
`Network.apply` (see `haiku_to_flax`).
"""
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

Array = jax.Array
Initializer = Callable[..., Array]

# --------------------------------------------------------------------------------------
# Initializers (same distributions as the Haiku initializers of the same name)
# --------------------------------------------------------------------------------------

def _compute_fans(shape: Sequence[int]) -> tuple[float, float]:
    if len(shape) < 1:
        return 1., 1.
    if len(shape) == 1:
        return float(shape[0]), float(shape[0])
    if len(shape) == 2:
        return float(shape[0]), float(shape[1])
    receptive_field_size = float(np.prod(shape[:-2]))
    return shape[-2] * receptive_field_size, shape[-1] * receptive_field_size


def truncated_normal(stddev: float = 1., mean: float = 0., lower: float = -2., upper: float = 2.) -> Initializer:
    """Normal distribution truncated to [lower, upper] standard deviations (no variance correction)."""
    def init(key, shape, dtype=jnp.float32):
        unscaled = jax.random.truncated_normal(key, lower, upper, shape, dtype)
        return lax.convert_element_type(stddev, dtype) * unscaled + lax.convert_element_type(mean, dtype)
    return init


def random_normal(stddev: float = 1., mean: float = 0.) -> Initializer:
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * stddev + mean
    return init


def random_uniform(minval: float = 0., maxval: float = 1.) -> Initializer:
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval, maxval)
    return init


def variance_scaling(scale: float = 1., mode: str = "fan_in", distribution: str = "truncated_normal") -> Initializer:
    """
    Variance scaling initializer. Unlike `flax.linen.initializers.variance_scaling`, it also
    accepts 1D shapes (e.g. biases), for which fan_in = fan_out = shape[0].
    """
    assert mode in ("fan_in", "fan_out", "fan_avg"), f"Invalid mode {mode}"
    assert distribution in ("truncated_normal", "normal", "uniform"), f"Invalid distribution {distribution}"
    def init(key, shape, dtype=jnp.float32):
        fan_in, fan_out = _compute_fans(shape)
        if mode == "fan_in":
            s = scale / max(1., fan_in)
        elif mode == "fan_out":
            s = scale / max(1., fan_out)
        else:
            s = scale / max(1., (fan_in + fan_out) / 2.)
        if distribution == "truncated_normal":
            stddev = np.sqrt(s) / np.asarray(.87962566103423978, dtype=dtype)
            return truncated_normal(stddev=stddev)(key, shape, dtype)
        if distribution == "normal":
            return random_normal(stddev=np.sqrt(s))(key, shape, dtype)
        limit = np.sqrt(3. * s)
        return random_uniform(minval=-limit, maxval=limit)(key, shape, dtype)
    return init


def orthogonal(scale: float = 1., axis: int = -1) -> Initializer:
    def init(key, shape, dtype=jnp.float32):
        if len(shape) < 2:
            raise ValueError("Orthogonal initializer requires at least a 2D shape.")
        n_rows = shape[axis]
        n_cols = int(np.prod(shape)) // n_rows
        matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
        q_mat, r_mat = jnp.linalg.qr(jax.random.normal(key, matrix_shape, dtype))
        q_mat = q_mat * jnp.sign(jnp.diag(r_mat))
        if n_rows < n_cols:
            q_mat = q_mat.T
        q_mat = jnp.reshape(q_mat, (n_rows,) + tuple(np.delete(np.asarray(shape), axis)))
        q_mat = jnp.moveaxis(q_mat, 0, axis)
        return lax.convert_element_type(scale, dtype) * q_mat
    return init


def constant(value: Any) -> Initializer:
    def init(key, shape, dtype=jnp.float32):
        return jnp.broadcast_to(jnp.asarray(value), shape).astype(dtype)
    return init


def zeros(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype)


def ones(key, shape, dtype=jnp.float32):
    return jnp.ones(shape, dtype)

# --------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------

class Linear(nn.Module):
    """Affine layer `x @ kernel + bias`. Default kernel init: truncated normal with stddev 1/sqrt(fan_in)."""
    features: int
    with_bias: bool = True
    w_init: Optional[Initializer] = None
    b_init: Optional[Initializer] = None

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs)
        input_size = inputs.shape[-1]
        w_init = self.w_init if self.w_init is not None else truncated_normal(stddev=1. / np.sqrt(input_size))
        kernel = self.param("kernel", w_init, (input_size, self.features), inputs.dtype)
        out = jnp.dot(inputs, kernel)
        if self.with_bias:
            bias = self.param("bias", self.b_init if self.b_init is not None else zeros, (self.features,), inputs.dtype)
            out = out + jnp.broadcast_to(bias, out.shape)
        return out


class MLP(nn.Module):
    """Sequence of `Linear` layers (named linear_0, linear_1, ...) with an activation in between."""
    output_sizes: Sequence[int]
    w_init: Optional[Initializer] = None
    b_init: Optional[Initializer] = None
    with_bias: bool = True
    activation: Callable[[Array], Array] = jax.nn.relu
    activate_final: bool = False

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        out = inputs
        n_layers = len(self.output_sizes)
        for i, size in enumerate(self.output_sizes):
            out = Linear(size, with_bias=self.with_bias, w_init=self.w_init, b_init=self.b_init, name=f"linear_{i}")(out)
            if i < n_layers - 1 or self.activate_final:
                out = self.activation(out)
        return out


class LayerNorm(nn.Module):
    """Layer normalization over the last axis, computed as `scale * rsqrt(var + eps) * (x - mean) + bias`."""
    eps: float = 1e-5
    create_scale: bool = True
    create_offset: bool = True

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs)
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)
        param_shape = (inputs.shape[-1],)
        scale = self.param("scale", ones, param_shape, inputs.dtype) if self.create_scale else np.array(1., dtype=inputs.dtype)
        offset = self.param("bias", zeros, param_shape, inputs.dtype) if self.create_offset else np.array(0., dtype=inputs.dtype)
        eps = lax.convert_element_type(self.eps, variance.dtype)
        inv = scale * lax.rsqrt(variance + eps)
        return inv * (inputs - mean) + offset


class Conv1D(nn.Module):
    """1D convolution over inputs shaped [N, W, C] or [W, C]. Default kernel init: truncated normal with stddev 1/sqrt(fan_in)."""
    output_channels: int
    kernel_shape: int
    stride: int = 1
    padding: str = "SAME"
    with_bias: bool = True
    w_init: Optional[Initializer] = None
    b_init: Optional[Initializer] = None

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs)
        unbatched = inputs.ndim == 2
        if unbatched:
            inputs = jnp.expand_dims(inputs, axis=0)
        w_shape = (self.kernel_shape, inputs.shape[-1], self.output_channels)
        w_init = self.w_init if self.w_init is not None else truncated_normal(stddev=1. / np.sqrt(np.prod(w_shape[:-1])))
        kernel = self.param("kernel", w_init, w_shape, inputs.dtype)
        out = lax.conv_general_dilated(
            inputs,
            kernel,
            window_strides=(self.stride,),
            padding=self.padding.upper(),
            lhs_dilation=(1,),
            rhs_dilation=(1,),
            dimension_numbers=lax.ConvDimensionNumbers(lhs_spec=(0, 2, 1), rhs_spec=(2, 1, 0), out_spec=(0, 2, 1)),
            feature_group_count=1,
        )
        if self.with_bias:
            bias = self.param("bias", self.b_init if self.b_init is not None else zeros, (self.output_channels,), inputs.dtype)
            out = out + jnp.broadcast_to(bias, out.shape)
        if unbatched:
            out = jnp.squeeze(out, axis=0)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention (Vaswani et al., 2017) that also returns the attention weights.

    Query, key, value and output projections are `Linear` layers named query, key, value and linear.
    If `mask` is given (shape [..., H or 1, T', T]), masked logits are excluded from the softmax and
    rows with no valid key get all-zero weights (`safe_mask=True`), or masked logits are set to -1e30
    (`safe_mask=False`, standard behaviour).

    Returns (output [..., T', model_size], attention weights [..., H, T', T]).
    """
    num_heads: int
    key_size: int
    w_init: Optional[Initializer] = None
    w_init_scale: Optional[float] = None
    with_bias: bool = True
    b_init: Optional[Initializer] = None
    value_size: Optional[int] = None
    model_size: Optional[int] = None
    safe_mask: bool = True

    def _w_init(self) -> Initializer:
        if self.w_init is not None:
            return self.w_init
        if self.w_init_scale is None:
            raise ValueError("Please provide a weight initializer: `w_init` or `w_init_scale`.")
        return variance_scaling(self.w_init_scale)

    def _linear_projection(self, x: Array, head_size: int, name: str) -> Array:
        y = Linear(self.num_heads * head_size, with_bias=self.with_bias, w_init=self._w_init(), b_init=self.b_init, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))

    @nn.compact
    def __call__(self, query: Array, key: Array, value: Array, mask: Optional[Array] = None) -> tuple[Array, Array]:
        value_size = self.value_size or self.key_size
        model_size = self.model_size or self.key_size * self.num_heads
        *leading_dims, sequence_length, _ = query.shape
        query_heads = self._linear_projection(query, self.key_size, "query")  # [T', H, Q=K]
        key_heads = self._linear_projection(key, self.key_size, "key")  # [T, H, K]
        value_heads = self._linear_projection(value, value_size, "value")  # [T, H, V]
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        if self.safe_mask:
            attn_logits = attn_logits / jnp.sqrt(self.key_size).astype(key.dtype)
        else:
            attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype)
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(f"Mask dimensionality {mask.ndim} must match logits dimensionality {attn_logits.ndim}.")
            if self.safe_mask:
                mask = mask.astype(jnp.bool_)
                has_keys = jnp.any(mask, axis=-1, keepdims=True)
                masked_logits = jnp.where(mask, attn_logits, -jnp.inf)
                safe_logits = jnp.where(has_keys, masked_logits, 0.0)
                attn_weights = jax.nn.softmax(safe_logits, axis=-1)
                attn_weights = jnp.where(mask, attn_weights, 0.0)  # [H, T', T]
            else:
                attn_logits = jnp.where(mask, attn_logits, -1e30)
                attn_weights = jax.nn.softmax(attn_logits)
        else:
            attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # [H, T', T]
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]
        out = Linear(model_size, with_bias=self.with_bias, w_init=self._w_init(), b_init=self.b_init, name="linear")(attn)
        return out, attn_weights

# --------------------------------------------------------------------------------------
# Parameters saved by Haiku-based versions of social-jym
# --------------------------------------------------------------------------------------

_HAIKU_LEAF_NAMES = {"w": "kernel", "b": "bias", "offset": "bias"}


def is_haiku_params(params: Any) -> bool:
    """True if `params` is a Haiku parameter dict: {"root/~/module/~/...": {"w": array, ...}, ...}."""
    if not isinstance(params, Mapping) or not params or "params" in params:
        return False
    return all(isinstance(k, str) and isinstance(v, Mapping) and all(not isinstance(vv, Mapping) for vv in v.values())
               for k, v in params.items())


def haiku_to_flax(params: Mapping) -> dict:
    """
    Convert Haiku parameters of one transformed network to Flax variables ({"params": ...}).

    Module paths ("root/~/child/~/linear_0" or "root/child") become nested dicts without the root
    module, and leaves w, b and offset become kernel, bias and bias. The arrays are not copied.
    """
    roots = {re.split(r"/~/|/", k)[0] for k in params}
    if len(roots) != 1:
        raise ValueError(f"Haiku parameters must belong to a single network, found roots {sorted(roots)}")
    out: dict = {}
    for path, leaves in params.items():
        node = out
        for part in re.split(r"/~/|/", path)[1:]:
            node = node.setdefault(part, {})
        for name, value in leaves.items():
            node[_HAIKU_LEAF_NAMES.get(name, name)] = value
    return {"params": out}


def as_flax_variables(params: Any) -> Any:
    """Return `params` unchanged if they are Flax variables, converted if they are Haiku parameters."""
    return haiku_to_flax(params) if is_haiku_params(params) else params


class Network:
    """
    A Flax module with `init` and `apply` functions.

    `apply` also accepts parameters saved by the Haiku-based versions of social-jym, which it
    converts with `haiku_to_flax` (the conversion runs when the function is traced).
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module

    def init(self, key: Array | int, *args: Any, **kwargs: Any) -> dict:
        """Initialize the parameters. `key` is a PRNG key or an integer seed (as accepted by Haiku)."""
        if jnp.ndim(key) == 0 and jnp.issubdtype(jnp.result_type(key), jnp.integer):
            key = jax.random.PRNGKey(key)
        return dict(self.module.init(key, *args, **kwargs))

    def apply(self, params: Any, *args: Any, **kwargs: Any) -> Any:
        return self.module.apply(as_flax_variables(params), *args, **kwargs)
