import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Callable


class MultiHeadAttentionFixed(hk.Module):
    """Multi-headed attention mechanism.

    With fixed attention scaling
    """

    def __init__(
            self,
            num_heads: int,
            key_size: int,
            w_init_scale: float,
            query_size: Optional[int] = None,
            value_size: Optional[int] = None,
            model_size: Optional[int] = None,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.query_size = query_size or key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)

    def __call__(
            self,
            query: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Compute (optionally masked) MHA with queries, keys & values."""
        query_heads = self._linear_projection(query, self.query_size, "query")
        key_heads = self._linear_projection(query, self.key_size, "key")
        value_heads = self._linear_projection(query, self.value_size, "value")

        sqrt_key_size = np.sqrt(self.key_size)
        query_heads = query_heads / sqrt_key_size

        attention_logits = jnp.einsum("bthd,bThd->bhtT", query_heads, key_heads)

        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask

        attention_logits -= 1e10 * (1. - mask)

        attention_weights = jax.nn.softmax(attention_logits)
        attention = jnp.einsum("bhtT,bThd->bthd", attention_weights, value_heads)
        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*query.shape[:2], -1))

        return hk.Linear(self.model_size, w_init=self.w_init)(attention_vec)

    @hk.transparent
    def _linear_projection(
            self,
            x: jnp.ndarray,
            head_size: int,
            name: Optional[str] = None
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        return y.reshape((*x.shape[:2], self.num_heads, head_size))


class DenseBlock(hk.Module):
    """A 2-layer MLP which widens then narrows the input."""

    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)


class SwarmModel:
    def __init__(self, vocab: int, d_model: int,
                 rev_init: Callable, rev_layers: int):
        self.vocab = vocab
        self.d_model = d_model
        self.rev_init = rev_init
        self.rev_layers = rev_layers


n_layer = 6

def char_layer_init(i):
    if i % 2:
        f = MultiHeadAttentionFixed(
            num_heads=8,
            key_size=128,
            w_init_scale=2. / n_layer,
            name=f'l{i}_f_attn',
        )
        g = DenseBlock(
            init_scale=2. / n_layer,
            name=f'l{i}_g_dense',
            widening_factor=4
        )
    else:
        f = DenseBlock(
            init_scale=2. / n_layer,
            name=f'l{i}_f_dense',
            widening_factor=4
        )
        g = MultiHeadAttentionFixed(
            num_heads=8,
            key_size=128,
            w_init_scale=2. / n_layer,
            name=f'l{i}_g_attn',
        )
    return f, g


SwarmCharTransformer = SwarmModel(
    vocab=256,
    d_model=512,
    rev_init=char_layer_init,
    rev_layers=n_layer
)

SwarmCharTransformerBig = SwarmModel(
    vocab=256,
    d_model=2048,
    rev_init=char_layer_init,
    rev_layers=n_layer
)
