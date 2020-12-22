import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda-10.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import random
from functools import partial

from typing import Optional

import optax
import ray
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from loader import TextLoader


class MemAttention(hk.MultiHeadAttention):
    """Self attention with a causal mask applied and persistant vectors."""

    def __init__(self,
                 mem_vectors: int = 1024,
                 **kwargs):
        super().__init__(**kwargs)
        self.mem_vectors = mem_vectors

    def __call__(
            self,
            query: jnp.ndarray,
            key: Optional[jnp.ndarray] = None,
            value: Optional[jnp.ndarray] = None,
            mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask

        query_heads = self._linear_projection(query, self.query_size, "query")
        key_heads = self._linear_projection(key, self.key_size, "key")
        value_heads = self._linear_projection(value, self.value_size, "value")

        sqrt_key_size = np.sqrt(self.key_size, dtype=key.dtype)
        query_heads = query_heads / sqrt_key_size

        mem_k = hk.get_parameter("mem_k", [self.mem_vectors, self.num_heads, self.key_size], query.dtype,
                                 init=self.w_init)
        mem_v = hk.get_parameter("mem_v", [self.mem_vectors, self.num_heads, self.value_size], query.dtype,
                                 init=self.w_init)

        mem_k_logit = jnp.einsum("bthd,mhd->bhtm", query_heads, mem_k * 64)

        attention_logits = jnp.einsum("bthd,bThd->bhtT", query_heads, key_heads)
        attention_logits -= 1e10 * (1. - mask)

        full_logits = jnp.concatenate((attention_logits, mem_k_logit), axis=-1)
        attention_weights = jax.nn.softmax(full_logits)

        context_weights = attention_weights[:, :, :, :-self.mem_vectors]
        mem_weights = attention_weights[:, :, :, -self.mem_vectors:]

        context_attention = jnp.einsum("bhtT,bThd->bthd", context_weights, value_heads)
        mem_attention = jnp.einsum("bhtm,mhd->bthd", mem_weights, mem_v * 64)

        attention = context_attention + mem_attention

        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*query.shape[:2], -1))

        return hk.Linear(self.model_size, w_init=self.w_init)(attention_vec)


@ray.remote(num_gpus=0.01, num_cpus=0.01)
class ReversibleLayer(object):
    def __init__(
            self,
            init_scale,
            num_heads,
            key_size,
            mem_vectors,
            layer,
            data
    ):
        self.layer = layer

        def forward(x):
            f = MemAttention(
                num_heads=num_heads,
                key_size=key_size,
                w_init_scale=init_scale,
                name=f'l{layer}_f',
                mem_vectors=mem_vectors
            )

            g = MemAttention(
                num_heads=num_heads,
                key_size=key_size,
                w_init_scale=init_scale,
                name=f'l{layer}_g',
                mem_vectors=mem_vectors
            )

            hidden = x.shape[-1]
            x1 = x[:, :, :hidden // 2]
            x2 = x[:, :, hidden // 2:]

            y1 = f(x2) + x1
            y2 = g(y1) + x2
            return jnp.concatenate((y1, y2), axis=-1)

        def reverse(y):
            f = MemAttention(
                num_heads=num_heads,
                key_size=key_size,
                w_init_scale=init_scale,
                name=f'l{layer}_f',
                mem_vectors=mem_vectors
            )

            g = MemAttention(
                num_heads=num_heads,
                key_size=key_size,
                w_init_scale=init_scale,
                name=f'l{layer}_g',
                mem_vectors=mem_vectors
            )

            hidden = y.shape[-1]
            y1 = y[:, :, :hidden // 2]
            y2 = y[:, :, hidden // 2:]

            x2 = y2 - g(y1)
            x1 = y1 - f(x2)
            return jnp.concatenate((x1, x2), axis=-1)

        self.forward_fn = hk.transform(forward)
        self.reverse_fn = hk.transform(reverse)

        master_rng = jax.random.PRNGKey(random.getrandbits(32))

        @functools.partial(jax.jit, static_argnums=0)
        def init_fn(master_rng, data):
            out_rng, init_rng = jax.random.split(master_rng)
            params = self.forward_fn.init(init_rng, data)

            # place optimizer state on CPU
            opt_state = jax.tree_map(partial(jax.device_put, device=jax.devices("cpu")), optimizer.init(params))

            return dict(
                step=np.array(0),
                rng=out_rng,
                opt_state=opt_state,
                params=params)

        @functools.partial(jax.jit)
        def forward_fn(x, state):
            params = state['params']
            out = self.forward_fn.apply(params, None, x)
            return out

        @functools.partial(jax.jit)
        def reverse_fn(y_dy, state):
            params = state['params']

            y, dy = y_dy
            reconstr_x = self.reverse_fn.apply(params, None, y)

            _, vjpfun = jax.vjp(self.forward_fn.apply, params, None, reconstr_x)
            weight_grad, _, x_grad = vjpfun(dy)
            return reconstr_x, x_grad

        self.state = init_fn(master_rng, data)
        self.forward_partial = partial(forward_fn, state=self.state)
        self.forward_partial(data)

        self.reverse_partial = partial(reverse_fn, state=self.state)
        self.reverse_partial((data, jnp.zeros_like(data)))

    def forward(self, h):
        return self.forward_partial(h)

    def backward(self, y_dy):
        return self.reverse_partial(y_dy)


@ray.remote(num_gpus=0.01, num_cpus=0.01)
class EmbeddingLayer(object):
    def __init__(self, data, vocab: int, d_model: int, optimizer: optax.GradientTransformation):
        self.vocab = vocab
        self.d_model = d_model

        def embed_forward(x):
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            return hk.Embed(vocab, d_model, w_init=embed_init, name="embedding")(x)

        def debed_forward(x):
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            embed = hk.Embed(vocab, d_model, w_init=embed_init, name="embedding").embeddings

            return x @ embed

        def debed_loss(x, target):
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            embed = hk.Embed(vocab, d_model, w_init=embed_init, name="embedding").embeddings

            logits = x @ embed

            target_onehot = jax.nn.one_hot(target, vocab)
            loss = -jnp.sum(target_onehot * jax.nn.log_softmax(logits), axis=-1)
            loss = jnp.sum(loss)

            return loss

        self.embed_fwd_fn = hk.transform(embed_forward)
        self.debed_fwd_fn = hk.transform(debed_forward)
        self.debed_loss_fn = hk.transform(debed_loss)

        master_rng = jax.random.PRNGKey(random.getrandbits(32))

        @functools.partial(jax.jit, static_argnums=0)
        def init_fn(master_rng, data):
            out_rng, init_rng = jax.random.split(master_rng)
            params = self.embed_fwd_fn.init(init_rng, data)

            # place optimizer state on CPU
            opt_state = jax.tree_map(partial(jax.device_put, device=jax.devices("cpu")), optimizer.init(params))

            return dict(
                step=np.array(0),
                rng=out_rng,
                opt_state=opt_state,
                params=params)

        @functools.partial(jax.jit)
        def embed_fwd_fn(x, state):
            params = state['params']
            out = self.embed_fwd_fn.apply(params, None, x)

            return out

        @functools.partial(jax.jit)
        def debed_fwd_fn(x, state):
            params = state['params']
            out = self.debed_fwd_fn.apply(params, None, x)

            return out

        @functools.partial(jax.jit)
        def debed_grad_fn(x, target, state):
            params = state['params']

            _, vjpfun = jax.vjp(self.debed_loss_fn.apply, params, None, x, target)
            weights_grad, _, x_grad, _ = vjpfun(np.ones((), dtype=x.dtype))

            return x, x_grad, weights_grad

        # we call all the functions here to trigger jit at init
        self.state = init_fn(master_rng, data)
        self.embed_fwd_partial = partial(embed_fwd_fn, state=self.state)
        e = self.embed_fwd_partial(data)

        self.debed_fwd_partial = partial(debed_fwd_fn, state=self.state)
        self.debed_fwd_partial(e)

        self.debed_grad_partial = partial(debed_grad_fn, state=self.state)
        self.debed_grad_partial(e, np.ones_like(e).mean(axis=-1))

    def embed_forward(self, obs):
        return self.embed_fwd_partial(obs)

    def debed_forward(self, h):
        return self.debed_fwd_partial(h)

    def debed_grad(self, h, targets):
        return self.debed_grad_partial(h, targets)[:2]

    def __repr__(self):
        return "EmbeddingLayer"


train_dataset = TextLoader("data/enwik8", batchsize=16, sample_size=128, length=90000000)
data = train_dataset.get_samples()
optimizer = optax.chain(
    optax.clip_by_global_norm(0.25),
    optax.adam(2e-4, b1=0.9, b2=0.99))

ray.init()

embedding_actor = EmbeddingLayer.remote(data["obs"], 256, 256, optimizer)

embedding = embedding_actor.embed_forward.remote(data["obs"])

layers = []

for i in range(6):
    layers.append(ReversibleLayer.remote(2/6, 4, 32, 512, i, embedding))

dbg = True

while True:
    data = train_dataset.get_samples()

    x = embedding_actor.embed_forward.remote(data["obs"])

    if dbg:
        fwd_activations = []
        bwd_activations = []

    for l in layers:
        if dbg:
            fwd_activations.append(x)
        x = l.forward.remote(x)
    y_dy = embedding_actor.debed_grad.remote(x, data["target"])

    for l in reversed(layers):
        y_dy = l.backward.remote(y_dy)
        if dbg:
            bwd_activations.append(y_dy)
        pass

    if dbg:
        fwd_activations = ray.get(fwd_activations)
        bwd_activations_grad = ray.get(bwd_activations)

        bwd_activations = [i[0] for i in bwd_activations_grad]
        bwd_activations.reverse()

        for f, b in zip(fwd_activations, bwd_activations):
            assert jnp.allclose(f, b, rtol=1e-4, atol=1e-4)

    pass

ray.shutdown()