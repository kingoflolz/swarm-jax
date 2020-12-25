import functools
import operator
import random
from functools import partial
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

from swarm_layer import save_checkpoint, load_checkpoint


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)


@ray.remote(num_gpus=0.01, num_cpus=0.01)
class EmbeddingLayer(object):
    def __init__(self, obs, vocab: int, d_model: int, optimizer: optax.GradientTransformation):
        self.vocab = vocab
        self.d_model = d_model

        def embed_forward(x):
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)

            seq_length = x.shape[1]
            positional_embeddings = hk.get_parameter('pos_embs', [seq_length, d_model], init=embed_init)

            o = hk.Embed(vocab, d_model, w_init=embed_init, name="embedding")(x) + positional_embeddings

            if hk.running_init():
                layer_norm(o)  # create this here for init
                hk.Linear(vocab)(o)

            return o

        def debed_forward(x):
            x = layer_norm(x)

            return hk.Linear(vocab)(x)

        def debed_loss(x, target):
            logits = debed_forward(x)
            target_onehot = jax.nn.one_hot(target, vocab)

            assert logits.shape == target_onehot.shape

            loss = -jnp.sum(target_onehot * jax.nn.log_softmax(logits), axis=-1)
            loss = jnp.mean(loss)

            return loss

        @functools.partial(jax.jit)
        def opt(state):
            def grad_to_cpu(x):
                return jax.device_put(x, device=jax.devices("cpu")) / state['grad_count']

            # grad_cpu = jax.tree_map(grad_to_cpu, state['grad_acc'])

            updates, opt_state = optimizer.update(state['grad_acc'], state['opt_state'])
            state['params'] = optax.apply_updates(state['params'], updates)
            state['opt_state'] = opt_state

            state['grad_acc'] = jax.tree_map(jnp.zeros_like, state['grad_acc'])
            state['grad_count'] = np.array(0)
            return state

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
                grad_acc=jax.tree_map(jnp.zeros_like, params),
                grad_count=np.array(0),
                params=params)

        @functools.partial(jax.jit)
        def embed_fwd_fn(obs, state):
            params = state['params']
            out = self.embed_fwd_fn.apply(params, None, obs)

            return out

        @functools.partial(jax.jit)
        def embed_grad_fn(obs, y_dy, state):
            params = state['params']
            acc = state['grad_acc']

            y, dy = y_dy

            y_new, vjpfun = jax.vjp(self.embed_fwd_fn.apply, params, None, obs)
            weights_grad, _, _ = vjpfun(dy)

            diff = jnp.square(y - y_new).mean()

            state['grad_acc'] = jax.tree_multimap(operator.add, acc, weights_grad)
            state['grad_count'] = state['grad_count'] + 0.5

            return diff, state

        @functools.partial(jax.jit)
        def debed_fwd_fn(target, state):
            params = state['params']
            out = self.debed_fwd_fn.apply(params, None, target)

            return out

        @functools.partial(jax.jit)
        def debed_grad_fn(hidden, target, state):
            params = state['params']
            acc = state['grad_acc']

            loss, vjpfun = jax.vjp(self.debed_loss_fn.apply, params, None, hidden, target)
            weights_grad, _, x_grad, _ = vjpfun(np.ones((), dtype=hidden.dtype))

            state['grad_acc'] = jax.tree_multimap(operator.add, acc, weights_grad)
            state['grad_count'] = state['grad_count'] + 1

            return hidden, x_grad, loss, state

        # we call all the functions here to trigger jit at init
        self.state = init_fn(master_rng, obs)

        num_params = hk.data_structures.tree_size(self.state["params"])
        print(f'Param count = {num_params}')

        self.embed_fwd = embed_fwd_fn
        e = self.embed_fwd(obs, self.state)

        self.embed_grad = embed_grad_fn
        self.embed_grad(obs, (e, e), self.state)

        self.debed_fwd = debed_fwd_fn
        self.debed_fwd(e, self.state)

        self.debed_grad = debed_grad_fn
        self.debed_grad(e, np.ones_like(e).mean(axis=-1), self.state)

        self.opt = opt
        self.opt(self.state)

        self.state = init_fn(master_rng, obs)

    def embed_forward(self, obs):
        return self.embed_fwd(obs, self.state)

    def embed_grad(self, obs, y_dy):
        diff, state = self.embed_grad(obs, y_dy, self.state)
        self.state = state

        return diff

    def debed_forward(self, h):
        return self.debed_fwd(h, self.state)

    @ray.method(num_returns=2)
    def debed_grad(self, h, targets):
        hidden, x_grad, loss, state = self.debed_grad(h, targets, self.state)
        self.state = state

        return (hidden, x_grad), loss

    def opt(self):
        self.state = self.opt(self.state)

    def get_params(self):
        return self.state["params"]

    def get_accum(self):
        return self.state["grad_acc"]

    def save(self, path, epoch):
        save_checkpoint(self.state, path, epoch)

    def load(self, path):
        ckpt = load_checkpoint(path)

        if ckpt:
            self.state = ckpt
            return True