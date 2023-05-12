import functools
import operator
import random
import sys
import time
from queue import Queue

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray
from typing import Optional

from .swarm_layer import save_checkpoint, load_checkpoint, opt_state, run_threads, run_function, NetworkPrecision, \
    quantize, dequantize, init_fn


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)


@ray.remote
class EmbeddingLayer(object):
    def __init__(self, obs, vocab: int, d_model: int, optimizer: optax.GradientTransformation,
                 precision: NetworkPrecision):
        self.vocab = vocab
        self.d_model = d_model
        self.optimizer = optimizer
        self.precision = precision

        print("start init")
        self.devices = jax.local_device_count()
        print("done jax init")

        def embed_forward(x):
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)

            seq_length = x.shape[1]
            positional_embeddings = hk.get_parameter('pos_embs', [seq_length, d_model], init=embed_init)

            o = hk.Embed(vocab, d_model, w_init=embed_init, name="embedding")(x) + positional_embeddings

            return o

        self.embed_fwd_fn = hk.transform(embed_forward)
        master_rng = jax.random.PRNGKey(random.getrandbits(32))

        @functools.partial(jax.pmap)
        def embed_fwd_fn(obs, params):
            out = self.embed_fwd_fn.apply(params, None, obs)

            return out

        @functools.partial(jax.pmap, donate_argnums=(1, 2))
        def embed_grad_fn(obs, y_dy, acc, params):
            y, dy = y_dy

            y_new, vjpfun = jax.vjp(self.embed_fwd_fn.apply, params, None, obs)
            weights_grad, _, _ = vjpfun(dy)
            diff = jnp.square(y - y_new).mean()
            cos_err = jnp.abs(1.0 - jnp.dot(y_new.flatten(), y.flatten()) / (
                    jnp.linalg.norm(y.flatten()) * jnp.linalg.norm(y_new.flatten())))

            new_acc = jax.tree_map(operator.add, acc, weights_grad)
            return diff, cos_err, new_acc

        # we call all the functions here to trigger jit at init
        self.state = init_fn(master_rng, obs, self.embed_fwd_fn.init, optimizer)

        num_params = hk.data_structures.tree_size(self.state["params"])
        print(f'Param count = {num_params}')

        self.embed_fwd = embed_fwd_fn
        e = self.embed_fwd(obs, self.state["params"])

        self.embed_grad = embed_grad_fn
        _, _, new_acc = self.embed_grad(obs, (e, e), self.state["grad_acc"], self.state["params"])
        self.state["grad_acc"] = new_acc

        self.state = opt_state(self.state, self.optimizer)
        self.state = init_fn(master_rng, obs, self.embed_fwd_fn.init, optimizer)

        self.init = False

    def run(self):
        def forward(obs, state):
            return quantize(self.embed_fwd(obs, state["params"]), self.precision.fwd_act)

        def backward(y_dy, obs, state):
            y, dy = y_dy
            y_dy = (dequantize(y, "float32"), dequantize(dy, "float32"))
            diff, cos_err, new_grad_acc = self.embed_grad(obs, y_dy, state["grad_acc"], state["params"])
            state["grad_acc"] = new_grad_acc
            state["grad_count"] = state["grad_count"] + 1

            self.state = state

            return diff, cos_err

        self.fwd_q = Queue(2)
        self.bwd_q = Queue(2)
        self.init = True

        run_threads(self.state, self.fwd_q, self.bwd_q, 2, forward, backward)

    @ray.method(num_returns=2)
    def embed_forward(self, obs):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.fwd_q, obs), None

    def embed_grad(self, obs, y_dy):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.bwd_q, y_dy, obs)

    def opt(self):
        self.state = opt_state(self.state, self.optimizer)

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


@ray.remote
class ProjLayer(object):
    def __init__(self, data, vocab: int, d_model: int, optimizer: optax.GradientTransformation, loss_scale: float,
                 precision: NetworkPrecision):
        self.vocab = vocab
        self.d_model = d_model
        self.optimizer = optimizer
        self.loss_scale = loss_scale
        self.precision = precision

        data = dequantize(data, "float32")

        def debed_forward(x):
            x = layer_norm(x)
            return hk.Linear(vocab)(x)

        def debed_loss(x, target):
            logits = debed_forward(x)
            target_onehot = jax.nn.one_hot(target, vocab)

            assert logits.shape == target_onehot.shape

            loss = -jnp.sum(target_onehot * jax.nn.log_softmax(logits), axis=-1)
            loss = jnp.mean(loss) * self.loss_scale

            return loss

        self.proj_fwd_fn = hk.transform(debed_forward)
        self.proj_loss_fn = hk.transform(debed_loss)

        master_rng = jax.random.PRNGKey(random.getrandbits(32))

        @functools.partial(jax.pmap)
        def debed_fwd_fn(target, params):
            out = self.proj_fwd_fn.apply(params, None, target)

            return out

        @functools.partial(jax.pmap, donate_argnums=(0, 2))
        def debed_grad_fn(hidden, target, acc, params):
            loss, vjpfun = jax.vjp(self.proj_loss_fn.apply, params, None, hidden, target)
            weights_grad, _, x_grad, _ = vjpfun(np.ones((), dtype=hidden.dtype))

            new_acc = jax.tree_map(operator.add, acc, weights_grad)
            return hidden, x_grad, loss, new_acc

        # we call all the functions here to trigger jit at init
        self.state = init_fn(master_rng, data, self.proj_fwd_fn.init, optimizer)

        num_params = hk.data_structures.tree_size(self.state["params"])
        print(f'Param count = {num_params}')

        self.debed_fwd = debed_fwd_fn
        self.debed_fwd(jnp.zeros_like(data), self.state["params"])

        self.debed_grad = debed_grad_fn
        _, _, _, new_acc = self.debed_grad(jnp.zeros_like(data), np.ones_like(data).mean(axis=-1),
                                           self.state["grad_acc"],
                                           self.state["params"])
        self.state["grad_acc"] = new_acc

        self.state = opt_state(self.state, self.optimizer)
        self.state = init_fn(master_rng, jnp.zeros_like(data), self.proj_fwd_fn.init, optimizer)

        self.init = False

    def run(self):
        def forward(h, state):
            return self.debed_fwd(dequantize(h, "float32"), state["params"])

        def backward(h, targets, state):
            hidden, x_grad, loss, new_acc = self.debed_grad(dequantize(h, "float32"), targets, state["grad_acc"],
                                                            state["params"])
            state["grad_acc"] = new_acc
            state["grad_count"] = state["grad_count"] + 1

            self.state = state

            return (quantize(hidden, self.precision.rev_act), quantize(x_grad, self.precision.grad)), loss

        self.fwd_q = Queue(2)
        self.bwd_q = Queue(2)
        self.init = True

        run_threads(self.state, self.fwd_q, self.bwd_q, 2, forward, backward)

    @ray.method(num_returns=2)
    def debed_forward(self, h):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.fwd_q, h), 0

    @ray.method(num_returns=2)
    def debed_grad(self, h, targets):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.bwd_q, h, targets)

    def opt(self):
        self.state = opt_state(self.state, self.optimizer)

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
