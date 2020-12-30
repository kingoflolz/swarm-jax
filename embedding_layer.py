import functools
import operator
import random
import time
from functools import partial
from queue import Queue
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

from swarm_layer import save_checkpoint, load_checkpoint, opt_state, run_threads, run_function, NetworkPrecision, \
    quantize, dequantize


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)


@ray.remote(num_gpus=0.01, num_cpus=0.01)
class EmbeddingLayer(object):
    def __init__(self, obs, vocab: int, d_model: int, optimizer: optax.GradientTransformation,
                 precision: NetworkPrecision):
        self.vocab = vocab
        self.d_model = d_model
        self.optimizer = optimizer
        self.precision = precision

        def embed_forward(x):
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)

            seq_length = x.shape[1]
            positional_embeddings = hk.get_parameter('pos_embs', [seq_length, d_model], init=embed_init)

            o = hk.Embed(vocab, d_model, w_init=embed_init, name="embedding")(x) + positional_embeddings

            return o

        self.embed_fwd_fn = hk.transform(embed_forward)

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

        @functools.partial(jax.jit, donate_argnums=(1, 2))
        def embed_grad_fn(obs, y_dy, acc, params):
            y, dy = y_dy

            y_new, vjpfun = jax.vjp(self.embed_fwd_fn.apply, params, None, obs)
            weights_grad, _, _ = vjpfun(dy)
            diff = jnp.square(y - y_new).mean()

            new_acc = jax.tree_multimap(operator.add, acc, weights_grad)
            return diff, new_acc

        # we call all the functions here to trigger jit at init
        self.state = init_fn(master_rng, obs)

        num_params = hk.data_structures.tree_size(self.state["params"])
        print(f'Param count = {num_params}')

        self.embed_fwd = embed_fwd_fn
        e = self.embed_fwd(obs, self.state)

        self.embed_grad = embed_grad_fn
        _, new_acc = self.embed_grad(obs, (e, e), self.state["grad_acc"], self.state["params"])
        self.state["grad_acc"] = new_acc

        self.state = opt_state(self.state, self.optimizer)
        self.state = init_fn(master_rng, obs)

        self.init = False

    def run(self):
        def forward(obs):
            return quantize(self.embed_fwd(obs, self.state), self.precision.fwd_act)

        def backward(y_dy, obs):
            y, dy = y_dy
            y_dy = (dequantize(y, "float32"), dequantize(dy, "float32"))
            diff, new_grad_acc = self.embed_grad(obs, y_dy, self.state["grad_acc"], self.state["params"])
            self.state["grad_acc"] = new_grad_acc
            self.state["grad_count"] = self.state["grad_count"] + 1

            return diff

        self.fwd_q = Queue(2)
        self.bwd_q = Queue(2)
        self.init = True

        run_threads(self.fwd_q, self.bwd_q, 2, forward, backward)

    def embed_forward(self, obs):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.fwd_q, obs)

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


@ray.remote(num_gpus=0.01, num_cpus=0.01)
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

        @functools.partial(jax.jit)
        def init_fn(master_rng, data):
            out_rng, init_rng = jax.random.split(master_rng)
            params = self.proj_fwd_fn.init(init_rng, data)

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
        def debed_fwd_fn(target, state):
            params = state['params']
            out = self.proj_fwd_fn.apply(params, None, target)

            return out

        @functools.partial(jax.jit, donate_argnums=(0, 2))
        def debed_grad_fn(hidden, target, acc, params):

            loss, vjpfun = jax.vjp(self.proj_loss_fn.apply, params, None, hidden, target)
            weights_grad, _, x_grad, _ = vjpfun(np.ones((), dtype=hidden.dtype))

            new_acc = jax.tree_multimap(operator.add, acc, weights_grad)
            return hidden, x_grad, loss, new_acc

        # we call all the functions here to trigger jit at init
        self.state = init_fn(master_rng, data)

        num_params = hk.data_structures.tree_size(self.state["params"])
        print(f'Param count = {num_params}')

        self.debed_fwd = debed_fwd_fn
        self.debed_fwd(jnp.zeros_like(data), self.state)

        self.debed_grad = debed_grad_fn
        _, _, _, new_acc = self.debed_grad(jnp.zeros_like(data), np.ones_like(data).mean(axis=-1),
                                           self.state["grad_acc"],
                                           self.state["params"])
        self.state["grad_acc"] = new_acc

        self.state = opt_state(self.state, self.optimizer)
        self.state = init_fn(master_rng, jnp.zeros_like(data))

        self.init = False

    def run(self):
        def forward(h):
            return self.debed_fwd(dequantize(h, "float32"), self.state)

        def backward(h, targets):
            hidden, x_grad, loss, new_acc = self.debed_grad(dequantize(h, "float32"), targets, self.state["grad_acc"],
                                                            self.state["params"])
            self.state["grad_acc"] = new_acc
            self.state["grad_count"] = self.state["grad_count"] + 1

            return (quantize(hidden, self.precision.rev_act), quantize(x_grad, self.precision.grad)), loss

        self.fwd_q = Queue(2)
        self.bwd_q = Queue(2)
        self.init = True

        run_threads(self.fwd_q, self.bwd_q, 2, forward, backward)

    def debed_forward(self, h):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.fwd_q, h)

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
