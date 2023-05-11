import functools
import operator
import random
import time
from queue import Queue

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import ray
from typing import Callable

from .swarm_layer import save_checkpoint, load_checkpoint, opt_state, run_threads, run_function, NetworkPrecision, \
    quantize, dequantize, init_fn


@ray.remote
class ReversibleLayer(object):
    def __init__(
            self,
            layer_init: Callable,
            layer: int,
            data: jnp.ndarray,
            optimizer: optax.GradientTransformation,
            precision: NetworkPrecision
    ):
        self.layer = layer
        self.optimizer = optimizer
        self.precision = precision

        data = dequantize(data, "float32")

        def forward(x):
            f, g = layer_init(layer)

            hidden = x.shape[-1]
            x1 = x[:, :, :hidden // 2]
            x2 = x[:, :, hidden // 2:]

            y1 = f(x2) + x1
            y2 = g(y1) + x2

            assert x1.shape == y1.shape
            assert x2.shape == y2.shape

            return jnp.concatenate((y1, y2), axis=-1)

        def reverse(y):
            f, g = layer_init(layer)

            hidden = y.shape[-1]
            y1 = y[:, :, :hidden // 2]
            y2 = y[:, :, hidden // 2:]

            x2 = y2 - g(y1)
            x1 = y1 - f(x2)

            return jnp.concatenate((x1, x2), axis=-1)

        self.forward_fn = hk.transform(forward)
        self.reverse_fn = hk.transform(reverse)

        master_rng = jax.random.PRNGKey(random.getrandbits(32))

        @functools.partial(jax.pmap, donate_argnums=0)
        def forward_fn(x, params):
            out = self.forward_fn.apply(params, None, x)
            return out

        @functools.partial(jax.pmap, donate_argnums=(0, 1))
        def reverse_fn(y_dy, acc, params):
            y, dy = y_dy
            reconstr_x = self.reverse_fn.apply(params, None, y)

            _, vjpfun = jax.vjp(self.forward_fn.apply, params, None, reconstr_x)
            weights_grad, _, x_grad = vjpfun(dy)

            new_acc = jax.tree_map(operator.add, acc, weights_grad)
            return (reconstr_x, x_grad), new_acc

        shape = (8, 16, 128, 2)
        random_array = jax.random.uniform(master_rng, shape=shape, dtype=jnp.float32)

        self.state = init_fn(master_rng, jnp.zeros_like(data), self.forward_fn.init, optimizer)
        num_params = hk.data_structures.tree_size(self.state["params"])
        print(f'Param count = {num_params}')

        self.forward = forward_fn
        self.forward(jnp.zeros_like(data), self.state["params"])

        self.reverse = reverse_fn
        _, new_acc = self.reverse((jnp.zeros_like(data), jnp.zeros_like(data)), self.state["grad_acc"],
                                  self.state["params"])
        self.state["grad_acc"] = new_acc

        self.state = opt_state(self.state, self.optimizer)
        self.state = init_fn(master_rng, jnp.zeros_like(data), self.forward_fn.init, optimizer)

        self.init = False

    def run(self):
        def forward(h, state):
            return quantize(self.forward(dequantize(h, "float32"), state["params"]), self.precision.fwd_act)

        def backward(y_dy, state):
            y, dy = y_dy
            y_dy = (dequantize(y, "float32"), dequantize(dy, "float32"))
            x_dx, new_acc = self.reverse(y_dy, state["grad_acc"], state["params"])
            state["grad_acc"] = new_acc
            state["grad_count"] = state["grad_count"] + 1

            self.state = state

            x, dx = x_dx
            return quantize(x, self.precision.rev_act), quantize(dx, self.precision.grad)

        self.fwd_q = Queue(2)
        self.bwd_q = Queue(2)
        self.init = True

        run_threads(self.state, self.fwd_q, self.bwd_q, 2, forward, backward)

    @ray.method(num_returns=2)
    def forward(self, h):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.fwd_q, h), None

    @ray.method(num_returns=2)
    def backward(self, y_dy):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.bwd_q, y_dy), None

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