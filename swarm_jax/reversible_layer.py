import functools
import operator
import random
import time
from functools import partial
from queue import Queue
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

from .swarm_layer import save_checkpoint, load_checkpoint, opt_state, run_threads, run_function, NetworkPrecision, \
    quantize, dequantize


@ray.remote(resources={"tpu": 1})
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

        @functools.partial(jax.jit)
        def init_fn(master_rng, data):
            out_rng, init_rng = jax.random.split(master_rng)
            params = self.forward_fn.init(init_rng, data)

            # place optimizer state on CPU
            opt_state = jax.tree_map(partial(jax.device_put, device=jax.devices("cpu")), optimizer.init(params))

            return dict(
                step=np.array(0),
                rng=out_rng,
                opt_state=opt_state,
                grad_acc=jax.tree_map(jnp.zeros_like, params),
                grad_count=np.array(1),
                params=params)

        @functools.partial(jax.jit, donate_argnums=0)
        def forward_fn(x, state):
            params = state['params']
            out = self.forward_fn.apply(params, None, x)
            return out

        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def reverse_fn(y_dy, acc, params):
            y, dy = y_dy
            reconstr_x = self.reverse_fn.apply(params, None, y)

            _, vjpfun = jax.vjp(self.forward_fn.apply, params, None, reconstr_x)
            weights_grad, _, x_grad = vjpfun(dy)

            new_acc = jax.tree_multimap(operator.add, acc, weights_grad)
            return (reconstr_x, x_grad), new_acc

        self.state = init_fn(master_rng, jnp.zeros_like(data))
        num_params = hk.data_structures.tree_size(self.state["params"])
        print(f'Param count = {num_params}')

        self.forward = forward_fn
        self.forward(jnp.zeros_like(data), self.state)

        self.reverse = reverse_fn
        _, new_acc = self.reverse((jnp.zeros_like(data), jnp.zeros_like(data)), self.state["grad_acc"],
                                  self.state["params"])
        self.state["grad_acc"] = new_acc

        self.state = opt_state(self.state, self.optimizer)
        self.state = init_fn(master_rng, jnp.zeros_like(data))

        self.init = False

    def run(self):
        def forward(h):
            return quantize(self.forward(dequantize(h, "float32"), self.state), self.precision.fwd_act)

        def backward(y_dy):
            y, dy = y_dy
            y_dy = (dequantize(y, "float32"), dequantize(dy, "float32"))
            x_dx, new_acc = self.reverse(y_dy, self.state["grad_acc"], self.state["params"])
            self.state["grad_acc"] = new_acc
            self.state["grad_count"] = self.state["grad_count"] + 1

            x, dx = x_dx
            return quantize(x, self.precision.rev_act), quantize(dx, self.precision.grad)

        self.fwd_q = Queue(2)
        self.bwd_q = Queue(2)
        self.init = True

        run_threads(self.fwd_q, self.bwd_q, 2, forward, backward)

    def forward(self, h):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.fwd_q, h)

    def backward(self, y_dy):
        while not self.init:
            time.sleep(0.1)
        return run_function(self.bwd_q, y_dy)

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