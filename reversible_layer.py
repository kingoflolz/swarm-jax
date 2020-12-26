import functools
import operator
import random
from functools import partial
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

from swarm_layer import save_checkpoint, load_checkpoint, opt_state


@ray.remote(num_gpus=0.01, num_cpus=0.01)
class ReversibleLayer(object):
    def __init__(
            self,
            layer_init: Callable,
            layer: int,
            data: jnp.ndarray,
            optimizer: optax.GradientTransformation
    ):
        self.layer = layer
        self.optimizer = optimizer

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
                grad_acc=jax.tree_map(jnp.zeros_like, params),
                grad_count=np.array(0),
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
            weights_grad, _, x_grad = vjpfun(dy)

            state['grad_acc'] = jax.tree_multimap(operator.add, state['grad_acc'], weights_grad)
            state['grad_count'] = state['grad_count'] + 1
            return (reconstr_x, x_grad), state

        self.state = init_fn(master_rng, data)
        num_params = hk.data_structures.tree_size(self.state["params"])
        print(f'Param count = {num_params}')

        self.forward = forward_fn
        self.forward(data, self.state)

        self.reverse = reverse_fn
        self.reverse((data, jnp.zeros_like(data)), self.state)

        self.state = opt_state(self.state, self.optimizer)
        self.state = init_fn(master_rng, data)

    def forward(self, h):
        return self.forward(ray.get(h[0]), self.state)

    def backward(self, y_dy):
        x_dx, new_state = self.reverse(ray.get(y_dy[0]), self.state)
        self.state = new_state
        return x_dx

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