"""
Common methods for layer actors
"""
import pickle
import re
from functools import partial
from pathlib import Path
from queue import Queue, Empty
from threading import Thread

import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray
from dataclasses import dataclass
from glob import glob
from typing import Callable


# TODO: more intellegent checkpoint saving with deleting old checkpoints etc
def save_checkpoint(state, path, epoch):
    Path(path).mkdir(parents=True, exist_ok=True)

    save_file = Path(path, f"ckpt_{epoch:06}.pkl")
    f = open(save_file, "wb")
    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path):
    checkpoints = [int(re.findall(r'\d+', i)[-1]) for i in glob(f"{path}ckpt_*.pkl")]
    checkpoints.sort(reverse=True)

    if checkpoints:
        checkpoint_to_load = checkpoints[0]

        f = open(f"{path}ckpt_{checkpoint_to_load:06}.pkl", "rb")
        return pickle.load(f)
    return None


# @partial(jax.jit, donate_argnums=(0, 1, 2), static_argnums=3)
@partial(jax.jit, static_argnums=3)
def opt_jit(grad_acc, opt_state, params, optimizer):
    total_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), grad_acc)

    cpu_device = jax.devices("cpu")[0]

    total_grad = jax.device_put(total_grad, device=cpu_device)
    cpu_params = jax.device_put(jax.tree_map(lambda x: x[0], params), device=cpu_device)

    updates, new_opt_state = optimizer.update(total_grad, opt_state)

    new_params = optax.apply_updates(cpu_params, updates)

    new_grad_acc = jax.tree_map(jnp.zeros_like, grad_acc)
    return new_grad_acc, new_opt_state, new_params


def opt_state(state, optimizer):
    new_grad_acc, new_opt_state, new_params = opt_jit(state["grad_acc"],
                                                      state["opt_state"],
                                                      state["params"],
                                                      optimizer)

    state["grad_acc"] = new_grad_acc
    state["opt_state"] = new_opt_state
    state["params"] = jax.device_put_replicated(new_params, jax.local_devices())
    state["grad_count"] = np.array(0)
    return state


# @partial(jax.jit)
def init_fn(master_rng, data, init_fn, optimizer):
    out_rng, init_rng = jax.random.split(master_rng)

    # copy the same initial params to each accelerator
    init_rng = jnp.broadcast_to(init_rng, (8,) + init_rng.shape)
    params = jax.pmap(init_fn)(init_rng, data)

    cpu_device = jax.devices("cpu")[0]

    # place optimizer state on CPU
    cpu_params = jax.tree_map(lambda x: jax.device_put(x[0], device=cpu_device), params)
    opt_state = optimizer.init(cpu_params)

    return dict(
        step=np.array(0),
        rng=out_rng,
        opt_state=opt_state,
        grad_acc=jax.tree_map(jnp.zeros_like, params),
        grad_count=np.array(0),
        params=params)


# Thread to overlap remote transfers with computation (with bounded memory usage)
# TODO: have hierarchy of remote -> RAM -> accelerator memory to overlap PCI-e transfer with computation
class GetThread(Thread):
    def __init__(self, in_queue: Queue, out_queue: Queue):
        super().__init__()
        self.i_q = in_queue
        self.o_q = out_queue

    def run(self):
        while True:
            ret_q, obj_id, *aux = self.i_q.get()

            if isinstance(obj_id, ray.ObjectID):
                # GIL released here
                o = ray.get(obj_id)
            else:
                o = obj_id

            self.o_q.put((ret_q, o, *aux))


# TODO: bound the number of pending outputs when layerdrop is added, currently equal to number of pending inputs
class RunThread(Thread):
    def __init__(self, fwd_q: Queue, bwd_q: Queue, fwd_fn: Callable, bwd_fn: Callable):
        super().__init__()
        self.fwd_q = fwd_q
        self.bwd_q = bwd_q

        self.fwd_fn = function_wrapper(fwd_fn)
        self.bwd_fn = function_wrapper(bwd_fn)

    def run(self):
        while True:
            # GIL released in fwd and bwd functions when e.g. XLA computation occurs
            # Prioritize backward over forward to minimize the amount of in flight batches (?)
            # TODO: figure out if this actually makes sense from queuing theory perspective
            while not self.bwd_q.empty():
                self.bwd_fn(*self.bwd_q.get_nowait())
                self.bwd_q.task_done()
            while not self.fwd_q.empty():
                if not self.bwd_q.empty():
                    break
                self.fwd_fn(*self.fwd_q.get_nowait())
                self.fwd_q.task_done()
            try:
                # also release GIL here to not busy wait
                self.bwd_fn(*self.bwd_q.get(timeout=0.01))
                self.bwd_q.task_done()
            except Empty:
                pass


# create a get thread for both fwd and bwd as well as a run thread (blocks forever)
def run_threads(state, fwd_in_q: Queue, bwd_in_q: Queue, queue_size: int, fwd_fn: Callable, bwd_fn: Callable):
    fwd_out_q = Queue(queue_size)
    bwd_out_q = Queue(queue_size)

    fwd_get = GetThread(fwd_in_q, fwd_out_q)
    bwd_get = GetThread(bwd_in_q, bwd_out_q)
    run = RunThread(fwd_out_q, bwd_out_q, partial(fwd_fn, state=state), partial(bwd_fn, state=state))

    fwd_get.start()
    bwd_get.start()
    run.start()

    run.join()
    # should never get here
    raise Exception("Run thread terminated unexpectedly")


# take a function and wrap it to return via queue instead
def function_wrapper(fun):
    def ret_fun(q: Queue, *args):
        ret = fun(*args)
        q.put(ret)

    return ret_fun


# runs a function via queue (blocking, run in threadpool)
def run_function(q: Queue, obj_id, *aux):
    ret_q = Queue(1)

    if isinstance(obj_id, tuple):
        q.put((ret_q, obj_id[0], *aux))
    else:
        q.put((ret_q, obj_id, *aux))

    return ret_q.get()


@partial(jax.jit, static_argnums=2)
def int_quantize_jit(x: jnp.ndarray, max_int: int, to_type: str):
    min = x.min(axis=1, keepdims=True)
    max = x.max(axis=1, keepdims=True)

    offset = min
    scale = max - min

    normalized = (x - min) / scale
    return offset, scale, (normalized * max_int + 0.5).astype(to_type)  # round to nearest instead of round to zero


def quantize(x: jnp.ndarray, to_type: str):
    assert to_type in ["float16", "float32", "uint16", "uint8"]

    if "int" in to_type:
        max_int = 2 ** 8 - 1 if to_type == "uint8" else 2 ** 16 - 1
        return to_type, int_quantize_jit(x, max_int, to_type)
    else:
        return to_type, x.astype(to_type)


@partial(jax.jit, static_argnums=4)
def int_dequantize_jit(x: jnp.ndarray, scale: jnp.ndarray, offset: jnp.ndarray, max_int: int, to_type: str):
    return x.astype(to_type) * scale.astype(to_type) / max_int + offset.astype(to_type)


def dequantize(x, to_type: str):
    from_type, data = x
    assert from_type in ["float16", "float32", "uint16", "uint8"]

    if "int" in from_type:
        offset, scale, data = data
        max_int = 2 ** 8 - 1 if from_type == "uint8" else 2 ** 16 - 1

        return int_dequantize_jit(data, scale, offset, max_int, to_type)
    else:
        return data.astype(to_type)


@dataclass
class NetworkPrecision:
    fwd_act: str
    rev_act: str
    grad: str


if __name__ == "__main__":
    import os

    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda-10.1"

    rng = jax.random.PRNGKey(0)

    r = jax.random.normal(rng, (16, 128, 512))

    q = quantize(r, "uint16")
    d = dequantize(q, "float32")
    assert jnp.allclose(r, d, atol=1e-3, rtol=1e-3)

    q = quantize(r, "uint8")
    d = dequantize(q, "float32")
    assert jnp.allclose(r, d, atol=1e-1, rtol=1e-1)

    q = quantize(r, "float16")
    d = dequantize(q, "float32")
    assert jnp.allclose(r, d, atol=1e-3, rtol=1e-3)

    q = quantize(r, "float32")
    d = dequantize(q, "float32")
    assert jnp.allclose(r, d)
