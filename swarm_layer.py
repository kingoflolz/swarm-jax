"""
Common methods for layer actors
"""
import functools
import pickle
import re
from glob import glob
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
# TODO: more intellegent checkpoint saving with deleting old checkpoints etc
import ray


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


@functools.partial(jax.jit, donate_argnums=(0, 1, 2), static_argnums=3)
def opt_jit(grad_acc, opt_state, params, optimizer):
    updates, new_opt_state = optimizer.update(grad_acc, opt_state)
    new_params = optax.apply_updates(params, updates)

    new_grad_acc = jax.tree_map(jnp.zeros_like, grad_acc)
    return new_grad_acc, new_opt_state, new_params


def opt_state(state, optimizer):
    new_grad_acc, new_opt_state, new_params = opt_jit(state["grad_acc"],
                                                  state["opt_state"],
                                                  state["params"],
                                                  optimizer)

    state["grad_acc"] = new_grad_acc
    state["opt_state"] = new_opt_state
    state["params"] = new_params
    state["grad_count"] = np.array(0)
    return state


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
def run_threads(fwd_in_q: Queue, bwd_in_q: Queue, queue_size: int, fwd_fn: Callable, bwd_fn: Callable):
    fwd_out_q = Queue(queue_size)
    bwd_out_q = Queue(queue_size)

    fwd_get = GetThread(fwd_in_q, fwd_out_q)
    bwd_get = GetThread(bwd_in_q, bwd_out_q)
    run = RunThread(fwd_out_q, bwd_out_q, fwd_fn, bwd_fn)

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
