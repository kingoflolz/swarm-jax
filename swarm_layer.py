"""
Common methods for layer actors
"""
import functools
import pickle
import re
from glob import glob
from pathlib import Path

import jax
import optax
import jax.numpy as jnp
import numpy as np

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


@functools.partial(jax.jit, donate_argnums=(0, 1, 2), static_argnums=3)
def opt(grad_acc, opt_state, params, optimizer):
    updates, new_opt_state = optimizer.update(grad_acc, opt_state)
    new_params = optax.apply_updates(params, updates)

    new_grad_acc = jax.tree_map(jnp.zeros_like, grad_acc)
    return new_grad_acc, new_opt_state, new_params


def opt_state(state, optimizer):
    new_grad_acc, new_opt_state, new_params = opt(state["grad_acc"],
                                                  state["opt_state"],
                                                  state["params"],
                                                  optimizer)

    state["grad_acc"] = new_grad_acc
    state["opt_state"] = new_opt_state
    state["params"] = new_params
    state["grad_count"] = np.array(0)
    return state