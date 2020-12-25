"""
Common methods for layer actors
"""
import pickle
import os
import re
from glob import glob
from pathlib import Path


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