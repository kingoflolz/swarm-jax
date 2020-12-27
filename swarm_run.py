import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda-10.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from loader import TextLoader
from model import SwarmCharTransformer
from swarm import Swarm

import ray
import optax

ray.init()

train_dataset = TextLoader("data/enwik8", batchsize=16, sample_size=128, length=90000000)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.25),
    optax.adam(2e-4, b1=0.9, b2=0.99))

model = SwarmCharTransformer
swarm = Swarm(model, optimizer, train_dataset.get_samples)
swarm.run(100000, "runs/512_swarm_thread", "ckpt/512_swarm_thread")

ray.shutdown()
