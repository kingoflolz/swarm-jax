import os

import jax
from jax.config import config

# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda-10.1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_DEBUG_NANS"] = "True"
os.environ["JAX_PLATFORMS"] = ""

from swarm_jax.swarm_layer import NetworkPrecision

from loader import TextLoader
from swarm_jax.model import SwarmCharTransformer
from swarm_jax.swarm import Swarm

import ray
import optax
import socket
import jax

local_ip = socket.gethostbyname(socket.gethostname())
config.FLAGS.jax_backend_target = "grpc://" + local_ip + ":8470"

ray.init(resources={"tpu": 999}) # pretend we have infinite tpus lol
train_dataset = TextLoader("data/enwik8", batchsize=(8, 16), sample_size=128, length=90000000)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.25),
    optax.adam(2e-4, b1=0.9, b2=0.99, eps=1e-5))

prec = NetworkPrecision(fwd_act="uint16", rev_act="uint16", grad="uint16")

model = SwarmCharTransformer
swarm = Swarm(model, optimizer, 2 ** 16, train_dataset.get_samples, prec)
swarm.run(100000, "runs/512_30L", "ckpt/512_30L")

ray.shutdown()
