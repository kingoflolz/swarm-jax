import optax
import ray

from loader import TextLoader
from ray_tpu import start_ray, get_connection, delete_tpu, create_tpu, wait_til
from swarm_jax.model import SwarmCharTransformer
from swarm_jax.swarm import Swarm
from swarm_jax.swarm_layer import NetworkPrecision

# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda-10.1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["JAX_DEBUG_NANS"] = "True"

head_info = ray.init(dashboard_host="0.0.0.0")
address = head_info['redis_address']

delete_tpu("swarm-jax-test", "europe-west4-a")

conns = []
for i in range(8):
    create_tpu(f"swarm-jax-test-{i}", "europe-west4-a", "v3-8", False)

for i in range(8):
    assert wait_til(f"swarm-jax-test-{i}", "europe-west4-a", {'state': 'READY', 'health': 'HEALTHY'})

for i in range(8):
    conns += get_connection(f"swarm-jax-test-{i}", "europe-west4-a")

# for i in range(8):
#     delete_tpu(f"swarm-jax-test-{i}", "europe-west4-a")

# exit()

for c in conns:
    start_ray(c, address)

train_dataset = TextLoader("data/enwik9", batchsize=128, sample_size=256, length=90000000)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.25),
    optax.adam(2e-4, b1=0.9, b2=0.99, eps=1e-5))

prec = NetworkPrecision(fwd_act="uint16", rev_act="uint16", grad="uint16")

model = SwarmCharTransformer
swarm = Swarm(model, optimizer, 2 ** 16, train_dataset.get_samples, prec)
swarm.run(100000, "runs/512_30L", "ckpt/512_30L")

ray.shutdown()
