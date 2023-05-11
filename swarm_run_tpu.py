import functools
import multiprocessing

import optax
import ray

from loader import TextLoader
from ray_tpu import start_ray, get_connection, create_tpu, wait_til
from swarm_jax.model import SwarmCharTransformerBig
from swarm_jax.swarm import Swarm
from swarm_jax.swarm_layer import NetworkPrecision

tpus = 8

# for i in range(tpus):
#     delete_tpu(f"swarm-jax-test-{i}", "europe-west4-a")
#
# exit()

head_info = ray.init(dashboard_host="0.0.0.0")
address = head_info['redis_address']
region = "us-central1-f"

conns = []
for i in range(tpus):
    create_tpu(f"swarm-jax-test-{i}", region, "v3-8", False)

for i in range(tpus):
    assert wait_til(f"swarm-jax-test-{i}", region, {'state': 'READY', 'health': 'HEALTHY'})

for i in range(tpus):
    conns += get_connection(f"swarm-jax-test-{i}", region)

with multiprocessing.Pool(processes=tpus) as p:
    p.map(functools.partial(start_ray, address=address), conns)

train_dataset = TextLoader("data/enwik9", batchsize=(8, 8), sample_size=1024, length=90000000)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.25),
    optax.adam(2e-4, b1=0.9, b2=0.99, eps=1e-5))

prec = NetworkPrecision(fwd_act="float32", rev_act="float32", grad="float32")

model = SwarmCharTransformerBig
swarm = Swarm(model, optimizer, 2 ** 16, train_dataset.get_samples, prec)
swarm.run(100000, "runs/512_30L", "ckpt/512_30L")

ray.shutdown()
