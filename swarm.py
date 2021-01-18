from multiprocessing.pool import ThreadPool
from typing import Callable

import numpy as np
import optax
import ray
from tensorboardX import SummaryWriter

from embedding_layer import EmbeddingLayer, ProjLayer
from model import SwarmModel
from reversible_layer import ReversibleLayer
from swarm_layer import NetworkPrecision


class Swarm:
    def __init__(self,
                 model: SwarmModel,
                 optimizer: optax.GradientTransformation,
                 loss_scale: float,
                 dataloader: Callable,
                 precision: NetworkPrecision):
        self.model = model
        self.optimizer = optax.chain(
            optax.scale(1 / loss_scale),
            optimizer
        )
        self.dataloader = dataloader
        self.minibatches = 1
        self.loss_scale = loss_scale

        assert ray.is_initialized()  # needs a valid ray cluster to start

        example = self.dataloader()
        self.embedding = EmbeddingLayer.options(max_concurrency=8).remote(example["obs"], self.model.vocab,
                                                                          self.model.d_model, self.optimizer, precision)
        self.embedding.run.remote()

        x = self.embedding.embed_forward.remote(example["obs"])

        self.proj = ProjLayer.options(max_concurrency=8).remote(x, self.model.vocab, self.model.d_model, self.optimizer,
                                                                self.loss_scale, precision)
        self.proj.run.remote()

        self.layers = []
        for i in range(model.rev_layers):
            self.layers.append(
                ReversibleLayer.options(max_concurrency=8).remote(self.model.rev_init, i, x, self.optimizer, precision))

        for l in self.layers:
            l.run.remote()

        self.all_layers = [self.embedding] + self.layers + [self.proj]

    def run(self, epochs, log_path, ckpt_path):
        assert ray.is_initialized()  # needs a valid ray cluster
        writer = SummaryWriter(log_path, flush_secs=5)

        ckpt_loads = [layer.load.remote(f"{ckpt_path}/{i}/") for i, layer in enumerate(self.all_layers)]
        print(f"checkpoint load status: {ray.get(ckpt_loads)}")

        pool = ThreadPool(16)  # have max 16 concurrent examples in the network

        for e in range(epochs):
            if e % 5000 == 0:
                ckpt_saves = [layer.save.remote(f"{ckpt_path}/{i}/", e) for i, layer in enumerate(self.all_layers)]
                ray.wait(ckpt_saves, num_returns=len(ckpt_saves))

                print(f"checkpoint saved")

            data = self.dataloader()

            def map_fn(_):
                return drive_example(self, data)

            result = list(pool.imap_unordered(map_fn, range(1)))  # 1 microbatches per batch
            result = np.array(result)
            error, cos_err, loss = result.mean(axis=0)

            opts = [layers.opt.remote() for layers in self.all_layers]
            ray.wait(opts, num_returns=len(opts))

            writer.add_scalar("loss", loss / self.loss_scale, e)
            writer.add_scalar("reconstruction_error", error, e)
            writer.add_scalar("reconstruction_cos_error", cos_err, e)


# take a training example and shoves it through forward and backward of all layers
def drive_example(swarm: Swarm, data):
    x = swarm.embedding.embed_forward.remote(data["obs"])
    ray.wait([x])

    # wrap all big ray objects in unit tuples to stop implicit .get
    for l in swarm.layers:
        x = l.forward.remote((x,))
        ray.wait([x])

    y_dy, loss = swarm.proj.debed_grad.remote((x,), data["target"])
    ray.wait([y_dy])

    for l in reversed(swarm.layers):
        y_dy = l.backward.remote((y_dy,))
        ray.wait([y_dy])

    error = swarm.embedding.embed_grad.remote(data["obs"], (y_dy,))
    ray.wait([error])

    return *ray.get(error), ray.get(loss)
