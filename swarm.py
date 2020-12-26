from typing import Callable

import optax
import ray
from tensorboardX import SummaryWriter

from embedding_layer import EmbeddingLayer, ProjLayer
from model import SwarmModel
from reversible_layer import ReversibleLayer


class Swarm:
    def __init__(self,
                 model: SwarmModel,
                 optimizer: optax.GradientTransformation,
                 dataloader: Callable):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.minibatches = 1

        assert ray.is_initialized()  # needs a valid ray cluster to start

        example = self.dataloader()
        self.embedding = EmbeddingLayer.remote(example["obs"], self.model.vocab, self.model.d_model, self.optimizer)

        x = self.embedding.embed_forward.remote(example["obs"])

        self.proj = ProjLayer.remote(x, self.model.vocab, self.model.d_model, self.optimizer)

        self.layers = []
        for i in range(model.rev_layers):
            self.layers.append(ReversibleLayer.options(max_concurrency=8).remote(self.model.rev_init, i, x, optimizer))

        for l in self.layers:
            l.run.remote()

        self.all_layers = [self.embedding] + self.layers + [self.proj]

    def run(self, epochs, log_path, ckpt_path):
        assert ray.is_initialized()  # needs a valid ray cluster
        writer = SummaryWriter(log_path, flush_secs=5)

        ckpt_loads = [layer.load.remote(f"{ckpt_path}/{i}/") for i, layer in enumerate(self.all_layers)]
        print(f"checkpoint load status: {ray.get(ckpt_loads)}")

        for e in range(epochs):
            if e % 1000 == 0:
                ckpt_saves = [layer.save.remote(f"{ckpt_path}/{i}/", e) for i, layer in enumerate(self.all_layers)]
                ray.wait(ckpt_saves, num_returns=len(ckpt_saves))

                print(f"checkpoint saved")

            data = self.dataloader()

            error, loss = drive_example(self, data)

            opts = [layers.opt.remote() for layers in self.all_layers]
            ray.wait(opts, num_returns=len(opts))

            writer.add_scalar("loss", loss, e)
            writer.add_scalar("reconstruction_error", error, e)


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

    return ray.get(error), ray.get(loss)