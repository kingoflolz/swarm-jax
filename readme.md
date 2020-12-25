# Pipelined Swarm Training

Swarm training "framework" using Haiku + Jax + Ray.

Designed for training large language models in a model parallel fashion with unreliable, heterogeneous nodes. (eventually)

Look in `swarm_run.py` for an example of running a character transformer on enwik8.

# TODOs

- [x] Forward passes
- [x] Backward passes with activation reconstruction
- [x] Run optimizer
- [x] Logging
- [x] Checkpointing
- [ ] Data parallelism with multiple nodes per layer and gradient/weight aggregation
- [ ] Heterogeneous nodes with potentially multiple layers per node
- [ ] Handle unbalanced and unreliable nodes (layerdrop)
- [ ] 1T or bust?