# Pipelined Swarm Training

Swarm training "framework" using Haiku + Jax + Ray.

Designed for training large language models in a model parallel fashion with unreliable, heterogeneous nodes. (eventually)

# TODOs

- [x] Forward passes
- [x] Backward passes with activation reconstruction
- [ ] Run optimizer
- [ ] Data parallelism with multiple nodes per layer and gradient/weight aggregation
- [ ] Heterogeneous nodes with potentially multiple layers per node
- [ ] Handle unbalanced and unreliable nodes (layerdrop)
- [ ] 1T or bust?