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
- [x] Actually do pipelining
- [x] fp16 with static loss scaling
- [x] Integer quantization for activations and gradients between layers 
- [ ] Get rid of pipeline stalls from running optimizer
- [ ] Data parallelism with multiple nodes per layer and gradient/weight aggregation
- [ ] Heterogeneous nodes with potentially multiple layers per node
- [ ] Handle unbalanced and unreliable nodes (layerdrop)
- [ ] Dynamic node addition
- [ ] 1T or bust?

# To debug
```bash
export RAY_BACKEND_LOG_LEVEL=debug
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_PLATFORMS=''
```