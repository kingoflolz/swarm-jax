#!/usr/bin/env bash
# initializes jax and installs ray on cloud TPUs

sudo pip install --upgrade jaxlib==0.1.59
sudo pip install --upgrade jax ray fabric dataclasses optax git+https://github.com/deepmind/dm-haiku