#!/usr/bin/env bash

pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install optax ray dm-haiku tensorboardX fabric decorator
mkdir data
wget https://cs.fit.edu/~mmahoney/compression/enwik8.zip -O data/enwik8.zip
wget https://cs.fit.edu/~mmahoney/compression/enwik9.zip -O data/enwik9.zip
unzip data/enwik8.zip -d data/
unzip data/enwik9.zip -d data/