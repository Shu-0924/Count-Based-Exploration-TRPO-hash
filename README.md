# Count-Based-Exploration-TRPO-hash
Our implementation of "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning": https://arxiv.org/pdf/1611.04717v3.pdf

<br>

## Continuous Control
We compared TRPO-SimHash with normal TRPO in continuous control, and found that Count-Based-Exploration works better in the environment with sparse rewards.

<br>

## Arcade Learning Environment
We provide following algorithms: 
- TRPO
- TRPO-pixel-SimHash
- TRPO-BASS-SimHash
- TRPO-AE-SimHash

> We also provide a notebook for TRPO_baseline which can be run on kaggle

<br>

## How to run

### Python version:
> Python 3.10

### To install requirements:
    pip install -r requirements.txt

### Run our code:
- You can just run any .py file directly

<br>

## Reference

Some of the code was modified from these two pages:
1. https://gist.github.com/elumixor/c16b7bdc38e90aa30c2825d53790d217
2. https://github.com/ikostrikov/pytorch-trpo
