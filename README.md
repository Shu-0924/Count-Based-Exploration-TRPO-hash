# Count-Based-Exploration-TRPO-hash
Our implementation of **#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning**

You can find original paper from **https://arxiv.org/pdf/1611.04717v3.pdf**

<br>

## Continuous Control
We compared **TRPO-SimHash** with normal **TRPO** in continuous control, and found that Count-Based-Exploration works better in the environment with sparse rewards.

<br>

![](https://cdn.discordapp.com/attachments/713005024111755304/1114120137289760858/image.png)
(Red line for **TRPO-SimHash** and Blue line for **TRPO**)

## Arcade Learning Environment
We provide following algorithms:
- **TRPO**
- **TRPO-pixel-SimHash**
- **TRPO-BASS-SimHash**
- **TRPO-AE-SimHash**

> *We also provide a notebook for TRPO_baseline which can be run on kaggle*

<br>

## How to run

#### Python version:
- Python 3.10

#### To install requirements:
    pip install -r requirements.txt

#### To run our code:
- You can just run any .py file directly

<br>

## Reference

Some of the code was modified from these two pages:

1. https://gist.github.com/elumixor/c16b7bdc38e90aa30c2825d53790d217

2. https://github.com/ikostrikov/pytorch-trpo
