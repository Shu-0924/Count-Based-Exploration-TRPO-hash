# Count-Based-Exploration-TRPO-hash
Our version of **#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning**

You can find original paper from **https://arxiv.org/pdf/1611.04717v3.pdf**

<br>

## Continuous Control
We compared **TRPO-SimHash** with normal **TRPO** in continuous control, and found that Count-Based-Exploration works better in the environment with sparse rewards.

<br>

![](https://cdn.discordapp.com/attachments/967620602997387374/1114129484052971572/image.png)

We use Red line for **TRPO-SimHash** and Blue line for **TRPO**. Also, we make the rewards of ***MountainCar*** the same as in the original paper. Note that there are no available ***SwimmerGather***, so we use ***ant*** instead. 

<br>

## Arcade Learning Environment
We provide following algorithms:
- **TRPO**
- **TRPO-pixel-SimHash**
- **TRPO-BASS-SimHash**
- **TRPO-AE-SimHash**

> *We also provide a notebook for TRPO_baseline which can be run on kaggle*

<br>

![](https://cdn.discordapp.com/attachments/855385710084751373/1118865662425710642/image.png)

<br>

We use 10k batch size and 15M total step to train. Although SimHash makes reward grows fast initially, it always stop at about 20 reward in Freeway environment. We tried the Count-Min Sketch method, but it only gave faster training time and even made the performance worse. We think it may contain some bugs in our SimHash implementation, or maybe our smaller batch size make the result quite different from the paper.

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
