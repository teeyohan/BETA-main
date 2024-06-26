# BETA-main

PyTorch implementation of "BETA: An Active Learning Strategy through Weight Perturbation for Images, Texts, and Molecules".

<div align="center">
  <img width="90%" alt="" src="beta.png">
</div>

The code includes the implementations of all the baselines presented in the paper. Parts of the code are borrowed from https://github.com/AminParvaneh/alpha_mix_active_learning.

## Setup
The dependencies are in requirements.txt. Python=3.8 is recommended for the installation of the environment.

## Dataset
It is recommended to download the following datasets from the official website and place them in the "BETA-main/data/" directory:
- MNIST: https://yann.lecun.com.
- IMDB: https://developer.imdb.com.
- BACE: https://moleculenet.org.

## Training
For training and evaluation, use the following script:

- `python main.py`
