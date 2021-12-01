#!/bin/bash

python3 train.py --model vae --epoch 1 --subset 100
python3 train.py --model vae --epoch 1 --subset 100 --archi convMnist --flatten 0
python3 train.py --model iwae --epoch 1 --device gpu --subset 100
python3 train.py --model iwae --epoch 1 --device gpu --subset 100 --archi large
python3 train.py --model flowvae --epoch 1 --subset 100
python3 train.py --model neqvae --epoch 1 --subset 100
python3 train.py --dataset mnist --model neqvae --epoch 1 --device gpu --subset 100 --archi large
python3 train.py --dataset mnist --model neqvae --epoch 1 --device gpu --subset 100 --archi large --logvar_p adaptative
python3 train.py --dataset mnist --model neqvae2 --epoch 1 --device gpu --subset 100
python3 train.py --dataset cifar --model vae --epoch 1 --subset 100 --archi convCifar --binarize 0 --flatten 0
python3 train.py --dataset cifar --model iwae --epoch 1 --subset 100 --archi convCifar --binarize 0 --flatten 0
