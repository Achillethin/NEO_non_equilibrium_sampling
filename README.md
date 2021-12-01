# NEO: Non Equilibrium Sampling on the orbit of a deterministic transform


##### Description of the code

This repo describes the NEO estimator described in the paper *NEO: Non Equilibrium Sampling on the orbit of a deterministic transform* published at NeurIPS 2021 and available https://papers.nips.cc/paper/2021/file/8dd291cbea8f231982db0fb1716dfc55-Paper.pdf. 


Three notebooks describe typical experiments of the main paper. 

- Mix_gaussian, the normalizing constant estimation on a mixture of Gaussian distributions
- Sampler the sampling of a mixture of Gaussian distributions or on Funnel distribution.
- Experiments_colab the training of VAE.


### Requirements

Mainly uses pytorch, pyro-ppl. Later tensorboard

`pip install -r requirements.txt`

### Train a VAE

`python train.py [optional arguments]`
