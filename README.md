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

### To cite this work

If you use this repository, please reference our article e.g. using bibtex

`@inproceedings{thin2021neo,
  title={NEO: Non Equilibrium Sampling on the Orbits of a Deterministic Transform},
  author={Thin, Achille and El Idrissi, Yazid Janati and Le Corff, Sylvain and Ollion, Charles and Moulines, Eric and Doucet, Arnaud and Durmus, Alain and Robert, Christian P},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}`

or other formats available at https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&authuser=1&q=neo+non+equilibrium&btnG=&oq=neo+no#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3AeV5WBKEHRfkJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den%26authuser%3D1.
