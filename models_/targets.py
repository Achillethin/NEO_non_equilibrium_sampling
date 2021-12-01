import torch
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical, Normal, Cauchy
import numpy as np
import torch.nn as nn 
import math 

class Target(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def log_prob(self):
        raise NotImplementedError
        
    def sample(self):
        raise NotImplementedError
        
    
class GaussianMixture(Target):
    
    def __init__(self, args):
        
        super().__init__()
        
        self.dim      = args['dim']
        self.mixtures = args['mixtures']
        self.means    = args['means']
        
        assert self.means.shape == (self.mixtures, self.dim)
        
        if args['weights'] is None:
            self.weights = torch.tensor([1/self.mixtures]*self.mixtures)
            
        else:
            self.weights  = args['weights']
            assert self.weights.shape == (self.mixtures,)
        
        if args['covs'].dim() == 1:

            if args['covs'].shape != (self.mixtures,) and args['covs'].shape != (1,):
                raise ValueError('length of variance vector must be equal to mixture or 1')
            
            else: 
                self.covs = args['covs'].view(-1,1,1) * torch.eye(self.dim).unsqueeze(0).repeat(self.mixtures,1,1)
                
        elif args['covs'].shape == (self.mixtures,self.dim,self.dim):
            self.covs = args['covs']
            
        else: 
            raise ValueError('incorrect cov matrix')
            
        mvns  = MultivariateNormal(self.means, self.covs)
        categorical = Categorical(self.weights / torch.sum(self.weights))
        
        self.distrib = MixtureSameFamily(categorical, mvns)
        

    def log_prob(self,x):
        
        return self.distrib.log_prob(x)
    
    def sample(self, N_samples):
        """
        N_samples must be tuple
        """
        return self.distrib.sample(N_samples)
    
class Funnel(Target):

    def __init__(self, args):
        
        super().__init__()
        self.device = args['device']
        self.a = torch.tensor(1.).to(self.device)
        self.b = torch.tensor(0.5).to(self.device)
        self.dim = args['dim']

        self.distrib_x1 = Normal(torch.zeros(1).to(self.device), torch.tensor(self.a).to(self.device))
        
    def log_prob(self, x):

        log_probx1 = self.distrib_x1.log_prob(x[:,0].unsqueeze(1))

        logprob_rem = (- x[:,1:] ** 2 * (-2*self.b*x[:,0].unsqueeze(-1)).exp() - \
                      2*self.b*x[:,0].unsqueeze(-1) - torch.tensor(2 * math.pi).log())/2
        #distrib_rem = Normal(torch.zeros(x.shape[0],1), (self.b*x[:,0].unsqueeze(1)).exp())
        #log_probrem = distrib_rem.log_prob(x[:,1:])
        logprob_rem = logprob_rem.sum(-1)

        return (log_probx1 + logprob_rem.unsqueeze(-1)).flatten()

    def sample(self, n_samples):

        x1 = self.distrib_x1.sample(n_samples)

        rem = torch.randn(n_samples + (self.dim-1,)) * (self.b * x1).exp() 

        return torch.cat([x1, rem], -1)

class Cauchy_(Target): 

    def __init__(self, args):
        
        super().__init__()
        self.loc = args['loc']
        self.scale = args['scale']
        self.dim   = args['dim']
        self.device = args['device']

        self.cauchy = Cauchy(self.loc, self.scale)

    def log_prob(self, x):
        return self.cauchy.log_prob(x).sum(-1)

    def sample(self, n_samples):
        return self.cauchy.sample(n_samples + (self.dim,))

class Cauchy_mixture(Target): 

    def __init__(self, args):
        
        super().__init__()
        self.loc   = args['loc']
        self.scale = args['scale']
        self.dim   = args['dim']
        self.device = args['device']

        cat = Categorical(torch.tensor([0.5,0.5]).to(self.device))
        cauchy = Cauchy(torch.tensor([-self.loc,self.loc]).to(self.device),
                        torch.tensor([self.scale,self.scale]).to(self.device))

        self.mixture = MixtureSameFamily(cat, cauchy)
    
    def log_prob(self, x):
        return self.mixture.log_prob(x).sum(-1)

    def sample(self, n_samples):
        return self.mixture.sample(n_samples + (self.dim,))
        
        