import math
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import gamma, invgamma
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
torchType = torch.float32
#import hamiltorch

class Target(nn.Module):
    """
    Base class for a custom target distribution
    """

    def __init__(self, kwargs):
        super().__init__()
        self.device = kwargs.device
        self.torchType = torchType
        self.device_zero = torch.tensor(0., dtype=self.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=self.torchType, device=self.device)

    def prob(self, x):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def log_prob(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def sample(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

class Gaussian_target(Target):
    """
    1 gaussian (multivariate)
    """

    def __init__(self, kwargs):
        super(Gaussian_target, self).__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType        
        self.mu = kwargs['mu']  # list of locations for each of these gaussians
        self.cov = kwargs['cov']  # list of covariance matrices for each of these gaussians
        self.dist = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.cov, device = 'cpu', dtype = torchType)

    def get_density(self, x):
        """
        The method returns target density
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
         
        return self.log_prob(x).exp()

    def log_prob(self, x):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        return self.dist.log_prob(x)
    
    def sample(self, n=1):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        
        return self.dist.sample(n)
    
    
class Gaussian_mixture(Target):
    
    """
    Mixture of n gaussians (multivariate)
    """

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.num = kwargs['num_gauss']
        self.pis = torch.tensor(kwargs['p_gaussians'])
        self.locs = torch.cat([*kwargs['locs']]).view(self.num,-1)  
        self.covs = torch.cat([*kwargs['covs']]).view(self.num,self.locs.shape[1],-1)

        self.mixture = MixtureSameFamily(Categorical(self.pis),
                                         MultivariateNormal(self.locs, self.covs))

    def get_density(self, x):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        density = self.log_prob(x).exp()
        return density

    def log_prob(self, z, x=None):

        return self.mixture.log_prob(z) + torch.log(torch.tensor(torch.sum(self.pis)))
    
    def sample(self, n, x=None):
        return self.mixture.sample(n)
    
    
class Cauchy_mixture(Target):
    
    def __init__(self, kwargs):
        
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.mu  = kwargs['mu']
        self.cov = kwargs['cov']
        
    def get_density(self,x):
        
        return self.log_prob(x).exp()
    
    def log_prob(self, z, x = None):

        cauchy = Cauchy(self.mu, self.cov)
        cauchy_minus = Cauchy(-self.mu, self.cov)

        catted = torch.cat([cauchy.log_prob(z)[None,...],cauchy_minus.log_prob(z)[None,...]],0)

        log_target = torch.sum(torch.logsumexp(catted, 0) - torch.tensor(2.).log(),-1)
        
        return log_target + torch.tensor([8.]).log()


'''class Gaussian_mixture(Target):
    """
    Mixture of n gaussians (multivariate)
    """

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.num = kwargs['num_gauss']
        self.pis = kwargs['p_gaussians']
        self.locs = kwargs['locs']  # list of locations for each of these gaussians
        self.covs = kwargs['covs']  # list of covariance matrices for each of these gaussians
        self.peak = [None] * self.num
        for i in range(self.num):
            self.peak[i] = torch.distributions.MultivariateNormal(loc=self.locs[i], covariance_matrix=self.covs[i])

    def get_density(self, x):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        density = self.log_prob(x).exp()
        return density

    def log_prob(self, z, x=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent variable
        Output:
        log_density - log p(x)
        """
        log_p = torch.tensor([], device=self.device)
        #pdb.set_trace()
        for i in range(self.num):
            log_paux = (torch.log(self.pis[i]) + self.peak[i].log_prob(z)).view(-1, 1)
            log_p = torch.cat([log_p, log_paux], dim=-1)
        log_density = torch.logsumexp(log_p, dim=-1)  # + torch.tensor(1337., device=self.device)
        return log_density'''


class Uniform(Target):
    ##Uniform on a box of dimension dim
    def __init__(self,kwargs):
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.xmin = kwargs.x_min
        self.xmax = kwargs.x_max
        self.dim = kwargs.dim
        self.dist = torch.distributions.Uniform(self.xmin,self.xmax)
        
    def get_density(self, x):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        Output:
        log p(x)
        """
        
        return self.dist.log_prob(x).exp()
        # You should define the class for your custom distribution

    def log_prob(self,  x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        return self.dist.log_prob(x).sum(-1)

    def sample(self, n=(1,)):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        samples =self.dist.sample(n + (self.dim,))
        return samples                                     





class BNAF_examples(Target):

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.data = kwargs.bnaf_data
        self.max = kwargs.qmax
        self.min = kwargs.qmin

    def log_prob(self, z, x=None):
        add_0 = torch.zeros_like(z).sum(-1)
        if (torch.max(z)>self.max)|(torch.min(z)<self.min):
            add_0 = -(5* z**2).sum(-1)
        if self.data == 't1':
            if len(z.shape) == 1:
                z = z.view(-1, 2)
            z_norm = torch.norm(z, 2, -1)
            add1 = 0.5 * ((z_norm - 2) / 0.3) ** 2
            add2 = - torch.log(torch.exp(-0.5 * ((z[..., 0] - 2) / 0.6) ** 2) + \
                               torch.exp(-0.5 * ((z[..., 0] + 2) / 0.6) ** 2) + 1e-9)
            return -add1 - add2 + add_0

        elif self.data == 't2':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[..., 0] / 4)
            return -0.5 * ((z[..., 1] - w1) / 0.4) ** 2 + add_0
        elif self.data == 't3':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[..., 0] / 4)
            w2 = 3 * torch.exp(-0.5 * ((z[..., 0] - 1) / 0.6) ** 2)
            in1 = torch.exp(-0.5 * ((z[..., 1] - w1) / 0.35) ** 2)
            in2 = torch.exp(-0.5 * ((z[..., 1] - w1 + w2) / 0.35) ** 2)
            return torch.log(in1 + in2 + 1e-9) + add_0
        elif self.data == 't4':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[..., 0] / 4)
            w3 = 3 * torch.sigmoid((z[..., 0] - 1) / 0.3)
            in1 = torch.exp(-0.5 * ((z[..., 1] - w1) / 0.4) ** 2)
            in2 = torch.exp(-0.5 * ((z[..., 1] - w1 + w3) / 0.35) ** 2)
            return torch.log(in1 + in2 + 1e-9) + add_0
        else:
            raise RuntimeError

    def get_density(self, z, x=None):
        density = self.distr.log_prob(z).exp()
        return density

    def sample(self, n):
        return torch.stack(hamiltorch.sample(log_prob_func=self.log_prob, params_init=torch.zeros(2),
                                             num_samples=n, step_size=.1, num_steps_per_sample=20))

    
    
    
    
    
    
class Funnel(Target):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.a = 1.*torch.ones(1)
        self.b = .5
        self.dim = kwargs.dim
        self.distr1 = torch.distributions.Normal(torch.zeros(1), self.a)
        
    def log_prob(self, z, x=None):
        #pdb.set_trace()
        logprob1 = self.distr1.log_prob(z[...,0])
        z1 = z[..., 0]
        #logprob2 = self.distr2(z[...,0])
        logprob2 = -(z[...,1:]**2).sum(-1) * (-2*self.b*z1).exp()/2. - (self.dim-1)/2 * np.log(2*np.pi) - (self.dim-1) * self.b*z1 
        return logprob1+logprob2
    
    
    
class Banana_32(Target):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.Q = 1.*torch.ones(1)
        self.dim = kwargs.dim
        
        
        
    def log_prob(self, z, x=None):
        n = self.dim/2
        even = np.arange(0, self.dim, 2)
        odd = np.arange(1, self.dim, 2)
        
        ll = - (z[..., even] - z[..., odd]**2)**2/self.Q - (z[..., odd]-1)**2   
        return ll.sum(-1)