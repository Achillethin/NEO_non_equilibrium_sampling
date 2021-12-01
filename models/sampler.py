import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from models.utils import binary_crossentropy_logits_stable, logprob_normal
from models.transformations import LeapFrog, DampedHamiltonian, BaseTransformation, Transfo_RNVP, DampedHamiltonian_lf
from models.flows import RNVP
from models.evidence import NonEq_Estimator_w_a, Estimator

from tqdm import tqdm

from utils.plotting import plot_traj_coloured
import pdb





call_distr = False



class ISIR(nn.Module):
    """
   Iterated Sampling Importance Resampling scheme.
    Supports importance distribution. 
    """

    def __init__(self, dim, num_samples, prior, importance_distr, verbose = False):
        super().__init__()
        self.estimator = Estimator(dim, num_samples-1, prior, 
                                             importance_distr)
        self.verbose = verbose
        self.dim = dim
        self.num_samples = num_samples
        self.importance_distr= importance_distr
        
    def sample_step(self, z, w, i, y, loglikelihood, x=None):
        #pdb.set_trace()
        traj_cur = z[i][:,0] ###We have traj_cur[k]= y
        shape =traj_cur.shape
        weights_cur = w[i][:,0]
        
        weights_new, traj_new = self.estimator.log_estimate_gibbs_correlated(traj_cur, loglikelihood, x, gibbs =True, n_chains = shape[0])
        weights_tot = torch.cat([weights_cur.view((1,)+shape[:-1]), weights_new], dim=0)
        traj_tot = torch.cat([traj_cur.view((1,)+shape), traj_new], dim=0)
                
        i = torch.multinomial((weights_tot - torch.logsumexp(weights_tot, dim=0)).exp().transpose(0,1), 1)[:,0]
        weights_cur = weights_tot[i][:,0]
        traj_cur = traj_tot[i][:,0]
        
        if ((torch.sum(torch.ones_like(i)[i!=0])>0) &self.verbose):
            print('changed point')
            
        return traj_tot, weights_tot, i, traj_cur
    
    
    def chain_sample(self, n, n_chain, loglikelihood, x=None):
        if callable(self.importance_distr)&call_distr:
            z = self.importance_distr(x).sample((self.num_samples, n_chain, ))
        else:
            z = self.importance_distr.sample((self.num_samples, n_chain, ))
        
        i= torch.tensor([0]*n_chain)
        y=None
        log_w = torch.zeros_like(z[...,0]).log()
        samples = torch.tensor(())
        n_eff = 0
        #pdb.set_trace()
        for _ in tqdm(range(n)):
            z, log_w, i,  y = self.sample_step(z, log_w, i,  y, loglikelihood, x)
            samples = torch.cat([samples, y[None,...]], dim=0)
            n_eff+= torch.sum(torch.ones_like(i)[i!=0])/n_chain
        return samples, n_eff
        





class Neq_Gibbs_sampler(nn.Module):
    """
   Sampler Non eq Gibbs.
    Supports importance distribution. 
    """

    def __init__(self, dim, num_samples, prior, importance_distr, transformation_params, a, logvar_p = torch.tensor(0., dtype = torch.float32), logvar_p_transfo = torch.tensor(0., dtype = torch.float32), verbose = False, vae = None):
        super().__init__()
        self.estimator = NonEq_Estimator_w_a(dim, num_samples-1, prior, 
                                             importance_distr, transformation_params, a,
                                             sample_gibbs = True, logvar_p= logvar_p,  logvar_p_transfo = logvar_p_transfo)
        self.verbose = verbose
        self.K = a.shape[0]
        self.dim = dim
        self.num_samples = num_samples
        self.importance_distr= importance_distr
        self.vae = vae
        
    def sample_step(self, z, w, i, k, y, loglikelihood, x=None):
        #pdb.set_trace()
        traj_cur = z[:,i][:,:,0] ###We have traj_cur[k]= y
        shape = traj_cur[k][0].shape
        weights_cur = w[:,i][:,:,0]
        
        weights_new, traj_new = self.estimator.log_estimate_gibbs_correlated(traj_cur, loglikelihood, x)
        weights_tot = torch.cat([weights_cur.view((self.K,1,)+shape[:-1]), weights_new], dim=1)
        traj_tot = torch.cat([traj_cur.view((self.K,1,)+shape), traj_new], dim=1)
        
        est_traj = torch.logsumexp(weights_tot, dim=0)
        est_traj[torch.isnan(est_traj)] = torch.zeros(1).log()
        i = torch.multinomial((est_traj - torch.logsumexp(est_traj, dim=0)).exp().transpose(0,1), 1)[:,0]
        weights_cur = weights_tot[:,i][:,:,0]
        traj_cur = traj_tot[:,i][:,:,0]
        k = torch.multinomial( (weights_cur - torch.logsumexp(weights_cur, dim=0)).exp().transpose(0,1), 1)[:,0]
        if self.vae is not None:
            self.plot_traj(traj_cur, k)
        if ((torch.sum(torch.ones_like(i)[i!=0])>0) &self.verbose):
            print('changed point')
            
        return traj_tot, weights_tot, i, k, (traj_cur[k][:,0])[..., :self.dim]
    
    
    def plot_traj(self, traj, k):
        plt.figure(figsize=(16*len(traj),6))
        for i,o in enumerate(traj):
            to_plot = self.vae.decode(o)
            plt.subplot(1,len(traj),1+i)
            plt.imshow(to_plot[0].detach().numpy().transpose(1,2,0))
            if k==i:
                plt.xlabel('choosed point')
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        return
    
    def chain_sample(self, n, n_chain, loglikelihood, x=None, vae = None):
        if callable(self.importance_distr)&call_distr:
            q = self.importance_distr(x).sample((self.K, self.num_samples, n_chain, ))
        else:
            q = self.importance_distr.sample((self.K, self.num_samples, n_chain, ))
        
        p = torch.randn_like(q)
        z = torch.cat([q,p], dim = -1)
        i= torch.tensor([0]*n_chain)
        k= torch.tensor([0]*n_chain)
        y=None
        log_w = torch.zeros_like(z[...,0]).log()
        samples = torch.tensor(())
        n_eff = 0
        #pdb.set_trace()
        for _ in tqdm(range(n)):
            z, log_w, i, k, y = self.sample_step(z, log_w, i, k, y, loglikelihood, x)
            samples = torch.cat([samples, y[None,...]], dim=0)
            n_eff+= torch.sum(torch.ones_like(i)[i!=0])/n_chain
        return samples, n_eff
        
        
        