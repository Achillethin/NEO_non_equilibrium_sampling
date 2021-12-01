import torch 
import numpy as np 
from torch.distributions import MultivariateNormal, Normal
from models_.transformations import *
from models_.dot_dict import DotDict
import torch.nn as nn

class infine(nn.Module):

    def __init__(self, args):
        
        super().__init__()

        self.K = args.K
        self.device = args.device
        self.dim    = args.dim
        self.target = args.target 
        self.distrib= args.distrib
        self.a_k    = args.a_k
        self.var_p  = args.var_p
        self.n_samples = args.n_samples
        self.hamiltonian = DampedHamiltonian(args.hamiltonian)

        self.importance_distr = args.importance_distr
        self.momentum = MultivariateNormal(torch.zeros(self.dim).to(self.device),
                                           self.var_p * torch.eye(self.dim).to(self.device))
        self.config   = f'K: {self.K}, var_p: {self.var_p}, n_samples: {self.n_samples},gamma: {self.hamiltonian.gamma}, dt: {self.hamiltonian.dt},var_Tp: {self.hamiltonian.var_Tp}, clamp: {self.hamiltonian.clamp}'

    def weights(self, sample_distrib = False):
        
        if sample_distrib:
            q = self.distrib.sample((self.n_samples,)).requires_grad_()
        else:
            q = self.importance_distr.sample((self.n_samples,)).requires_grad_()

        p = self.momentum.sample([self.n_samples]).requires_grad_()

        q_k, p_k = q.clone(), p.clone()
        q_invk, p_invk = q.clone(), p.clone()

        w_0 = (self.importance_distr.log_prob(q_k) + self.momentum.log_prob(p_k)).unsqueeze(0)

        w_forward = torch.cat([w_0, torch.zeros(self.K,self.n_samples).to(self.device)])
        w_backward = torch.cat([torch.zeros(self.K,self.n_samples).to(self.device), w_0])

        log_ratio  = torch.cat([(self.target(q_k) - self.importance_distr.log_prob(q_k)).unsqueeze(0),
                                torch.zeros(self.K, self.n_samples).to(self.device)])
        
        traj = torch.cat([q_k.unsqueeze(0), p_k.unsqueeze(0)], -1)
        
        for k in range(self.K):

            q_k, p_k, log_jac = self.hamiltonian.forward(q_k, p_k, k + 1, self.target)
            q_invk, p_invk, log_jac_inv = self.hamiltonian.inverse(q_invk, p_invk, k + 1, self.target)

            w_forward[k+1] = self.importance_distr.log_prob(q_k) + self.momentum.log_prob(p_k) + log_jac
            w_backward[-k-2] = self.importance_distr.log_prob(q_invk) + self.momentum.log_prob(p_invk) + log_jac_inv

            log_ratio[k+1] = self.target(q_k) - self.importance_distr.log_prob(q_k)

            T_k  = torch.cat([q_k.unsqueeze(0), p_k.unsqueeze(0)], -1)
            traj = torch.cat([traj, T_k], 0)

        weights_num = torch.cat([w_backward[:-1], w_forward], 0)

        del w_forward
        del w_backward

        return weights_num, log_ratio, traj 

    def estimate(self, verbose = False): 

        log_weights = torch.zeros(self.K + 1, self.n_samples).to(self.device)
        weights_num, log_ratio, traj = self.weights()
        
        for k in range(self.K + 1):
            log_weights[k] = weights_num[self.K + k] - torch.logsumexp(weights_num[k:self.K + 1 + k],0)
        
        if verbose:
            print(f'weights: {log_weights.exp()[:,:5]}')
            print(f'ratio: {log_ratio.exp()[:,:5]}')
            print(f'weights * ratio: {(log_weights + log_ratio).exp()[:,:5]}')

        log_cst = torch.logsumexp(log_weights + log_ratio, (0,1)) 

        return log_cst.exp() / self.n_samples, traj

    def estimate_E_T(self, sample_distrib = False):

        log_weights = torch.zeros(self.K + 1, self.n_samples).to(self.device)
        weights_num, log_ratio, traj = self.weights(sample_distrib)

        if sample_distrib:
            samples = traj[0][:,:self.dim]
            loglikelihood = self.target(samples) - self.importance_distr.log_prob(samples)

        for k in range(self.K + 1):
            log_weights[k] = weights_num[self.K + k] - torch.logsumexp(weights_num[k:self.K + 1 + k],0)        

        squared_sum = 2*torch.logsumexp(log_weights + log_ratio,0)

        ratio = squared_sum - loglikelihood

        return torch.logsumexp(ratio,0).exp() / self.n_samples
    
    def classic_IS(self, n_samples):

        q = self.importance_distr.sample((n_samples,))
        log_ratio = self.target(q) - self.importance_distr.log_prob(q)

        return torch.logsumexp(log_ratio, 0).exp() / n_samples
    
    def asymptotic_variance(self, n_samples):

        q = self.distrib.sample((n_samples,))
        log_ratio = self.target(q) - self.importance_distr.log_prob(q)

        return torch.logsumexp(log_ratio,0).exp() / ((self.K+1) * n_samples)