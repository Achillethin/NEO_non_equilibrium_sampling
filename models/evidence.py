import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from random import randrange

import matplotlib.pyplot as plt

from models.utils import binary_crossentropy_logits_stable, logprob_normal
from models.transformations import LeapFrog, DampedHamiltonian, BaseTransformation, Transfo_RNVP, DampedHamiltonian_lf
from models.flows import RNVP

from utils.plotting import plot_traj_coloured
import pdb



call_distr = True
###TODO If estimate VAES, then importance distr is a function lambda x: importance_distr(mu(x), sigma(x))


class Estimator(nn.Module):
    """
    Estimator base class.
    Supports importance distribution. The importance distribution can be:
    - None
    - A distribution (with log_prob and sample callable)
    - A function of an input, returning a distribution
    """
    def __init__(self, dim, num_samples, prior, importance_distr=None):
        super().__init__()
        self.n = num_samples
        self.dim = dim
        self.prior = prior

        self.importance_sampling = importance_distr is not None

        if self.importance_sampling:
            self.importance_distr = importance_distr


    def functional(self, loglikelihood, samples, x):
        # TODO: make sure it works regardless of the shape (here i have a sum(-1) for vaes)
        with torch.no_grad():
            if self.importance_sampling:
                if callable(self.importance_distr)&call_distr:
                    return (loglikelihood(samples, x=x) + self.prior.log_prob(samples).sum(-1) - self.importance_distr(x).log_prob(samples).sum(-1))
                else:
                    return (loglikelihood(samples, x=x) + self.prior.log_prob(samples) - self.importance_distr.log_prob(samples))
            else:
                return (loglikelihood(samples, x=x))
    
    
    
            
            
        
    def M(self, q_0_curr, loglikelihood, x, stepsize = 1e-2):
        ###Do nothing -- will just change momentum
        #return q_0_curr
        ####RWM
        u = torch.randn_like(q_0_curr)
        if callable(self.importance_distr)&call_distr:
            current_target = lambda q, x: self.importance_distr(x).log_prob(q).sum(-1) 
        else:
            current_target = lambda q, x=None: self.importance_distr.log_prob(q)
        
        q_0_prop = q_0_curr + stepsize*u
        log_t = current_target(q_0_prop, x) - current_target(q_0_curr, x)
        log_rand = torch.log(torch.rand_like(log_t))

        accept = (log_rand <= log_t) ###Should be min(0, log_t) but this expression would then still be correct

        q_0_prop[~accept] = q_0_curr[~accept] ##Set the non accepted to old state
        return q_0_prop

    def log_estimate(self, loglikelihood, x, gibbs = False, n_chains = 1):

        if self.importance_sampling:
            if callable(self.importance_distr)&call_distr:
                if gibbs:
                    samples = self.importance_distr(x).sample((self.n,n_chains,))
                else:
                    samples = self.importance_distr(x).sample((self.n,))
            else:
                if gibbs:
                    samples = self.importance_distr.sample((self.n,n_chains,))
                else:
                    samples = self.importance_distr.sample((self.n,))
        else:
            samples = self.prior.sample((self.n, x.shape[0]))
        
        log_f = self.functional(loglikelihood, samples, x)
        if gibbs:
            return log_f, samples
        f = torch.logsumexp(log_f, dim=0) - np.log(self.n)
        return f ###This returns a vector of size batch_size(x)

    def estimate(self, loglikelihood, x=None):
        return self.log_estimate(loglikelihood, x).exp()
    
    
    def log_estimate_gibbs_correlated(self,z_0_curr, loglikelihood, x, gibbs = True, n_chains = 1):
        l = 1#self.n
        if callable(self.importance_distr)&call_distr:
            z = self.importance_distr(x).sample((self.n,))
        else:
            z = self.importance_distr.sample((self.n,) + z_0_curr.shape[:-1])
        z[0] = self.M(z_0_curr, loglikelihood, x)
        for i in range(1,l):
            z[i] =self.M(z[i-1], loglikelihood, x)
        
        
        log_f = self.functional(loglikelihood, z, x)

        return log_f, z


class NonEq_Estimator_w_a(Estimator):

    def __init__(self, dim, num_samples, prior, importance_distr, transformation_params, a, sample_gibbs = False,logvar_p = torch.tensor(0., dtype = torch.float32), logvar_p_transfo = torch.tensor(0., dtype = torch.float32)):
        super().__init__( dim, num_samples, prior, importance_distr)

        t_params = transformation_params.copy()
        transformation_name = t_params.pop('name')
        if transformation_name == 'LeapFrog':
            self.transformation = LeapFrog( **t_params)
            self.flows_different = False
            self.hamiltonian = True
        elif transformation_name == 'DampedHamiltonian':
            self.transformation = DampedHamiltonian(**t_params)
            self.flows_different = False
            self.hamiltonian = True
        elif transformation_name == 'DampedHamiltonian_vde':
            self.transformation = DampedHamiltonian_vde(**t_params)
            self.flows_different = False
            self.hamiltonian = True
        elif transformation_name == 'DampedHamiltonian_maddison':
            self.transformation = DampedHamiltonian_maddison(**t_params)
            self.flows_different = False
            self.hamiltonian = True
        elif transformation_name == 'DampedHamiltonian_wo':
            self.transformation = DampedHamiltonian_wo(**t_params)
            self.flows_different = False
            self.hamiltonian = True
        elif transformation_name == 'DampedHamiltonian_lf':
            self.transformation = DampedHamiltonian_lf(**t_params)
            self.flows_different = False
            self.hamiltonian = True
        elif transformation_name == 'Identity':
            self.transformation = BaseTransformation(dim=dim)
            self.flows_different = False
            self.hamiltonian = False
        elif transformation_name == 'Real-NVP':
            self.transformation = Transfo_RNVP(dim = dim, **t_params)
            self.flows_different = True
            self.hamiltonian = False
        else:
            raise ValueError

        self.a = a
        self.K = a.shape[0]
        self.latent_dim=dim
        self.gibbs = sample_gibbs
        self.logvar_p = logvar_p ###Can be scalar or vector of size dim
        self.logvar_p_transfo = logvar_p_transfo

    def compute_and_push(self, x, z, loglikelihood, log_jac, backward = False, k=None):
        ###First we compute the weights and the value of the current pushforward
        if self.hamiltonian:
            if callable(self.importance_distr)&call_distr:
                q_phi_k = self.importance_distr(x).log_prob(z[...,:self.dim]).sum(-1)
            else:
                q_phi_k = self.importance_distr.log_prob(z[...,:self.dim])
            w =  q_phi_k + \
                        logprob_normal(z[...,self.dim:], logvar = self.logvar_p).sum(-1) + \
                        log_jac
        else:
            q_phi_k = self.importance_distr.log_prob(z)
            w =  q_phi_k + \
                        log_jac
####Then we push, backward or forward
        if backward:
            joint_ll = None
            if self.flows_different:
                z, log_jac_cur = self.transformation.inverse(z=z, x=x, k=k, logvar_p = self.logvar_p_transfo)
            else:
                z, log_jac_cur = self.transformation.inverse(z=z, target = loglikelihood, x=x, logvar_p = self.logvar_p_transfo)
                #z, log_jac_cur = self.transformation.inverse(z=z, target = loglikelihood, x=x)
        else:
            if self.hamiltonian:
                if callable(self.importance_distr)&call_distr:
                    joint_ll = loglikelihood(z[...,:self.latent_dim], x) + \
                    self.prior.log_prob(z[...,:self.latent_dim]).sum(-1)
                else: 
                    joint_ll = loglikelihood(z[...,:self.latent_dim], x) + \
                    self.prior.log_prob(z[...,:self.latent_dim])
            else:
                joint_ll = loglikelihood(z, x) + self.prior.log_prob(z)
            if self.flows_different:
                z, log_jac_cur = self.transformation(z=z, x=x, k=k, logvar_p = self.logvar_p_transfo)
            else:
                z, log_jac_cur = self.transformation(z=z, target = loglikelihood, x=x, logvar_p = self.logvar_p_transfo)
                #z, log_jac_cur = self.transformation(z=z, target = loglikelihood, x=x)
        log_jac += log_jac_cur
### Return updated point and jacobian plus weights and relevant quantities to compute
        return w, joint_ll, q_phi_k, z, log_jac


    def weights(self, x, z, loglikelihood):
        #pdb.set_trace()
        shape = (self.K,)+ z.shape[:-1]
        w = torch.zeros(shape) ###tensor containing numerator of the weights
        joint_ll = torch.zeros(shape)
        q_phi_k = torch.zeros(shape)

        z_num = z ##point going through the flow at numerator at each step k
        log_jac_num = torch.zeros_like(z[...,0])
        z_traj = None
        if self.gibbs:
            z_traj = torch.zeros((self.K,)+z.shape)
        #########For plotting trajectories##########
        plotting = (len(z.shape)==2)& False
        if plotting:
            idx = [0,1]
            z_plot = torch.zeros((len(idx), self.K, z.shape[-1]))
            z_plot[:, 0, self.latent_dim:]= z[idx, self.latent_dim:] 
         ######################   
        w_den = torch.zeros((self.K,)+ shape)
        z_den= z.clone()
        log_jac_den = torch.zeros_like(log_jac_num)
        w_minus = torch.zeros_like(w) ###Contains a_-k q_phi(T^-k(x)) J_T^-k(x)
        for k in range(self.K):
            if self.gibbs:
                z_traj[k] = z_num
            w[k], joint_ll[k], q_phi_k[k], z_num, log_jac_num =  self.compute_and_push(x, z_num, 
                                                                                       loglikelihood, 
                                                                                       log_jac_num)
            w_minus[k], _, _, z_den, log_jac_den = self.compute_and_push(x, z_den, 
                                                                         loglikelihood, 
                                                                         log_jac_den, backward=True)
 
            ###############Plotting#######
            if (plotting):
                if k<(self.K-1):
                    z_plot[:,k+1, self.latent_dim:] = z_num[idx, self.latent_dim:]
                z_plot[:,k, :self.latent_dim] = z_num[idx, :self.latent_dim]
            ##############################
###Now we need to compute w_den
### General formula : w_den[k,i] = a[i] q_phi(T^{k-i}(x))J_T^{k-i}(x)
        idx_array = np.arange(self.K)
        for k in range(self.K):
            w_den[k,:k] = w[k-idx_array[:k]]
            w_den[k,k:] = w_minus[idx_array[k:]-k]
            #for i in range(self.K):###Writing it naive so that we can be sure
            #    w_den[k,i] = w[k-i] if (k-i>0) else w_minus[i-k]
            w_den[k] += torch.log(self.a).view((self.K,)+ (1,)*(len(z.shape)-1))
        if plotting:
            weights_plot = w[:, idx]+ torch.log(self.a).view(self.K, 1) - torch.logsumexp(w_den[...,idx], dim=1)
            plot_traj_coloured(z_plot,weights_plot, loglikelihood, x, self.transformation)
            
        return w+torch.log(self.a).view((self.K,)+ (1,)*(len(z.shape)-1)),joint_ll, q_phi_k, w_den, z_traj ###I need to return w_den to have nice expression for the gradient

    
    
#    if not plotting:
#
#                w_den[k] += torch.log(self.a).view(self.K, 1)
#        if plotting:
#            a1= torch.tensor(np.exp(-1.*np.arange(self.K)*self.dim*self.gamma*self.h))
#            weights_plot = w[:, idx]+ torch.log(a1) - \
#                                            torch.logsumexp(w_den[...,idx] + torch.log(a1).view(self.K, 1), dim=1)
#            plot_traj_coloured(z_plot,weights_plot, loglikelihood, x)
#            a2= torch.ones(self.K)
#            weights_plot = w[:, idx]+ torch.log(a2) - \
#                                            torch.logsumexp(w_den[...,idx] + torch.log(a2).view(self.K, 1), dim=1)
#            plot_traj_coloured(z_plot,weights_plot, loglikelihood, x)

    def log_estimate(self, loglikelihood, x):
        if callable(self.importance_distr)&call_distr:
            q = self.importance_distr(x).sample((self.n,))
        else:
            q = self.importance_distr.sample((self.n,))
        p = (.5*self.logvar_p).exp()*torch.randn_like(q)
        z = torch.cat([q,p], dim = -1)
        log_weights_num, joint_ll, q_phi_k, log_rho, z_traj = self.weights(x, z, loglikelihood)

        log_weights = log_weights_num - torch.logsumexp(log_rho, dim = 1) ###Contains weights w_k K*batch_size

        log_varpi = log_weights + joint_ll - q_phi_k ###Contains log unnormalised varpi K*batch_size
        if self.gibbs:
            return log_varpi, z_traj
        log_varpi[torch.isnan(log_varpi)] = torch.zeros(1).log()
        log_est = torch.logsumexp(log_varpi, dim=0) ###logsum exp varpi, estimator of the elbo

        f = torch.logsumexp(log_est, dim=0) - np.log(self.n)
        return f
    
    
    def M(self, q_0_curr, loglikelihood, x, stepsize = 1e-1):
        ###Do nothing -- will just change momentum
        #return q_0_curr
        ####RWM
        u = torch.randn_like(q_0_curr)
        if callable(self.importance_distr)&call_distr:
            current_target = lambda q, x: self.importance_distr(x).log_prob(q).sum(-1) 
        else:
            current_target = lambda q, x=None: self.importance_distr.log_prob(q)
        
        q_0_prop = q_0_curr + stepsize*u
        log_t = current_target(q_0_prop, x) - current_target(q_0_curr, x)
        log_rand = torch.log(torch.rand_like(log_t))

        accept = (log_rand <= log_t) ###Should be min(0, log_t) but this expression would then still be correct

        q_0_prop[~accept] = q_0_curr[~accept] ##Set the non accepted to old state
        return q_0_prop
        
        
    
    def log_estimate_gibbs_correlated(self, z_curr, loglikelihood, x):
        ###Here, we want to use a \rho reversible kernel for initializing points (reversible wrt importance distr)
        #pdb.set_trace()
        z_0_curr = z_curr[0]
        q_0_curr = z_0_curr[...,:self.latent_dim]
        l = self.n
        if callable(self.importance_distr)&call_distr:
            q = self.importance_distr(x).sample((self.n,))
        else:
            q = self.importance_distr.sample((self.n,) + q_0_curr.shape[:-1])
        q[0] = self.M(q_0_curr, loglikelihood, x)
        for i in range(1,l):
            q[i] =self.M(q[i-1], loglikelihood, x)
        
        
        p = torch.randn_like(q)
        z = torch.cat([q,p], dim = -1)
        log_weights_num, joint_ll, q_phi_k, log_rho, z_traj = self.weights(x, z, loglikelihood)

        log_weights = log_weights_num - torch.logsumexp(log_rho, dim = 1) ###Contains weights w_k K*batch_size

        log_varpi = log_weights + joint_ll - q_phi_k ###Contains log unnormalised varpi K*batch_size
        if self.gibbs:
            return log_varpi, z_traj
        
        log_est = torch.logsumexp(log_varpi, dim=0) ###logsum exp varpi, estimator of the elbo

        f = torch.logsumexp(log_est, dim=0) - np.log(self.n)
        return f


        ###TODO add replicas of z to do directly tensorized estimator with many samples





class AIS_Estimator(Estimator):
    ##HMC transitions for AIS estimator
    def __init__(self, dim, num_samples, prior, importance_distr, transformation_params, K, logvar_p = torch.tensor(0., dtype = torch.float32)):
        super().__init__( dim, num_samples, prior, importance_distr)

        t_params = transformation_params.copy()
        transformation_name = t_params.pop('name')
        if transformation_name == 'LeapFrog':
            self.transformation = LeapFrog( **t_params)
            self.hamiltonian = True
        else:
            raise ValueError

        self.K = K
        self.latent_dim=dim
        beta_unnorm = torch.sigmoid(4*(torch.tensor(2*np.linspace(0., 1., self.K + 2), dtype=torch.float32)-1.))
        self.beta = (beta_unnorm - beta_unnorm[0])/(beta_unnorm[-1] - beta_unnorm[0])
        self.logvar_p = logvar_p
        self.v = 0.
###AIS estimator

    def one_step(self, target, z, x):
        #pdb.set_trace()
        ###Refresh momentum
        update = ((.5*self.logvar_p).exp())*torch.randn_like(z[..., self.dim:])
        z[..., self.dim:] = self.v * z[..., self.dim:] + (1- self.v**2)**(.5) * update
        ###Compute proposal
        z_new, _ = self.transformation(z, target=target, x=x, logvar_p = self.logvar_p)
        ###Accept-reject step
        log_t = target(z_new[..., :self.dim], x=x) + logprob_normal(z_new[...,self.dim:], logvar = self.logvar_p).sum(-1) - (target(z[..., :self.dim], x=x) + logprob_normal(z[...,self.dim:], logvar =self.logvar_p).sum(-1))

        log_rand = torch.log(torch.rand_like(log_t))

        accept = (log_rand <= log_t) ###Should be min(0, log_t) but this expression would then still be correct

        z_new[~accept][..., :self.dim] = z[~accept][..., :self.dim]
        z_new[~accept][..., self.dim:] = -1.*z[~accept][..., self.dim:]
        ##Set the non accepted to old state
        #print((1.0*accept).mean())
        return z_new, accept

    def incremental_weight(self, loglikelihood, q, k, x):
        if callable(self.importance_distr):
            return (self.beta[k+1] - self.beta[k])*(loglikelihood(q, x=x) + self.prior.log_prob(q).sum(-1) - self.importance_distr(x).log_prob(q).sum(-1))
        else:
            return (self.beta[k+1] - self.beta[k])*(loglikelihood(q, x=x) + self.prior.log_prob(q) - self.importance_distr.log_prob(q))



    def log_estimate(self, loglikelihood, x):
        #pdb.set_trace()
        if callable(self.importance_distr):
            q = self.importance_distr(x).sample((self.n,))
        else:
            q = self.importance_distr.sample((self.n,))
        p = (.5*self.logvar_p).exp()*torch.randn_like(q)
        z = torch.cat([q,p], dim = -1)
        sum_log_weights = self.incremental_weight(loglikelihood=loglikelihood, q=q, k=0, x=x)
        if callable(self.importance_distr):
            current_target = lambda k: lambda z, x:(self.beta[k])*(loglikelihood(z, x)+self.prior.log_prob(z).sum(-1)) + (1-self.beta[k])* self.importance_distr(x).log_prob(z).sum(-1)
        else:
            current_target = lambda k: lambda z, x=None:(self.beta[k])*(loglikelihood(z, x)+self.prior.log_prob(z)) + (1-self.beta[k])* self.importance_distr.log_prob(z)
        for k in range(1, len(self.beta)-1):
            z, accept = self.one_step(target = current_target(k), z=z, x=x)
            sum_log_weights += self.incremental_weight(loglikelihood=loglikelihood, q=z[..., :self.dim], k=k, x=x)
        f = torch.logsumexp(sum_log_weights, dim=0) - np.log(self.n)
        return f
