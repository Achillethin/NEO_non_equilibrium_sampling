import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.utils import logprob_normal, view_flat, bce, mse
from models.transformations import LeapFrog, DampedHamiltonian, DampedHamiltonian_lf, BaseTransformation, Transfo_RNVP
from models.flows import RNVP
from models.evidence import Estimator, AIS_Estimator, NonEq_Estimator_w_a
from models.nets import SimpleNN, ConvNN

import pdb


class VanillaVAE(nn.Module):
    '''
    Vanilla VAE. Kept as simple as possible
    '''
    def __init__(self, latent_dim, activation = "relu", hidden_dim = 256, output_dim = 784, archi="basic", data_type="binary"):
        super().__init__()
        if archi == "basic":
            self.net = SimpleNN(latent_dim, activation, hidden_dim, output_dim, large=False)
        elif archi == "large":
            self.net = SimpleNN(latent_dim, activation, hidden_dim, output_dim, large=True)
        elif archi == "convCifar":
            self.net = ConvNN(latent_dim, activation, hidden_dim)
        elif archi == "convMnist":
            self.net = ConvNN(latent_dim, activation, hidden_dim, image_size = (28,28), channel_num=1, num_layers=2)
        else:
            raise ValueError("Architecture unknown: " + str(archi))
        self.latent_dim = latent_dim
        self.description = "vae"
        self.data_type = data_type
        if data_type == "binary":
            self.reconstruction_loss = bce
        elif data_type == "continuous":
            self.reconstruction_loss = mse


    def encode(self, x):
        return self.net.encode(x)


    def sample(self, mu, logvar):
        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z):
        return self.net.decode(z)


    def log_prob(self, z, x=None):
        log_likelihood = - self.reconstruction_loss(self.decode(z), x)
        log_prior = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                               scale=torch.tensor(1., device=z.device, dtype=torch.float32)).log_prob(z).sum(-1)
        return log_likelihood + log_prior


    def forward(self, z):
        if self.data_type== "binary":
            return torch.sigmoid(self.decode(z))
        else:
            return self.decode(z)


    def loss_function(self, x_rec, x, mu, logvar):
        BCE = self.reconstruction_loss(x_rec, x).mean()
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).sum(-1))
        loss = BCE + KLD
        return loss, BCE


    def step(self, x):
        # Computes the loss from a batch of data
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)
        loss, BCE = self.loss_function(view_flat(x_rec), view_flat(x), mu, logvar)
        return loss, x_rec, z, BCE


    def get_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


    def importance_distr(self, x):
        mu, logvar = self.encode(x)
        return torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar))


    def mean_estimate(self, num_samples, dataset):
        prior = torch.distributions.Normal(
                                        loc=torch.tensor([0.]*self.latent_dim, device=dataset.device, dtype=torch.float32),
                                        scale=torch.tensor([1.]*self.latent_dim, device=dataset.device, dtype=torch.float32))
        estim = Estimator(self.latent_dim, num_samples, prior, None)
        loglikelihood = lambda samples, x: - self.reconstruction_loss(view_flat(self.decode(samples)), view_flat(x))
        return estim.log_estimate(loglikelihood, dataset)


    def importance_estimate(self, num_samples, dataset):
        prior = torch.distributions.Normal(
                                        loc=torch.tensor([0.]*self.latent_dim, device=dataset.device, dtype=torch.float32),
                                        scale=torch.tensor([1.]*self.latent_dim, device=dataset.device, dtype=torch.float32))
        estim = Estimator(self.latent_dim, num_samples, prior, self.importance_distr)
        loglikelihood = lambda samples, x: - self.reconstruction_loss(view_flat(self.decode(samples)), view_flat(x))
        return estim.log_estimate(loglikelihood, dataset)


    def ais_estimate(self, num_samples, args, K, dataset):
        prior = torch.distributions.Normal(
                                        loc=torch.tensor([0.]*self.latent_dim, device=dataset.device, dtype=torch.float32),
                                        scale=torch.tensor([1.]*self.latent_dim, device=dataset.device, dtype=torch.float32))
        estim = AIS_Estimator(self.latent_dim, num_samples, prior, self.importance_distr, args, K)
        loglikelihood = lambda samples, x: - self.reconstruction_loss(view_flat(self.decode(samples)), view_flat(x))
        return estim.log_estimate(loglikelihood, dataset)

    def neq_estimate(self, num_samples, args, a, dataset, logvar_p, logvar_p_transfo):
        prior = torch.distributions.Normal(
                                        loc=torch.tensor([0.]*self.latent_dim, device=dataset.device, dtype=torch.float32),
                                        scale=torch.tensor([1.]*self.latent_dim, device=dataset.device, dtype=torch.float32))
        args.estimation = True  ###to make sure, to comment if it is ok
        estim =NonEq_Estimator_w_a(self.latent_dim, num_samples, prior, self.importance_distr, args, a, logvar_p=logvar_p, logvar_p_transfo=logvar_p_transfo)
        loglikelihood = lambda z, x: - self.reconstruction_loss(view_flat(self.decode(z)), view_flat(x))
        return estim.log_estimate(loglikelihood, dataset)


class IWAE(VanillaVAE):
    '''
    Importance Weighted Auto Encoder
    '''
    def __init__(self, num_samples, latent_dim, activation="relu", clamp=1e6, hidden_dim = 256, output_dim = 784, archi="basic", data_type="binary"):
        super().__init__(latent_dim, activation, hidden_dim, output_dim, archi, data_type)
        self.num_samples = num_samples
        self.clamp_kl = clamp
        self.description = f"iwae n={num_samples}"

    def sample(self, mu, logvar):
        ''' Reparametrization trick
        Will output a tensor of shape ``(num_sample, batch_size, [input_dim])`
        '''
        std = torch.exp(0.5 * logvar)
        # Repeat the std along the first axis to sample multiple times
        dims = (self.num_samples,) + (std.shape)
        eps = torch.randn(*dims, device=mu.device)
        return mu + eps * std


    def loss_function(self, x_rec, x, mu, logvar, z):
        log_Q = logprob_normal(z,mu,logvar).view((self.num_samples, -1, self.latent_dim)).sum(-1)
        log_Pr = (-0.5 * z ** 2).view((self.num_samples, -1, self.latent_dim)).sum(-1)
        KL_eq = torch.clamp(log_Q - log_Pr, -self.clamp_kl, self.clamp_kl)
        BCE = self.reconstruction_loss(x_rec, x)

        log_weight = - BCE - KL_eq
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()
        loss = torch.mean(torch.sum(weight * (BCE + KL_eq), 0))

        return loss, torch.sum(BCE * weight, dim=0).mean()


    def step(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)
        loss, BCE = self.loss_function(view_flat(x_rec), view_flat(x), mu, logvar, z)
        return loss, x_rec, z, BCE



class FlowVAE(VanillaVAE):
    '''VAE with flow
    Simple VAE with `num_flows` RNVP transforming the latent `z`
    '''
    def __init__(self, num_flows, latent_dim, activation="relu", hidden_dim = 256, output_dim = 784, archi="basic", data_type="binary"):
        super().__init__(latent_dim, activation, hidden_dim, output_dim, archi, data_type)
        self.Flow = RNVP(num_flows, latent_dim)
        self.description = "flowvae"

    def loss_function(self, x_rec, x, mu, logvar, z, z_transformed, log_jac):
        BCE = self.reconstruction_loss(x_rec, x).mean()
        log_Q = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(z)
        log_Pr = (-0.5 * z_transformed ** 2)
        KLD = torch.mean((log_Q - log_Pr).sum(-1) - log_jac)
        loss = BCE + KLD
        return loss, BCE


    def step(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        z_transformed, log_jac = self.Flow(z)
        x_rec = self.decode(z_transformed)
        loss, BCE = self.loss_function(view_flat(x_rec), view_flat(x), mu, logvar, z, z_transformed, log_jac)
        return loss, x_rec, z_transformed, BCE



class NeqVAE(VanillaVAE):
    '''Non-equilibrium NeqVAE
    Expects a transformation `transformation_params` with a name and its parameters
    Expects a tensor of `a` weights, which are non-zero, its length controls the
        number of times we apply the flow
    '''
    def __init__(self, latent_dim, transformation_params, a, activation="relu", hidden_dim = 256, output_dim = 784, archi="basic", data_type="binary", logvar_p=torch.tensor([0.]), logvar_p_tranfo_train=False):
        super().__init__(latent_dim, activation, hidden_dim, output_dim, archi, data_type)

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
        elif transformation_name == 'DampedHamiltonian_lf':
            self.transformation = DampedHamiltonian_lf(**t_params)
            self.flows_different = False
            self.hamiltonian = True
        elif transformation_name == 'Identity':
            self.transformation = BaseTransformation(dim=latent_dim)
            self.flows_different = False
            self.hamiltonian = False
        elif transformation_name == 'Real-NVP':
            self.transformation = Transfo_RNVP(dim = latent_dim, **t_params)
            self.flows_different = True
            self.hamiltonian = False
        else:
            raise ValueError

        self.a = a
        self.K = a.shape[0]
        self.description = f"neqvae a={a.cpu().numpy()} {transformation_name} h:{self.transformation.dt:.2f} gamma:{self.transformation.gamma:.2f}"
        if logvar_p == "adaptative":
            self.logvar_p = torch.nn.Parameter(torch.zeros(latent_dim), requires_grad=False)
            self.adaptative_logvar_p = True
            self.logvar_p_moment = 0.9
            self.description = self.description + " adaptative"
        elif logvar_p == "trainable":
            self.logvar_p = torch.nn.Parameter(torch.zeros(latent_dim))
            self.adaptative_logvar_p = False
            self.description = self.description + " trainable"
        else:
            self.logvar_p = torch.nn.Parameter(logvar_p)
            self.adaptative_logvar_p = False
        self.logvar_p_transfo = torch.zeros(latent_dim, requires_grad=logvar_p_tranfo_train)


    def to(self, *args, **kwargs):
        """
        overloads to method to make sure the manually registered buffers are sent to device
        """
        self = super().to(*args, **kwargs)
        self.logvar_p = self.logvar_p.to(*args, **kwargs)
        self.logvar_p_transfo = self.logvar_p_transfo.to(*args, **kwargs)
        return self


    def compute_and_push(self, x, mu, logvar, z, log_jac, backward = False, k=None):
        ###First we compute the weights and the value of the current pushforward
        if self.hamiltonian:
            q_phi_k = logprob_normal(z[:,:self.latent_dim], mu, logvar).sum(-1)
            w =  q_phi_k + \
                        logprob_normal(z[:,self.latent_dim:], logvar=self.logvar_p).sum(-1) + \
                        log_jac
        else:
            q_phi_k = logprob_normal(z, mu, logvar).sum(-1)
            w =  q_phi_k + \
                        log_jac


        ####Then we push, backward or forward
        if backward:
            joint_ll = None
            if self.flows_different:
                z, log_jac_cur = self.transformation.inverse(z=z, x=x, k=k, logvar_p = self.logvar_p_transfo)
            else:
                z, log_jac_cur = self.transformation.inverse(z=z, target = self.log_prob, x=x, logvar_p = self.logvar_p_transfo)
        else:
            if self.hamiltonian:
                joint_ll = self.log_prob(z[:,:self.latent_dim], x)
            else:
                joint_ll = self.log_prob(z, x)
            if self.flows_different:
                z, log_jac_cur = self.transformation(z=z, x=x, k=k, logvar_p = self.logvar_p_transfo)
            else:
                z, log_jac_cur = self.transformation(z=z, target = self.log_prob, x=x, logvar_p = self.logvar_p_transfo)
        log_jac+=log_jac_cur
        ### Return updated point and jacobian plus weights and relevant quantities to compute
        return w, joint_ll, q_phi_k, z, log_jac


    def weights(self, x, mu, logvar, z):
        w = torch.zeros((self.K, z.shape[0])) ###tensor containing numerator of the weights
        joint_ll = torch.zeros((self.K, z.shape[0]))
        q_phi_k = torch.zeros((self.K, z.shape[0]))
        z_num = z ##point going through the flow at numerator at each step k
        z_init = z.clone().detach()
        log_jac_num = torch.zeros_like(z[:,0])
        w_den = torch.zeros((self.K, self.K, z.shape[0]))
        if self.flows_different:  #### This is left to future work.
            for k in range(self.K):
                ###Compute the weights
                z_den = z_num
                w[k], joint_ll[k], q_phi_k[k], z_num, _ = self.compute_and_push(x,mu, logvar, z_num, log_jac_num, k=k)
                log_jac_den = torch.zeros_like(log_jac_num)
                ###Compute the weights on denumenator
                for j in range(self.K):
                    w_den[k,j], _, _, z_den, log_jac_den = self.compute_and_push(x,mu,
                                                                              logvar, z_den,
                                                                              log_jac_den, backward=True, k=j)
                
                w_den[k,:,:] += torch.log(self.a).view(self.K, 1) ###Good summation
        else:
            z_den= z
            log_jac_den = torch.zeros_like(log_jac_num)
            w_minus = torch.zeros_like(w) ###Contains a_-k q_phi(T^-k(x)) J_T^-k(x)
            for k in range(self.K):
                w[k], joint_ll[k], q_phi_k[k], z_num, log_jac_num =  self.compute_and_push(x,mu,
                                                                                        logvar, z_num,
                                                                                        log_jac_num)
                w_minus[k], _, _, z_den, log_jac_den = self.compute_and_push(x,mu,
                                                                            logvar, z_den,
                                                                            log_jac_den, backward=True)
            ###Now we need to compute w_den
            ### General formula : w_den[k,i] = a[i] q_phi(T^{k-i}(x))J_T^{k-i}(x)
            idx_array = np.arange(self.K)
            for k in range(self.K):
                w_den[k,:k] = w[k-idx_array[:k]]
                w_den[k,k:] = w_minus[idx_array[k:]-k]
                #for i in range(self.K):###Writing it naive so that we can be sure
                #    w_den[k,i] = w[k-i] if (k-i>0) else w_minus[i-k]
                w_den[k] += torch.log(self.a).view(self.K, 1)
        distances = torch.norm(z_num.detach()-z_init, dim=-1)
        return w+torch.log(self.a).view(self.K, 1), joint_ll, q_phi_k, w_den, distances ###I need to return w_den to have nice expression for the gradient


    def loss_function(self, x, mu, logvar, z):
        ''' Returns the weighted average of the function of interest
        '''
        log_weights_num, joint_ll, q_phi_k, log_rho, distances = self.weights(x, mu, logvar, z)

        log_weights = log_weights_num - torch.logsumexp(log_rho, dim = 1) ###Contains weights w_k K*batch_size

        log_varpi = log_weights + joint_ll - q_phi_k ###Contains log unnormalised varpi K*batch_size

        elbo = torch.logsumexp(log_varpi, dim=0) ###logsum exp varpi, estimator of the elbo

        #log_norm_varpi = log_varpi - elbo ###Contains log normalized  varpi K*batch_size

        #log_norm_rho = log_rho - torch.logsumexp(log_rho, dim=1) ###Contains log normalized rho, K*K*batch_size

        ##To use for expression of the gradient ?
        #norm_rho = torch.exp(log_norm_rho).detach()
        #norm_varpi = torch.exp(log_norm_varpi).detach()

        #loss_1 = norm_varpi * joint_ll ###First term of the loss, dim K*batch_size
        #loss_2 = norm_rho* log_rho    ###Second term of the loss, dim K*K*batch_size
        #loss = (loss_1 - torch.sum(loss_2, dim=1)).sum(0) ###Summing over index k

        mean_elbo = torch.mean(elbo)
        ###Estimator of the elbo is logsumexp log varpi
        return -mean_elbo, mean_elbo, distances#-loss.mean(), torch.mean(elbo)


    def step(self, batch):
        x = batch.view(batch.shape[0], -1)
        mu, logvar = self.encode(x)
        q = self.sample(mu, logvar)
        x_rec = self.decode(q)
        if self.adaptative_logvar_p:
            self.logvar_p.data = self.logvar_p.data * self.logvar_p_moment + \
                        torch.mean(logvar.detach(), axis=0) * (1. - self.logvar_p_moment)
        if self.hamiltonian:
            p = torch.exp(0.5 * self.logvar_p)*torch.randn_like(q)
            z = torch.cat([q,p], dim = 1)
            loss, elbo, displacement = self.loss_function(x, mu, logvar, z)
        else:
            loss, elbo, displacement = self.loss_function(x, mu, logvar, q)
        return loss, x_rec, displacement, elbo


    def sample_trajectory(self, q_init, steps, gamma=None, scale = 1.):
        """
        From q_init points with shape: (batch, latent_dim), generate a trajectory
        in latent space and the associated decoded trajectory
        """
        if gamma!=None:
            old_gamma = self.transformation.gamma.data
            self.transformation.gamma.data=gamma
        p = torch.exp(0.5 * self.logvar_p)*torch.randn_like(q_init)* scale
        z_t = torch.cat([q_init,p], dim = 1)
        z_t.requires_grad_()
        xts = []
        zts = []
        for t in range(steps):
            zts.append(z_t.detach())
            x_t = self.decode(z_t[:,:self.latent_dim])
            z_t = self.transformation(z_t, target = self.log_prob, x=x_t, logvar_p = self.logvar_p)[0]
            xts.append(x_t.detach())
        xts = torch.stack(xts, dim=1)
        zts = torch.stack(zts, dim=1)
        if gamma!=None:
            self.transformation.gamma.data=old_gamma
        return xts, zts
