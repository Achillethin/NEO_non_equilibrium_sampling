import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flows import RNVP
from pyro.distributions.transforms import AffineAutoregressive, BlockAutoregressive, AffineCoupling
from pyro.nn import AutoRegressiveNN, DenseNN


class BaseTransformation(nn.Module):
    def __init__(self):
        super().__init__()

        self.grad_U = None


    def get_grad(self, q, target, x=None, clip=1e3):
        if self.grad_U is not None:
            return self.grad_U(q)
        else:
            if self.estimation:
                q = q.detach().requires_grad_()
            if x is not None:
                s = target(z=q, x= x)
            else:
                s = target(q)
            grad = torch.autograd.grad(s.sum(), q)[0]
            if clip:
                total_norm = torch.norm(grad.detach(), 2.0)
                clip_coef = clip / (total_norm + 1e-6)
                if clip_coef < 1.:
                    grad.mul_(clip_coef)
                grad = torch.clamp(grad, -20, 20)
            return -grad


    def forward(self, z, x= None):
        return z, torch.zeros_like(z[:,0])


    def backward(self, z, x= None):
        return z, torch.zeros_like(z[:,0])



class LeapFrog(BaseTransformation):
    ###We assume we perform LF per LF here... Maybe add more LF steps ?
    ###If 1 LF, step, we rather use Drift-Kick-Drift/// Not the case here
    def __init__(self, h=0.1, N_LF=1, train_ = False,  grad_U=None, estimation=False, logvar_p = torch.tensor(0., dtype = torch.float32)):
        super().__init__()
        self.n_leapfrogs = N_LF
        self.dt = nn.Parameter(data= torch.tensor(h), requires_grad=train_)

        self.grad_U = grad_U
        self.estimation = estimation

    def get_var_p(self):
        return (self.logvar_p).exp()

    def forward(self, z, target, x= None, logvar_p = torch.tensor(0., dtype = torch.float32)):

        dim = z.shape[-1] // 2
        z_current, p_current = z[..., :dim], z[..., dim:]
        step_size = self.dt
        z_ = z_current
        p_ = p_current
        p_ =  p_ - (step_size/2.)  * self.get_grad(q=z_current, target=target, x=x)

        for l in range(self.n_leapfrogs):
            z_ = z_ + step_size * (-logvar_p).exp() * p_
            if (l != (self.n_leapfrogs-1)):
                p_ =  p_ - step_size  * self.get_grad(q=z_current, target=target, x=x)

        p_ =  p_ - (step_size/2.)  * self.get_grad(q=z_current, target=target, x=x)
        z_f = torch.cat([z_, p_], dim=-1)
        return z_f, torch.zeros_like(z[...,0])


    def inverse(self, z, target,  x= None, logvar_p = torch.tensor(0., dtype = torch.float32)):
        dim = z.shape[-1] // 2
        z_current, p_current = z[..., :dim], z[..., dim:]
        step_size = self.dt
        z_ = z_current
        p_ = -p_current
        p_ =  p_ - (step_size/2.)  * self.get_grad(q=z_current, target=target, x=x)

        for l in range(self.n_leapfrogs):
            z_ = z_ + step_size * (-logvar_p).exp() * p_
            if (l != (self.n_leapfrogs-1)):
                p_ =  p_ - step_size  * self.get_grad(q=z_current, target=target, x=x)

        p_ =  p_ - (step_size/2.)  * self.get_grad(q=z_current, target=target, x=x)
        p_ *= -1
        z_f = torch.cat([z_, p_], dim=-1)
        return z_f, torch.zeros_like(z[...,0])



class DampedHamiltonian(BaseTransformation):
    def __init__(self, gamma= 0., h=0.1, train_=False, grad_U=None, estimation = False, clip = True):
        super().__init__()

        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=train_)
        self.dt = nn.Parameter(torch.tensor(h), requires_grad=train_)

        self.grad_U = grad_U
        self.estimation = estimation
        self.clip = 5.


    def forward(self, z, target, x=None, logvar_p = torch.tensor(0., dtype = torch.float32)):
        dim = z.shape[-1]//2
        q,p = z[..., :dim], z[..., dim:]
        pt = torch.exp(self.gamma*self.dt) * p
        p1 = pt - self.dt * self.get_grad(q, target=target,  x=x, clip=self.clip)
        q1 = q + self.dt * (-logvar_p).exp() * p1
        z = torch.cat([q1, p1], dim=-1)
        log_jac_single = dim * self.gamma * self.dt
        return z, log_jac_single * torch.ones_like(z[...,0])


    def inverse(self, z, target,  x= None, logvar_p = torch.tensor(0., dtype = torch.float32)):
        dim = z.shape[-1]//2
        q,p = z[..., :dim], z[..., dim:]
        q1 = q - self.dt * (-logvar_p).exp() * p
        pt =   torch.exp(-self.gamma*self.dt)*p
        p1 = pt + self.dt *  torch.exp(-self.gamma*self.dt) * self.get_grad(q1, target= target,  x=x, clip=self.clip)
        x = torch.cat([q1, p1], dim=-1)
        log_jac_single =  - dim * self.gamma * self.dt
        return x, log_jac_single * torch.ones_like(z[...,0])


class DampedHamiltonian_lf(BaseTransformation):
    def __init__(self, gamma= 0., h=0.1, train_=False, grad_U=None, estimation = False, clip = True):
        super().__init__()

        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=train_)
        self.dt = nn.Parameter(torch.tensor(h), requires_grad=train_)

        self.grad_U = grad_U
        self.estimation = estimation
        self.clip = 10.



    def forward(self, z, target, x=None, logvar_p = torch.tensor(0., dtype = torch.float32)):
        dim = z.shape[-1]//2
        q,p = z[..., :dim], z[..., dim:]

        qt = q + self.dt/2. * torch.exp(self.gamma*self.dt/2.) * (-logvar_p).exp() * p

        pt = torch.exp(self.gamma*self.dt/2.) * p
        p1 = pt - self.dt * self.get_grad(qt, target=target,  x=x, clip=self.clip)

        q1 = qt + self.dt/2. * (-logvar_p).exp() * p1

        p1 *=  torch.exp(self.gamma*self.dt/2.)


        z = torch.cat([q1, p1], dim=-1)
        log_jac_single = dim * self.gamma * self.dt
        return z, log_jac_single * torch.ones_like(z[...,0])


    def inverse(self, z, target,  x= None, logvar_p = torch.tensor(0., dtype = torch.float32)):
        dim = z.shape[-1]//2
        q,p = z[..., :dim], z[..., dim:]

        pt =  torch.exp(-self.gamma*self.dt/2.) * p

        qt = q - self.dt/2. * (-logvar_p).exp() * pt

        p1 = (pt + self.dt * self.get_grad(qt, target= target,  x=x, clip=self.clip)) * torch.exp(-self.gamma*self.dt/2.)

        q1 = qt - self.dt/2. * torch.exp(-self.gamma*self.dt/2.) * (-logvar_p).exp() * p1

        x = torch.cat([q1, p1], dim=-1)
        log_jac_single =  - dim * self.gamma * self.dt

        return x, log_jac_single * torch.ones_like(z[...,0])


class Transfo_RNVP(BaseTransformation):
    def __init__(self, dim, flows=None, num_flows_per_transfo = None, K= None, train_ = True):
        super().__init__()

        if flows is not None:
            if len(flows)!= K:
                print('Mistake on number of flows')
            else:
                self.flows = flows

        else:
            if num_flows_per_transfo <2:
                print('it\'s not advised to use less than 2 flows per step !!')
            self.flows = [RNVP(num_flows_per_transfo, dim) for _ in range(K)]


    def forward(self, z, x=None, k=None):
        z, log_jac = self.flows[k](z)
        return z, log_jac

    def inverse(self, z, x=None, k=None):
        z, log_jac = self.flows[k].inverse(z)
        return z, log_jac
