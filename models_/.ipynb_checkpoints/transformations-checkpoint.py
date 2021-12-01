import torch
from models_.dot_dict import DotDict

class DampedHamiltonian:

    def __init__(self, args):
        
        self.dim       = args.dim
        self.gamma     = args.gamma
        self.dt        = args.dt
        self.K         = args.K
        self.var_Tp    = args.var_Tp
        self.clamp     = args.clamp


    def forward(self, q_t, p_t, k, target):
        """
        returns trajectory and log_probs
        """
        p_t = torch.exp(torch.tensor(-self.gamma*self.dt)) * p_t

        if self.clamp:
            grad = torch.clamp(torch.autograd.grad(target(q_t).sum(), q_t)[0],-self.clamp,self.clamp)
        else:
            grad = torch.autograd.grad(target(q_t).sum(), q_t)[0]

        p_t = p_t + self.dt * grad
        q_t = q_t + (self.dt * p_t / self.var_Tp)

        return q_t, p_t, - self.dt * k * self.dim * self.gamma

    def inverse(self, q_t, p_t, k, target):

        q_t = q_t - (self.dt * p_t / self.var_Tp)

        if self.clamp:
            grad = - torch.clamp(torch.autograd.grad(target(q_t).sum(), q_t)[0],-self.clamp,self.clamp)
        else:
            grad = - torch.autograd.grad(target(q_t).sum(), q_t)[0]

        p_t = torch.exp(self.gamma * self.dt) * (p_t + self.dt * grad)

        return q_t, p_t, self.dt * k * self.dim * self.gamma
        
        
        
        
