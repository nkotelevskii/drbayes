"""
    inferences class w/in the subspace
    currently only fitting the Gaussian associated is implemented
"""

import abc
import torch
import numpy as np

from torch.distributions import LowRankMultivariateNormal
from .elliptical_slice import elliptical_slice
from subspace_inference.utils import unflatten_like, flatten, train_epoch
from .proj_model import ProjectedModel
from .vi_model import VIModel, ELBO

class MCMC_Proj(torch.nn.Module, metaclass=abc.ABCMeta):

    subclasses = {}

    @classmethod
    def register_subclass(cls, mcmc_type):
        def decorator(subclass):
            cls.subclasses[mcmc_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, mcmc_type, **kwargs):
        if inference_type not in cls.subclasses:
            raise ValueError('Bad MCMC type {}'.format(mcmc_type))
        return cls.subclasses[mcmc_type](**kwargs)

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        super(MCMC_Proj, self).__init__()

    #@abc.abstractmethod
    #def fit(self, mean, variance, cov_factor, *args, **kwargs):
    #    pass

    @abc.abstractmethod
    
    def sample(self, num_samples, *args, **kwargs):
        pass





@MCMC_Proj.register_subclass('projected_sgld')
class ProjSGLD(MCMC_Proj):
    def __init__(self, model, loader, criterion, mean, subspace, epochs = 10, cyclic= False, T = 1, lr = 1e-2,  **kwargs):
        super(ProjSGLD, self).__init__()
        self.kwargs = kwargs
        self.optimizer = None

        self.mean, self.var, self.subspace = None, None, None
        self.optimizer = None
        self.proj_params = None
        self.loader, self.criterion = loader, criterion
        self. lr = lr
        self.cyclic = cyclic
        self.T = T
        self.model = model
    
    def sample(self, num_samples, num_epochs, *args, **kwargs):
        
        
        
        if use_cuda and torch.cuda.is_available():
            self.mean = mean.cuda()
            self.subspace = subspace.cuda()
        else:
            self.mean = mean
            self.subspace = subspace
        
        if self.proj_params is None:
            proj_params = torch.zeros(self.subspace.rank, 1, dtype = self.subspace.mean.dtype, device = self.subspace.mean.device, requires_grad = True)
            #print(proj_params.device)
            self.proj_model = ProjectedModel(model=self.model, mean=self.mean.unsqueeze(1),  projection=self.subspace, proj_params=proj_params)
            
            # define optimizer
            self.optimizer = torch.optim.SGD([proj_params], lr )
            
        else:
            proj_params = self.proj_params.clone()
        if self.cyclic:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.T)
        samples = torch.zeros((num_samples,) + proj_params.shape)
        T_s = max(1,n_epochs//n_samples)
        # now train projected parameters
        loss_vec = []
        for i in range(self.epochs):
            loss = train_epoch(loader=self.loader, optimizer=self.optimizer, model=self.proj_model, criterion=self.criterion, scheduler = self.scheduler, **kwargs)
            loss_vec.append(loss)
            if ((i%T_s==0)&(i//T_s<num_samples)):
                samples[i//T_s] = proj_params
                
        self.proj_params = proj_params
        
        return samples
        
    
    
@MCMC_Proj.register_subclass('vi')
class VI(MCMC_Proj):

    def __init__(self, base, base_args, base_kwargs, rank, init_inv_softplus_simga=-6.0, prior_log_sigma=0.0):
        super(VI, self).__init__()

        self.vi_model = VIModel(
            base=base,
            base_args=base_args,
            base_kwargs=base_kwargs,
            rank=rank,
            init_inv_softplus_simga=init_inv_softplus_simga,
            prior_log_sigma=prior_log_sigma
        )


    def fit(self, mean, variance, cov_factor, loader, criterion, epochs=100):
        print('Fitting VI')
        self.vi_model.set_subspace(mean, cov_factor)

        elbo = ELBO(criterion, len(loader.dataset))

        optimizer = torch.optim.Adam([param for param in self.vi_model.parameters() if param.requires_grad])

        for _ in range(epochs):
            train_res = train_epoch(loader, self.vi_model, elbo, optimizer)
            print(train_res)



    def sample(self, num_samples):
        samples = torch.zeros((num_samples,) + self.vi_model.sample(num_samples).shape)
        for i in range(num_samples):
            samples[i] = self.vi_model.sample(num_samples)

        return samples