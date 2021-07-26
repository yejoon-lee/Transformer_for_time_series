import seaborn as sns
from torch._C import device
sns.set_style('darkgrid')
import numpy as np
import torch
import torch.nn as nn


class InferShortTerm:
    '''Short-term forecasting (i.e. rolling prediction)
    '''
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, 
                input : tuple):
        '''Vanilla forward of self.model
        Args:
            input: ((N, S, 1), (N, T, 1))
        '''
        # source and target
        src, tgt = input
        # forward
        mean, var = self.model(src, tgt)
        # to numpy
        return mean, var # (N, T, 1), (N, T, 1)

    def eval(self, 
             input : tuple, 
             target : torch.Tensor, 
             criterion : nn.modules = nn.GaussianNLLLoss(), 
             metric_funcs : list = [nn.L1Loss()]):
        '''Evaluate loss(GaussianNLL by default) and metrics given by list
        Args:
            input: ((N, S, 1), (N, T, 1))
            target: (N, T, 1)
            criterion: loss function.
            criterion shouldb return value with __call__, receiving mean, target, and var as arguments.
            Default is nn.GaussianNLLLoss(), and changing this arg would be unnecessary for most of the times.

            metrics: list made up of supplementary metric functions. 
            Each function should return value with __call__, receiving mean and target as arguments.
            Defualt is a list of which length is 1, consisting nn.L1Loss().

        Help:
            1. input and target should be torch.Tensor whose device is same to self.model
            2. Use `with torch.no_grad():`   
        '''
        mean, var= self.forward(input) # (N, T, 1), (N, T, 1)
        loss = criterion(mean, target, var)

        metrics = torch.zeros(len(metric_funcs))
        for i, metric_func in enumerate(metric_funcs):
            metrics[i] = metric_func(mean, target)

        return loss, metrics
    
    def plot(self, 
             input : tuple, 
             target : torch.Tensor, 
             axes : np.ndarray, 
             cut_head : int = 0, 
             quantile : float = 0.5,
             num_draw : int = 10000):
        '''Plot prediction with ground truth
        Args:
            input: ((N, S, 1), (N, T, 1)) = (src, tgt); each should be Tensor
            target: (N, T, 1)
            axes: axes.size should be (*, 1) (i.e. single column)
            cut_head: Cut head of the x axis to enhance visibility
            quantile: Quantile to be shown in plot
            num_draw: Numbers drawing samples from probability distribution

        Help:
            1. input and target should be torch.Tensor whose device is same to self.model
            2. Use `with torch.no_grad():`
        '''
        # params
        self.quantile = quantile
        self.num_draw = num_draw

        # forward
        src, tgt = input # (N, S, 1), (N, T, 1)
        mean, var = self.forward(input) # (N, T, 1), (N, T, 1)

        # to cpu and numpy
        src, target, mean, var = src.cpu().numpy(), target.cpu().numpy(), mean.cpu().numpy(), var.cpu().numpy()

        # x-axis
        self.x_axis = np.arange(src.shape[1] + tgt.shape[1]) # (S+T, )
        # ground truth
        truth = np.concatenate((src, target), axis=1) # (N, S+T, 1)
        # cut head to enhance visibility
        self.x_axis = self.x_axis[cut_head:] 

        # draw samples & plot
        for i, ax in enumerate(axes):
            # get single sample from batch
            truth_sampled = truth[i].squeeze(-1) # (S+T, )
            mean_sampled = mean[i].squeeze(-1) # (T, )
            var_sampled = var[i].squeeze(-1) # (T, )

            # cut head to enhance visibility
            truth_sampled = truth_sampled[cut_head:]

            self._plot_single(truth_sampled, mean_sampled, var_sampled, ax)


    def _plot_single(self, truth, mean, var, ax):
        # draw samples from probability distribution
        samples = np.random.normal(mean, var, (self.num_draw, len(mean))).T # (T, num_draw)
        lbounds, medians, ubounds = np.zeros_like(mean), np.zeros_like(mean), np.zeros_like(mean)

        # get stats from each token
        for i, token_sample in enumerate(samples):
            lbound, median, ubound = get_stats(token_sample, self.quantile)
            lbounds[i] = lbound
            medians[i] = median
            ubounds[i] = ubound

        # plot
        ax.plot(self.x_axis, truth, label='ground truth')
        ax.plot(self.x_axis[-len(medians):], medians, label='forecast')
        ax.fill_between(self.x_axis[-len(medians):], lbounds, ubounds, alpha=0.3, color='orange')
        ax.legend()


class InferLongTerm(InferShortTerm):
    '''Long-term forecasting using random sampling.

    Inherit InferShortTerm, only overriding the forward method.
    '''
    def forward(self, 
                input : tuple,
                num_draw : int = 10):
        '''Forward of self.model with recurrence. 
        For tgt, input of t+1_step is the output of t_step. 
        In other words, model is only provided with first token in tgt. 

        Args:
            input: ((N, S, 1), (N, T, 1))
            num_draw: Numbers drawing samples from probability distribution when deciding
            input of t+1_step (scalar) with output of t_step (mean and variance).
        '''
        # unpack input
        src, tgt = input  # (N, S, 1), (N, T, 1)

        # store mean and var for n_draw times
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        drawn_mean = torch.zeros(num_draw, *list(tgt.shape), device=device) # (n_draw, N, T, 1)
        drawn_var = torch.zeros(num_draw, *list(tgt.shape), device=device)  # (n_draw, N, T, 1)

        for i in range(num_draw):
            # mask out except for the first token in each seq
            tgt[:, 1:, :] = 0  

            for j in range(tgt.shape[1]): # iter within seq
                # model forward
                output_mean, output_var = self.model(src, tgt) # (N, T, 1), (N, T, 1)
                mean = output_mean[:, j, :] # (N, 1)
                var = output_var[:, j, :] # (N, 1)

                # append in final forecast value
                drawn_mean[i ,:, j, :] = mean
                drawn_var[i, :, j, :] = var

                # sample from given distribution and append in tgt; not executed in last iter
                if j < tgt.shape[1] - 1:
                    sampled_ts = torch.normal(mean, var) # (N, 1)
                    tgt[:, j+1, :] = sampled_ts

        # get median of draws
        median_mean = torch.quantile(drawn_mean, 0.5, dim=0) # (N, T, 1)
        median_var =torch.quantile(drawn_var, 0.5, dim=0) # (N, T, 1)

        return median_mean, median_var


def get_stats(samples, quantile):
    lbound = np.quantile(samples, 0.5 - quantile/2)
    median = np.quantile(samples, 0.5)
    ubound = np.quantile(samples, 0.5 + quantile/2)
    return lbound, median, ubound
