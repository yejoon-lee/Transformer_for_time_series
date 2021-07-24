import matplotlib
from matplotlib.pyplot import axis
import seaborn as sns
sns.set_style('darkgrid')
import numpy as np
import torch
import torch.nn as nn


def get_stats(samples, quantile):
    lbound = np.quantile(samples, 0.5 - quantile/2)
    median = np.quantile(samples, 0.5)
    ubound = np.quantile(samples, 0.5 + quantile/2)
    return lbound, median, ubound
    

class InferShortTerm:
    '''Short-term forecasting (i.e. rolling prediction)
    '''
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, 
                input : tuple):
        '''input: ((N, S, 1), (N, T, 1))
        '''
        # source and target
        src, tgt = input
        # forward
        mean, var = self.model(src, tgt)
        # to numpy
        return mean, var

    def eval(self, 
             input : tuple, 
             target : torch.Tensor, 
             criterion : nn.modules = nn.GaussianNLLLoss(), 
             metric_func : nn.modules = nn.L1Loss()):
        '''Evaluate loss(GaussianNLL by default) and metric(MAE by default)
        Args:
            input: ((N, S, 1), (N, T, 1))
            target: (N, T, 1)
        '''
        mean, var= self.forward(input) # (N, T, 1), (N, T, 1)
        loss = criterion(mean, target, var)
        metric = metric_func(mean, target)

        return loss, metric
    
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
        truth = truth[cut_head:]

        # draw samples & plot
        for i, ax in enumerate(axes):
            # get single sample from batch
            truth_sampled = truth[i].squeeze(-1) # (S+T, )
            mean_sampled = mean[i].squeeze(-1) # (T, )
            var_sampled = var[i].squeeze(-1) # (T, )

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
        ax.fill_between(self.x_axis[-len(medians):], lbounds, ubounds, alpha=0.3)
        ax.legend()



class InferLongTerm():
    '''Long-term forecasting using random sampling.
    Args:
    '''
    def __init__(self, model, beam_width):
        self.model = model
        self.beam_width = beam_width

    def forward(self, input:tuple):
        '''
        Args
            input: ((N, S, 1), (N, T, 1))
            target: (N, T, 1)
        '''
        # src and tgt
        src, tgt = input
        tgt[:, 1:, :] = 0 # mask out except for the first token in each seq        
        # forecast
        final_mean = torch.zeros(tgt.shape)
        final_var = torch.zeros(tgt.shape)

        for i in range(tgt.shape[1] - 1):
            # model forward
            output_mean, output_var = self.model(src, tgt)
            mean = output_mean[:, i, :] # (N, 1)
            var = output_var[:, i, :] # (N, 1)

            # append in final forecast value
            final_mean[:, i, :] = mean
            final_var[:, i, :] = var

            # sampled_ts = sample(mean, var) # (N, 1)
            # tgt[:, i+1, :] = sampled_ts


