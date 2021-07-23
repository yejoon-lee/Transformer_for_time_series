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

    def forward(self, input:tuple):
        '''input: ((N, S, 1), (N, T, 1))
        '''
        # source and target
        src, tgt = input
        # forward
        mean, var = self.model(src, tgt)
        # to numpy
        return mean.cpu().detach().numpy(), var.cpu().detach().numpy()

    def eval(self, input:tuple, target, criterion=nn.GaussianNLLLoss(), metric_func=nn.L1Loss()):
        '''Evaluate loss(GaussianNLL by default) and metric(MAE by default)
        Args:
            input: ((N, S, 1), (N, T, 1))
            target: (N, T, 1)
        '''
        mean, var= self.forward(input) # (N, T, 1), (N, T, 1)
        loss = criterion(mean, target, var)
        metric = metric_func(mean, target)

        return loss, metric
    
    def plot(self, input:tuple, target, axes, cut_head:int=0, quantile:float=0.5, num_draw:int=10000):
        '''Plot prediction with ground truth
        Args:
            input: ((N, S, 1), (N, T, 1)) = (src, tgt)
            target: (N, T, 1)
            axes: axes.size should be (*, 1) (i.e. single column)
            cut_head: Cut head of the x axis to enhance visibility
            quantile: Quantile to be shown in plot
            num_draw: Numbers drawing samples from probability distribution
        '''
        # forward
        src, tgt = input # (N, S, 1), (N, T, 1)
        mean, var = self.forward(input) # (N, T, 1), (N, T, 1)

        # x-axis
        x_axis = np.arange(src.shape[1] + tgt.shape[1])
        # ground truth
        truth = np.concatenate((src,target), axis=1) # (N, S+T, 1)
        # cut head to enhance visibility
        x_axis = x_axis[cut_head:]
        truth = truth[cut_head:]

        # draw samples & plot
        sample_num_list = np.random.choice(truth.shape[0], len(axes), replace=False)
        for ax, sample_num in zip(axes, sample_num_list):
            # get single sample from batch
            truth_sampled = truth[sample_num] # (S+T, 1)
            mean_sampled = mean[sample_num] # (T, 1)
            var_sampled = var[sample_num] # (T, 1)

            # draw samples from probability distribution
            samples = np.random.normal(mean_sampled, var_sampled, (num_draw, len(mean_sampled))).T # (T, num_iter)
            lbounds, medians, ubounds = [], [], []

            for token_sample in samples:
                lbound, median, ubound = get_stats(token_sample, quantile)
                lbounds.append(lbound)
                medians.append(median)
                ubounds.append(ubound)

            ax.plot(x_axis, truth_sampled, label='ground truth')
            ax.plot(x_axis[-len(median):], medians, label='forecast')
            ax.fill_between(x_axis[-len(median):], lbounds, ubounds)
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


