import matplotlib.pyplot as plt
from numpy.lib.function_base import quantile
import seaborn as sns
sns.set_style('darkgrid')
import numpy as np



def plot_eval_gauss(input, target, mean, var, ax, cut_head=0, quantile=0.5): # inputs are numpy 1d array
    # check cut_head
    if cut_head > len(input):
        raise Exception("cut_head should be smaller than len(input).")

    # x-axis
    x = np.arange(len(input)+1)

    # real data
    real = np.zeros(len(input) + 1)
    real[:len(input)] = input
    real[-1] = target[-1]

    # forecast
    q1, q2, q3 = sample_quantile(mean, var, quantile)
        
    # cut head (cut first values in the seq)
    x = x[cut_head:]
    real = real[cut_head:]

    # plot
    ax.plot(x, real, label='real')
    ax.plot(x[-len(q2):], q2, label='forecast')
    ax.fill_between(x[-len(q2):], q1, q3, alpha=0.3)
    ax.legend()


def plot_eval_point(input, target, forecast, ax, cut_head=0, quantile=0.5):  # inputs are numpy 1d array
    # x-axis
    x = np.arange(len(input)+1)

    # real data
    real = np.zeros(len(input) + 1)
    real[:len(input)] = input
    real[-1] = target[-1]

    # cut head (cut first values in the seq)
    x = x[cut_head:]
    real = real[cut_head:]

    # plot
    ax.plot(x, real, label='real')
    ax.plot(x[-len(forecast):], forecast, label='forecast')
    ax.legend()


def sample_quantile(mean, var, quantile):
    '''
    Sample from given distribution.

    Args:
        mean: 1d array
        var: 1d array whose size is equal to mean.
        quantile: float between 0 and 1; calculate range for given quantile
    '''
    q1, q2, q3 = [], [], []
    for mu, sigma in zip(mean, var):
        std = pow(sigma, 0.5)
        samples = np.random.normal(mu, std, 10000)

        # get quantiles
        q1.append(np.quantile(samples, 0.5 - quantile/2))
        q2.append(np.quantile(samples, 0.5))
        q3.append(np.quantile(samples, 0.5 + quantile/2))

    return q1, q2, q3
