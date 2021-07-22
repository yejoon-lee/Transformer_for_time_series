import numpy as np
import matplotlib.pyplot as plt

def create_multi(sample_num, t0):
    y = np.zeros((sample_num, t0+24, 1))

    for i in range(sample_num):
        y[i,:, :] = create_single(t0)

    return y


def create_single(t0):
    # A
    a1 = np.random.uniform(0, 60)
    a2 = np.random.uniform(0, 60)
    a3 = np.random.uniform(0, 60)
    a4 = max(a1, a2)

    a_list = [a1, a2, a3, a4]

    # range
    ranges = [0, 12, 24, t0, t0+24]

    # axis
    y_axis = np.zeros(ranges[-1])

    # make sinusoidal curves
    for i, a in enumerate(a_list):
        x_arr = np.arange(ranges[i], ranges[i+1])
        cycle = 12 if i == 3 else 6

        for x in x_arr:
            noise = np.random.normal()
            y_axis[x] = a * np.sin(np.pi * x / cycle) + 72 + noise

    y_axis = np.expand_dims(y_axis, -1)
    
    return y_axis
