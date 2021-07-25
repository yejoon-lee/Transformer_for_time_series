import numpy as np

def create_multi(sample_num, src_len, tgt_len):
    y = np.zeros((sample_num, src_len+tgt_len, 1))

    for i in range(sample_num):
        y[i,:, :] = create_single(src_len, tgt_len)

    return y # (N, S+T, 1)


def create_single(src_len, tgt_len):
    # A
    a1 = np.random.uniform(0, 60)
    a2 = np.random.uniform(0, 60)
    a3 = np.random.uniform(0, 60)
    a4 = max(a1, a2)

    a_list = [a1, a2, a3, a4]

    # range
    ranges = [0, tgt_len//2, tgt_len, src_len, src_len+tgt_len]

    # axis
    y_axis = np.zeros(ranges[-1])

    # make sinusoidal curves
    for i, a in enumerate(a_list):
        x_arr = np.arange(ranges[i], ranges[i+1])
        cycle = tgt_len//2 if i == 3 else tgt_len//4

        for x in x_arr:
            noise = np.random.normal()
            y_axis[x] = a * np.sin(np.pi * x / cycle) + 72 + noise

    y_axis = np.expand_dims(y_axis, -1)
    
    return y_axis