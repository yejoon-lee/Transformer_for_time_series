import numpy as np

def make_input_target(data, src_len, tgt_len): # data: (N, src_len + tgt_len, ...)
    assert data.shape[1] == src_len + tgt_len, "seq_len == src_len + tgt_len should be satisfied. seq_len is expected to be axis=1"
    return data[:, :-1, ...], data[:, -tgt_len:, ...] # input: (N, seq_Len-1, ...), target: (N, tgt_Len, ...)


def make_src_tgt(input, src_len, tgt_len):
    assert input.shape[1] == src_len + tgt_len - 1, "seq_len == src_len + tgt_len - 1 should be satisfied. seq_len is expected to be axis=1"
    return input[:, :src_len, ...], input[:, src_len-1:, ...]  # src: (N, S, ...), tgt: (N, T, 1)


def cut_seq(data,
            window_len: int,
            stride: int = 1):
    '''Cut seq by given window_len and stride.

    Args:
        data: array which has more than 2 dims; (n_sample, seq_len, ...)
        window_len: window length
        stride: stride of window
    '''
    new_data = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):  # loop which ends with break
            if (j*stride + window_len) > data.shape[1]:
                break
            new_data.append(data[i, j*stride: j*stride + window_len, ...])

    return np.stack(new_data, axis=0)
