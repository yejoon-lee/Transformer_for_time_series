import numpy as np

def make_input_target(data, 
                      src_len : int, 
                      tgt_len : int): 
    '''Make data(seq) into input(src and tgt combined) and target. 
    Only suitable for encoder-decoder structure.

    data: (N, src_len + tgt_len, ...)
    '''
    assert data.shape[1] == src_len + tgt_len, "seq_len == src_len + tgt_len should be satisfied. seq_len is expected to be axis=1"
    return data[:, :-1, ...], data[:, -tgt_len:, ...] # input: (N, seq_Len-1, ...), target: (N, tgt_Len, ...)


def make_src_tgt(input, 
                 src_len : int, 
                tgt_len : int): 
    '''Make input(seq) into src(input of encoder) and tgt(input of decoder).
    Only sutiable for encoder-decoder structure.
    Ideally, arg 'input' should be first value returned by make_input_target (i.e. make_input_target(args)[0])

    input: (N, src_len + tgt_len - 1, ...)
    '''
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
