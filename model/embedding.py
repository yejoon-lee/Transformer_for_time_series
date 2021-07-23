import torch
import torch.nn as nn
import torch.nn.functional as F
import math

    
class WavenetTSEmbedding(nn.Module):
    '''
    Wavenet-style embedding for time series value.
    Convolutions are only applied to the left. (causal convolution)
    '''
    def __init__(self, embedding_dim, dilation_list = (1,2,4,8), input_channel=1):
        super(WavenetTSEmbedding, self).__init__()

        self.fc = nn.Linear(input_channel, embedding_dim)

        self.dilation_list = dilation_list
        self.conv_list = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, 
        out_channels=embedding_dim, kernel_size=2, dilation=d) for d in self.dilation_list])


    def forward(self, x): # (N, seq_len, input_channel)
        x = self.fc(x) # (N, seq_len, embedding_dim)

        x = torch.permute(x, (0,2,1)) 
        for conv, d in zip(self.conv_list, self.dilation_list):
            x = F.pad(x, (d,0)) 
            x = conv(x)
            
        return torch.permute(x, (0,2,1)) # (N, seq_len, embedding_dim)


class ConvTSEmbedding(nn.Module):
    '''
    Causal convolutional embedding for time series value.
    Convolutions are only applied to the left. (causal convolution)
    '''
    def __init__(self, embedding_dim, kernel_size=3, conv_depth=4, input_channel=1):
        super(ConvTSEmbedding, self).__init__()

        self.fc = nn.Linear(input_channel, embedding_dim)

        self.kernel_size = kernel_size
        self.conv_list = nn.ModuleList([nn.Conv1d(
            in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=self.kernel_size) for _ in range(conv_depth)])


    def forward(self, x): # (N, seq_len, input_channel)
        x = self.fc(x) # (N, seq_len, embedding_dim)

        x = torch.permute(x, (0,2,1)) # (N, embedding_dim, seq_len)
        for conv in self.conv_list:
            x = F.pad(x, (self.kernel_size-1,0))
            x = conv(x)

        return torch.permute(x, (0,2,1)) # (N, seq_len, embedding_dim)


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_dim):
        super(LearnedPositionEmbedding, self).__init__()

        pos_tensor = torch.arange(seq_len)
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

        self.register_buffer('pos_tensor', pos_tensor)

    def forward(self, x): # x.shape == (N, ??, ??)
        pos_embedded = self.pos_embedding(self.pos_tensor) # pos_embedded.shape == (seq_len, embedding_dim)
        return pos_embedded.repeat(x.shape[0], 1, 1) # (N, seq_len, embedding_dikm)


class FixedPositionEmbedding(nn.Module):
    '''
    Fixed position embedding in "Attention is all you need".
    Code from "Informer".
    '''
    def __init__(self, seq_len, embedding_dim):
        super(FixedPositionEmbedding, self).__init__()

        pos_embedding = torch.zeros((seq_len, embedding_dim)).float()
        pos_embedding.requires_grad = False

        pos_tensor = torch.arange(seq_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embedding_dim, 2).float()
                    * -(math.log(10000.0) / embedding_dim)).exp()

        pos_embedding[:, 0::2] = torch.sin(pos_tensor * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos_tensor * div_term)

        pos_embedding.unsqueeze_(0) # dimension for batch

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x): # (N, ??, ??)
        return self.pos_embedding.repeat(x.shape[0], 1, 1) # (N, seq_len, embedding_dim)
