import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

# user-made embedding
from .embedding import ConvTSEmbedding, WavenetTSEmbedding, FixedPositionEmbedding, LearnedPositionEmbedding


class Transformer_fcst(nn.Module):
    '''Transformer(Encoder-decoder) for time series forecasting.

    Point forecast (fcst_mode = 'point')
        Point forecast which directly predicts the time series value.
        MSE is a feasible loss when doing point forecast.

    Probabilistic forecast under Gaussian distribution (fcst_mode = 'gauss')
        Probabilistic forecast which predicts the params(mean and variance) of a Gaussian distribution.
        GaussianNLLLoss should be used.

    Args:
        fcst_mode: 'point' for point forecasting, 'gauss' for probabilistic forecasting under Gaussian distribution.
        seq_len: (length of source seq, length of target seq)
        embedding_dim: embedding dimension of time series embedding and position embedding.
        nhead: number of heads in multi-head attention. (embedding_dim) % nhead == 0 should be satisfied.
        num_layers: (number of encoder layers in encoder, number of decoder layers in decoder)
        device: torch.device to store position tensor and attention mask
        ts_embed: time series embedding. 'wavenet' for wavenet-style embedding, 'conv' for vanilla convolutional embedding
        pos_embed: 'learned' for learned position embedding, 'fixed' for fixed position embedding
        d_ff: dimension of linear layer in output layer
        dropout_rate: dropout rate for embedding and output layer
        **embedding_args: keyword arguments for embedding. 
    ''' 

    def __init__(self, fcst_mode, 
                       seq_len : tuple, 
                       embedding_dim : int, 
                       nhead : int, 
                       num_layers : tuple, 
                       device, 
                       input_dim : int = 1,
                       ts_embed='wavenet',
                       pos_embed='learned',
                       d_ff=512, 
                       dropout_rate=0.1,
                       **embedding_args):
        super(Transformer_fcst, self).__init__()

        # set fcst_mode
        if fcst_mode == 'point':
            self.fcst_mode = 'point'
        elif fcst_mode == 'gauss':
            self.fcst_mode = 'gauss'
        else:
            raise Exception("fcst_mode should be either 'point' or 'gauss'.")

        # arg check
        assert (embedding_dim) % nhead == 0, '(embedding_dim) % nhead == 0 should be satisfied.'
        assert (embedding_dim) % 2 == 0, 'embedding_dim % 2 == 0 should be satisfied (due to implementation)'

        # device
        self.device = device

        # input
        ## time series embedding
        ts_embed_func = ConvTSEmbedding if ts_embed == 'conv' else WavenetTSEmbedding
        self.src_ts_embedding = ts_embed_func(embedding_dim=embedding_dim, input_channel=input_dim, **embedding_args)
        self.tgt_ts_embedding = ts_embed_func(embedding_dim=embedding_dim, input_channel=input_dim, **embedding_args)

        ## position embedding   
        pos_embed_func = FixedPositionEmbedding if pos_embed == 'fixed' else LearnedPositionEmbedding
        self.src_pos_embedding = pos_embed_func(seq_len=seq_len[0], embedding_dim=embedding_dim)
        self.tgt_pos_embedding = pos_embed_func(seq_len=seq_len[1], embedding_dim=embedding_dim)

        ## embedding dropout
        self.src_embedding_dropout = nn.Dropout(p=dropout_rate)
        self.tgt_embedding_dropout = nn.Dropout(p=dropout_rate)
        
        # transformer
        ## transformer
        self.transformer = Transformer(d_model=embedding_dim, nhead=nhead, num_encoder_layers=num_layers[0], 
        num_decoder_layers=num_layers[1], dropout=dropout_rate, batch_first=True)

        ## target look-ahead mask
        self.tgt_mask = self._get_attn_mask(seq_len[1])

       
        # output
        ## forecast layer (depends on 'fcst_mode')
        if self.fcst_mode == 'point':
            self.point_fcst_layer = nn.Sequential(  # (N, seq_len, embedding_dim)
                nn.Linear(embedding_dim, d_ff),  # (N, seq_len, d_ff)
                nn.Dropout(dropout_rate),
                nn.LayerNorm(d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, 1)
            )

        else:
            self.mean_layer = nn.Sequential(  # (N, seq_len, embedding_dim)
                nn.Linear(embedding_dim, d_ff),  # (N, seq_len, d_ff)
                nn.Dropout(dropout_rate),
                nn.LayerNorm(d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, 1)
            )
            self.var_layer = nn.Sequential(  # (N, seq_len, embedding_dim)
                nn.Linear(embedding_dim, d_ff),  # (N, seq_len, d_ff)
                nn.Dropout(dropout_rate),
                nn.LayerNorm(d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, 1),
                nn.Softplus()
            )



    def forward(self, src, tgt): 
        '''src.shape == (N, S, dim), tgt.shape == (N, T, dim)
        'dim' could be any intger, but usually dim = 1 in time series setting.
        '''
        # shape check
        assert src.dim() == 3 & tgt.dim() == 3, "src and tgt should be 3-dimensional"

        # input
        ## timeseries embedding
        src_ts_embedded = self.src_ts_embedding(src)  # (N, S, embedding_dim)
        tgt_ts_embedded = self.tgt_ts_embedding(tgt)  # (N, T, embedding_dim)

        ## pos_embedding
        src_pos_embedded = self.src_pos_embedding(src) # (N, S, embedding_dim)
        tgt_pos_embedded = self.tgt_pos_embedding(tgt) # (N, T, embedding_dim)
        
        ## element-wise addition between two embeddings & dropout
        src_embedded_sum = src_ts_embedded + src_pos_embedded
        tgt_embedded_sum = tgt_ts_embedded + tgt_pos_embedded

        src_embedded_sum = self.src_embedding_dropout(src_embedded_sum)
        tgt_embedded_sum = self.tgt_embedding_dropout(tgt_embedded_sum)


        # transformer
        transformer_output = self.transformer(src=src_embedded_sum, tgt=tgt_embedded_sum, tgt_mask=self.tgt_mask) # (N, T, embedding_dim)


        # output
        ## forecast layer (depends on 'fcst_mode')
        if self.fcst_mode == 'point':
            forecast = self.point_fcst_layer(transformer_output) # (N, T, 1)
            return forecast

        else:
            mean = self.mean_layer(transformer_output)  # (N, T, 1)
            var = self.var_layer(transformer_output)  # (N, T, 1)
            return mean, var



    def _get_attn_mask(self, seq_len):
        '''
        Get attention mask(look-ahead mask) to use in decoder. 
        Attention mask is element-wisely summed with attention matrix.
        Upper trinagle = -1e8; otherwise = 0
        '''
        attn_mask = torch.full((seq_len, seq_len), -1e8, device=self.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        return attn_mask  # (seq_len, seq_len)





class TransformerDecoder_fcst(nn.Module):
    '''
    Transformer-decoder for time series forecasting.

    Point forecast (fcst_mode = 'point')
        Point forecast which directly predicts the time series value.
        MSE is a feasible loss when doing point forecast.

    Probabilistic forecast under Gaussian distribution (fcst_mode = 'gauss')
        Probabilistic forecast which predicts the params(mean and variance) of a Gaussian distribution.
        GaussianNLLLoss should be used.

    Args:
        fcst_mode: 'point' for point forecasting, 'gauss' for probabilistic forecasting under Gaussian distribution.
        seq_len: length of sequence.
        embedding_dim: embedding dimension of time series embedding and position embeddinng.
        nhead: number of heads in multi-head attention. (embedding_dim) % nhead == 0 should be satisfied.
        num_layers: number of decoder layers in decoder.
        device: torch.device to store position tensor and attention mask
        dropout_rate: dropout rate for embedding and output layer
        d_ff: dimension of linear layer in output layer
    '''

    def __init__(self, fcst_mode, seq_len, embedding_dim, nhead, num_layers, device, kernel_size=3, d_ff=512, dropout_rate=0.2):
        super(TransformerDecoder_fcst, self).__init__()

        # set fcst_mode
        if fcst_mode == 'point':
            self.fcst_mode = 'point'
        elif fcst_mode == 'gauss':
            self.fcst_mode = 'gauss'
        else:
            raise Exception("fcst_mode should be either 'point' or 'gauss'.")

        # arg check
        assert (embedding_dim) % nhead == 0, '(embedding_dim) % nhead == 0 should be satisfied.'

        # device
        self.device = device

        # input
        ## ts_embedding
        self.ts_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU()    
        )


        ## pos_embedding (for seq order; item(dim=0)-independent)
        pos_numpy = np.arange(seq_len)
        self.pos_tensor = torch.tensor(pos_numpy, dtype=torch.int32) # self.pos_tensor.shape == (seq_len)
        self.register_buffer('pos_tensor_for_batch_size_check', self.pos_tensor)
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

        ## dropout
        self.embedding_dropout = nn.Dropout(dropout_rate)


        # transformer-decoder
        ## causal convolution
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=self.kernel_size)

        ## decoder module
        decoder_layer = TransformerEncoderLayer(d_model = embedding_dim, nhead = nhead, batch_first=True)
        self.decoder = TransformerEncoder(decoder_layer, num_layers = num_layers)

        ## attention mask (look-ahead mask)
        self.attn_mask = self._get_attn_mask(seq_len) # self.attn_mask.shape == (seq_len, seq_len)
        self.register_buffer('attn_mask', self.attn_mask)


        # output
        ## output layer
        self.output_layer = nn.Sequential(  # (N, seq_len, embedding_dim)
            nn.Linear(embedding_dim, d_ff),  # (N, seq_len, d_ff)
            nn.Dropout(dropout_rate),
            nn.LayerNorm(d_ff),
            nn.ReLU()
        )
        
        ## forecast layer (depends on 'fcst_mode')
        if self.fcst_mode == 'point':
            self.point_fcst_layer = nn.Linear(d_ff, 1)

        elif self.fcst_mode == 'gauss':
            self.mean_layer = nn.Linear(d_ff, 1)
            self.var_layer = nn.Sequential(
                nn.Linear(d_ff, 1),
                nn.Softplus()
            )

        
            
    def forward(self, x): # x.shape == (N, seq_len, 1)
        # input
        ## timeseries_embedding
        ts_embedded = self.ts_embedding(x) # ts_embedded.shape == (N, seq_len, embedding_dim)

        ## pos_embedding
        pos_embedded = self.pos_embedding(self.pos_tensor) # pos_embedded.shape == (seq_len, embedding_dim)
        pos_embedded = pos_embedded.repeat(x.shape[0], 1, 1) # pos_embedded.shape == (N, seq_len, embedding_dim)

        ## element-wise addition between two embeddings & dropout
        embedded_sum = ts_embedded + pos_embedded 
        embedded_sum = self.embedding_dropout(embedded_sum) # embedded_sum.shape == (N, seq_len, embedding_dim)


        # transformer-decoder
        ## conv
        embedded_pad = torch.permute(embedded_sum, (0, 2, 1)) # embedded_pad.shape == (N, embedding_dim, seq_len)
        embedded_pad = F.pad(embedded_pad, (self.kernel_size-1,0)) # embedded_pad.shape == (N, embedding_dim ,seq_len+@)

        decoder_input = self.conv(embedded_pad) # decoder_input.shape == (N, embedding_dim ,seq_len)
        decoder_input = torch.permute(decoder_input, (0,2,1))  # decoder_input.shape == (N, seq_len, embedding_dim)

        ## decoder
        decoder_output = self.decoder(decoder_input, mask=self.attn_mask) # decoder_output.shape == (N, seq_len, embedding_dim)


        # output
        ## output layer
        output = self.output_layer(decoder_output) # output.shape == (N, seq_len, d_ff)

        ## forecast layer (depends on 'fcst_mode')
        if self.fcst_mode == 'point': # point forecast layer
            forecast = self.point_fcst_layer(output)  # forecast.shape == (N, seq_len, 1)
            return forecast

        elif self.fcst_mode == 'gauss': # probabilistic forecast layer
            mean = self.mean_layer(output)  # mean.shape == (N, seq_len, 1)
            var = self.var_layer(output)  # var.shape == (N, seq_len, 1)
            return mean, var



    def _get_attn_mask(self, seq_len):
        '''
        Get attention mask(look-ahead mask) to use in decoder. 
        Attention mask is element-wisely summed with attention matrix.
        Upper trinagle = -1e8; otherwise = 0
        '''
        attn_mask = torch.full((seq_len, seq_len), -1e8)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        return attn_mask # attn_mask.shape = (seq_len, seq_len)
