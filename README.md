# Transformer_for_time_series
Project done in SNU class, Field Application of IoT, AI, Big Data 1, in 2021 Summer.

## Dataset
> Synthetic dataset  

- See tools/create_synthetic.py.   
- Following the setup provided in *Li, Shiyang, et al., “Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting”, NeurIPS, 2019*.  

> Coin dataset  

- 'Open' price of cryptocurrencies per minute. 
- Data is not provided in this repo due to its size.

## Model
- Use standard **encoder-decoder structure(Transformer)** from *Vaswani et al., "Attention is all you need", NeurIPS, 2017*.
- **Embed** time series value by **convolution**.
- **Probabilistic forecast** is applied in which the model outputs the parameters of certain probability distribution and trained to maximize the log likelihood. Gaussian distribution is used here, hence the model ouputs mean and variance.  
*(See Salinas et al., "DeepAR: Probabilistic forecasting with autoregressive recurrent networks", International Journal of Forecasting, 2020 for more info)*

## Weights & Biases Dashborad
> Synthetic dataset  

[Link](https://wandb.ai/wittgensteinian/transformer_for_time_series_synthetic_data)

> Coin dataset  

[Link](https://wandb.ai/wittgensteinian/transformer_for_time_series_coin_data)
