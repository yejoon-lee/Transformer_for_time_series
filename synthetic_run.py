from run.run import RunSynthetic

config = dict(
    src_len=96,
    tgt_len=24,
    embedding_dim=16,
    nhead=8,
    num_layers=(2, 2),
    ts_embed='wavenet',
    pos_embed='fixed',
    batch_size=32,
    lr=5e-4,
    betas=(0.9, 0.98),
    sch_stepsize=10,
    sch_gamma=0.7,
    es_patience=5,
    n_epoch=50,
)

synth = RunSynthetic('transformer_for_time_series_synthetic_data', config, verbose=0, watch=False, checkpoint_path='checkpoints/synthetic')