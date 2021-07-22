# script (i.e. to be ran in terminal)

from run.run import RunSynthetic
config = dict(
    t0=96,
    embedding_dim=16,
    nhead=4,
    num_layers=(2,2),
    ts_embed='wavenet',
    pos_embed='learned',
    batch_size=64,
    lr=1e-3,
    betas=(0.9,0.98),
    sch_stepsize=5,
    sch_gamma=0.5,
    es_patience=5,
    n_epoch=50,
)
run = RunSynthetic('demo', config)

