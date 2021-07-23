import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from tools.preprocess import make_input_target, make_src_tgt
from tools.train import EarlyStopping, make_loader
from tools.create_synthetic import create_multi
from model.model import Transformer_fcst


class RunSynthetic:
    '''Run for synthetic data using wandb.
    Methods:
        'make' and 'train' are the major funtions which are called in __init__. Others are supplementary.
    '''
    def __init__(self, project_name:str, config:dict, run_name=None, checkpoint_path='_model_pkls', verbose=2):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        wandb.login()
        with wandb.init(project=project_name, config=config):
            config = wandb.config
            if run_name:
                wandb.run.name = run_name

            self.make(config)
            self.train(config, checkpoint_path)

    def make(self, config):
        # load data
        train = create_multi(4500, config.t0)
        val = create_multi(500, config.t0)

        # dloader
        self.src_len, self.tgt_len = config.t0, 24

        input, target = make_input_target(train, self.src_len, self.tgt_len)
        self.train_dloader = make_loader(input, target, config.batch_size)
        input, target = make_input_target(val, self.src_len, self.tgt_len)
        self.val_dloader = make_loader(input, target, config.batch_size)

        # model
        self.model = Transformer_fcst(fcst_mode='gauss',
                                      seq_len=(self.src_len, self.tgt_len),
                                      embedding_dim=config.embedding_dim,
                                      nhead=config.nhead,
                                      num_layers=config.num_layers,
                                      device=self.device,
                                      ts_embed=config.ts_embed,
                                      pos_embed=config.pos_embed).to(self.device)

        # train configs
        self.criterion = nn.GaussianNLLLoss()
        self.metric_func = nn.L1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, betas=config.betas, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.sch_stepsize, gamma=config.sch_gamma)
        self.earlystopping = EarlyStopping(
            patience=config.es_patience, verbose=self.verbose)

    def train(self, config, checkpoint_path):
        print('Start training')
        wandb.watch(self.model, self.criterion, log='all', log_freq=config.batch_size)

        # verbose setting (whether to use tqdm)
        train_dloader_ = tqdm(self.train_dloader) if self.verbose else self.train_dloader

        # epoch loop
        for epoch in range(config.n_epoch):
            # train loop
            self.model.train()
            running_train_loss = 0.0
            running_train_metric = 0.0

            for input, target in train_dloader_:
                loss, metric = self.train_eval_batch(input, target)
                running_train_loss += loss
                running_train_metric += metric
            self.scheduler.step()

            # val loop
            self.model.eval()
            running_val_loss = 0.0
            running_val_metric = 0.0

            with torch.no_grad():
                for input, target in self.val_dloader:
                    loss, metric = self.train_eval_batch(input, target)
                    running_val_loss += loss
                    running_val_metric += metric

            # wandb log
            train_loss = running_train_loss / len(self.train_dloader)
            train_metric = running_train_metric / len(self.train_dloader)
            val_loss = running_val_loss / len(self.val_dloader)
            val_metric = running_val_metric / len(self.val_dloader)

            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_metric": train_metric,
                       "val_loss": val_loss, "val_metric": val_metric})

            # verbose setting (whether to print progress)
            if self.verbose > 1:
                print(f'[Epoch {epoch+1}] Train loss:', f'{train_loss:.4f}', end=' ')
                print('Val loss:', f'{val_loss:.4f}')

            # early stopping
            self.earlystopping(val_loss, self.model, checkpoint_path)
            if self.earlystopping.early_stop:
                self.model.load_state_dict(torch.load(checkpoint_path+'/checkpoint.pth'))
                break

        # export as onnx
        input = input.to(self.device)
        src, tgt = make_src_tgt(input, self.src_len, self.tgt_len)
        torch.onnx.export(self.model, (src, tgt), "model.onnx")
        wandb.save("model.onnx")

        print('Finish training')


    def train_eval_batch(self, input, target):
        # get src and tgt
        input, target = input.to(self.device), target.to(self.device)
        src, tgt = make_src_tgt(input, self.src_len, self.tgt_len)

        # forward
        mean, var = self.model(src, tgt)
        loss = self.criterion(mean, target, var)
        metric = self.metric_func(mean, target)

        # backward if training (not exectued in eval)
        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), metric.item()