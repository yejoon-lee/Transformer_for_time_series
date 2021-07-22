
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import wandb

from tools.train import EarlyStopping
from tools.create_synthetic import create_multi
from model.model import Transformer_fcst


class RunSynthetic:
    def __init__(self, project_name, config, checkpoint_path='_model_pkls', verbose=2):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        wandb.login()
        with wandb.init(project=project_name, config=config):
            config = wandb.config
            self.make(config)
            self.train(config, checkpoint_path)

    def make(self, config):
        # load data
        train = create_multi(4500, config.t0)
        val = create_multi(500, config.t0)

        # dloader
        self.src_len, self.tgt_len = config.t0, 23

        input, target = self.make_src_tgt(train, self.src_len, self.tgt_len)
        self.train_dloader = self.make_loader(input, target, config.batch_size)
        input, target = self.make_src_tgt(val, self.src_len, self.tgt_len)
        self.val_dloader = self.make_loader(input, target, config.batch_size)

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
        wandb.watch(self.model, self.criterion,
                    log='all', log_freq=config.batch_size)

        # verbose setting (whether to use tqdm)
        to_iter = tqdm(self.train_dloader) if self.verbose else self.train_dloader

        # epoch loop
        for epoch in range(config.n_epoch):
            # train loop
            self.model.train()
            running_train_loss = 0.0
            running_train_metric = 0.0

            for input, target in to_iter:
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
                print(f'[Epoch {epoch+1}] Train loss:',
                      f'{train_loss:.4f}', end=' ')
                print('Val loss:', f'{val_loss:.4f}')

            # early stoppingc
            self.earlystopping(val_loss, self.model, checkpoint_path)
            if self.earlystopping.early_stop:
                print('Early Stopping')
                self.model.load_state_dict(torch.load(
                    checkpoint_path+'/checkpoint.pth'))
                break

        # export as onnx
        src, tgt = input[:, :self.src_len, ...].to(
            self.device), input[:, self.src_len:, ...].to(self.device)
        torch.onnx.export(self.model, (src, tgt), "model.onnx")
        wandb.save("model.onnx")

    def train_eval_batch(self, input, target):
        # get source and target
        input, target = input.to(self.device), target.to(self.device)
        src, tgt = input[:, :self.src_len, ...], input[:, self.src_len:, ...]

        # forward
        mean, var = self.model(src, tgt)
        loss = self.criterion(mean, target, var)
        metric = self.metric_func(mean, target)

        # backward if necessary
        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), metric.item()

    def make_src_tgt(self, data, src_len, tgt_len):
        assert data.shape[1] == src_len + tgt_len + \
            1, "seq_len == self.src_len +self. tgt_len + 1 should be satisfied."
        return data[:, :-1, ...], data[:, -tgt_len:, ...]

    def make_loader(self, input, target, batch_size):
        dset = TensorDataset(torch.Tensor(input), torch.Tensor(target))
        dloader = DataLoader(dset, shuffle=True, batch_size=batch_size)
        return dloader
