import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from nt_xent import NTXentLoss
from resnet_simclr import ResNetSimCLR
import yaml

torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder,config):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder,'config.yaml'), 'w') as f:
            config = yaml.dump(config, stream=f,
                            default_flow_style=False, sort_keys=False)


class SimCLR(object):
    def __init__(self, train_dataloader,val_dataloader, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(self.config['summary_dir'])
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.nt_xent_criterion = NTXentLoss(self.device, config["batch_size"], **config["loss"])
        self.gradient_accumulate_every = self.config['gradient_accumulate_every']

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        train_dataloader, val_dataloader=self.train_dataloader, self.val_dataloader
        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config["weight_decay"]))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0, last_epoch=-1)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")

        # save config file
        _save_config_file(model_checkpoints_folder,config=self.config)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        total_loss = torch.tensor(0.).to(self.device)
        for epoch_counter in range(self.config["epochs"]):
            for xis, xjs, _ in train_dataloader:

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter) / self.gradient_accumulate_every
                loss.backward()
                
                n_iter += 1
                total_loss += loss.detach().item()
                if (n_iter) % self.gradient_accumulate_every == 0:
                    optimizer.step()
                    if n_iter//self.gradient_accumulate_every % self.config["log_every_n_steps"] == 0:
                        self.writer.add_scalar("train_loss", total_loss, global_step=n_iter//self.gradient_accumulate_every)
                    optimizer.zero_grad()
                    total_loss = 0
                
            # validate the model if requested
            if epoch_counter % self.config["eval_every_n_epochs"] == 0:
                valid_loss = self._validate(model, val_dataloader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(
                        model.state_dict(), os.path.join(model_checkpoints_folder, "model.pth"),
                    )
                self.writer.add_scalar("validation_loss", valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            torch.save(
                        model.state_dict(), os.path.join(model_checkpoints_folder, f"model-{epoch_counter}.pth"),
                    )
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar("cosine_lr_decay", scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = self.config["fine_tune_from"]
            state_dict = torch.load(os.path.join(checkpoints_folder, "model.pth"))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        print("start tensorboard to monitor training")

        return model

    def _validate(self, model, val_dataloader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for xis, xjs, _ in val_dataloader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            if counter != 0:
                valid_loss /= counter
        model.train()
        return valid_loss
