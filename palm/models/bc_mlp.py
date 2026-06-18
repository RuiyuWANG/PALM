import torch
import torch.nn as nn

from palm.models.blocks import (
    CropRandomizer,
    ObsNormalizer,
    RandomOverlay,
    make_mlp,
    make_visual_encoder,
)
from palm.tools.robo_augmentor import RoboAugmentor


class BCMLP(nn.Module):
    def __init__(self, args):
        super(BCMLP, self).__init__()

        self.args = args
        self.net = nn.ModuleDict()
        self.net.rgb_encoders = nn.ModuleDict()

        self.create_network()
        self.random_overlay = RandomOverlay(args.network.random_overlay)
        self.robo_augmentor = RoboAugmentor(args.network.robo_augmentor)
        self.obs_normalizer = ObsNormalizer(args.network.normalizer)
        self.crop_randomizer = CropRandomizer(args.network.crop_randomizer)
        self.criterion = self.get_criterion()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_criterion(self):
        if self.args.train.loss.type.lower() in ["mse", "l2"]:
            return nn.MSELoss()
        elif self.args.train.loss.type.lower() in ["huber", "smooth_l1", "smoothl1"]:
            return nn.SmoothL1Loss()
        elif self.args.train.loss.type.lower() == "l1":
            return nn.L1Loss()
        else:
            raise NotImplementedError(
                f"Loss {self.args.train.loss.type} not implemented"
            )

    def create_network(self):
        net_args = self.args.network
        self.rgb_keys = net_args.obs.rgb
        self.low_dim_keys = net_args.obs.low_dim
        for key in self.rgb_keys:
            self.net.rgb_encoders[key] = make_visual_encoder(net_args)
        self.net.mlp = make_mlp(net_args)

    def forward(self, batch: dict, saga=None, epoch_idx=None):
        self.robo_augmentor(batch)
        for rgb_view in batch["obs"]["rgb"].keys():
            x = self.random_overlay(batch["obs"]["rgb"][rgb_view], epoch_idx)
        batch = self.obs_normalizer(batch)
        
        embs = []
        for rgb_view in batch["obs"]["rgb"].keys():
            x, _ = self.crop_randomizer(batch["obs"]["rgb"][rgb_view])
            embs.append(self.net.rgb_encoders[rgb_view](x))
        for low_dim_state in self.low_dim_keys:
            x = batch["obs"]["low_dim"][low_dim_state]
            embs.append(batch["obs"]["low_dim"][low_dim_state])
        embs = torch.cat(embs, dim=1)
        embs = embs.to(dtype=torch.float32)
        
        return self.net.mlp(embs)

    def configure_optimizers(self):
        optimizer_args = self.args.train.optimizer
        lr = float(optimizer_args.lr)
        finetune_scale = float(optimizer_args.encoder_finetune_scale)
        param_groups = [
            {"params": self.net.rgb_encoders.parameters(), "lr": lr * finetune_scale},
            {"params": self.net.mlp.parameters(), "lr": lr},
        ]
        wd = float(optimizer_args.weight_decay)
        if self.args.train.optimizer.type == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=wd)
        elif self.args.train.optimizer.type == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
        elif self.args.train.optimizer.type == "sgd":
            optimizer = torch.optim.SGD(param_groups, lr=lr)
        else:
            raise NotImplementedError(
                f"Optimizer {self.args.train.optimizer.type} not implemented"
            )
        # TODO: Add scheduler
        return optimizer, None

    def get_label(self, batch):
        vec = []
        for action_mod, val in batch["actions"].items():
            vec.append(val)
        return torch.cat(vec, dim=1)

    def training_step(self, batch, batch_idx, logger, to_tb):
        action_pred = self.forward(batch, epoch_idx=logger.current_epoch)
        loss = self.criterion(self.get_label(batch), action_pred)
        logger.log_scalar(
            "policy_loss", loss.item(), mode="train", tb=to_tb, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx, logger, to_tb):
        action_pred = self.forward(batch)
        loss = self.criterion(self.get_label(batch), action_pred)

        if logger.global_step is None:
            to_tb = False
        logger.log_scalar(
            "policy_loss", loss.item(), mode="val", tb=to_tb, prog_bar=True
        )
        logger.dump_epoch_val_stats_to_tb()
        return loss

    def test_step(self, batch, batch_idx, logger, to_tb):
        loss = self.shared_step(batch, mode="test")
        logger.log_scalar("policy_loss", loss.item(), mode="test", prog_bar=True)
        logger.dump_epoch_test_stats_to_tb()
        return loss