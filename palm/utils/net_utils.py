import json
import os
from tqdm import tqdm
from easydict import EasyDict
from tabulate import tabulate

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage

from palm.utils.palm_utils import reps_to_shapes
from palm.utils.print_utils import divider, plain_divider, print_dict_tree


def should_be_left_blank_or_match(str_for_check: str, str_to_match: str) -> bool:
    if str_for_check in ["", str_to_match, None]:
        return True
    raise ValueError(f"{str_for_check} should be left blank or match {str_to_match}")


def parse_network_configs(config_path: str, print_config: bool = False) -> EasyDict:
    with open(config_path, "r") as f:
        palm_config = json.load(f)
    cfg = EasyDict(palm_config)

    # Ensure network and dataset configurations match or are left blank
    if should_be_left_blank_or_match(cfg.network.obs, cfg.dataset.obs):
        cfg.network.obs = cfg.dataset.obs
    if should_be_left_blank_or_match(cfg.network.actions, cfg.dataset.actions):
        cfg.network.actions = cfg.dataset.actions
    if not cfg.network.encoder.pretrained:
        cfg.train.optimizer.encoder_finetune_scale = 1.0

    if should_be_left_blank_or_match(
        cfg.network.n_steps_ahead, cfg.dataset.n_steps_ahead
    ):
        cfg.network.n_steps_ahead = cfg.dataset.n_steps_ahead

    # Ensure n_steps_ahead is supported
    assert cfg.network.n_steps_ahead == 1, "n_steps_ahead > 1 not supported yet"

    # Calculate action shape
    # if should_be_left_blank_or_match(cfg.network.action_shape, cfg.network.action_shape):
    if cfg.network.low_dim_shape is None:
        cfg.network.action_shape = (
            sum(reps_to_shapes(cfg.network.actions)) * cfg.network.n_steps_ahead
        )
    cfg.network.normalizer.action_shape = cfg.network.action_shape

    # Calculate low_dim_shape if not provided
    if cfg.network.low_dim_shape is None:
        cfg.network.low_dim_shape = sum(reps_to_shapes(cfg.network.obs.low_dim))

    # Set dataset path for normalizer
    cfg.network.normalizer.dataset_path = cfg.dataset.dataset_path

    # Print configuration if required
    if print_config:
        print_network_configs(cfg)

    return cfg


def print_network_configs(configs, width=60):
    required_keys = ["dataset", "network", "train"]
    for key in required_keys:
        assert key in configs.keys(), f"Key {key} not found in config"
    print("\n")
    divider("Network Configs", line_max=60, char="-")
    print_dict_tree(configs.network)
    divider("Training Configs", line_max=60, char="-")
    print_dict_tree(configs.train, wrap_width=50)
    divider("Dataset Configs", line_max=60, char="-")
    print_dict_tree(configs.dataset, wrap_width=50)
    print("\n")


# -------------------------------------------- Logger ------------------------------------------- #

class Logger:
    def __init__(self, args):
        self.args = args
        self.batch_stats = {"train": {}, "val": {}, "test": {}}
        self.batch_raw_data = {"train": {}, "val": {}, "test": {}}
        self.prog_bar_dict = {}
        self.log_dir = os.path.join(args.experiment_dir, args.experiment)
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "runs"))
        self.end_of_val_batch = False
        self.end_of_train_batch = False
        self.end_of_test_batch = False
        self.global_step = None
        self.current_epoch = 0
        self.log_interval = args.log_interval

    def log_scalar(
        self, name: str, value: float, mode="train", tb=False, prog_bar=False
    ):
        assert mode in self.batch_raw_data.keys()
        assert (tb and self.global_step is not None) or not tb, (
            "global_step must be specified when tb=True"
        )
        if prog_bar:
            self.prog_bar_dict[name] = round(
                value, 4
            )  # not averaged, displayed in real time
        if name not in self.batch_raw_data[mode].keys():
            self.batch_raw_data[mode][name] = [value]
        else:
            self.batch_raw_data[mode][name].append(value)
        if tb and self.global_step % self.log_interval:
            self.writer.add_scalar(
                f"{mode.capitalize()}-{name}", value, self.global_step
            )

    def _compute_batch_stats(self):
        for mode, states in self.batch_raw_data.items():
            for metric, value in states.items():
                assert isinstance(value, list), "value must be stored in a list"
                if value == []:
                    return
                value = torch.tensor(value)
                _mean = torch.mean(value)
                _std = torch.std(value)
                self.batch_stats[mode][metric] = {"mean": _mean, "std": _std}

    def dump_epoch_val_stats_to_tb(self):
        if not self.end_of_val_batch:
            return
        self._compute_batch_stats()
        for mode, states in self.batch_stats.items():
            if mode == "val":
                for metric, value in states.items():
                    self.writer.add_scalar(
                        f"{mode.capitalize()}-{metric}",
                        value["mean"],
                        self.current_epoch,
                    )

    def dump_epoch_test_stats_to_tb(self):
        if not self.end_of_test_batch:
            return
        self._compute_batch_stats()
        for mode, states in self.batch_stats.items():
            if mode == "test":
                for metric, value in states.items():
                    self.writer.add_scalar(
                        f"{mode.capitalize()}-{metric}",
                        value["mean"],
                        self.current_epoch,
                    )

    def log_stats_to_file(self, title_name, title_value):
        self._compute_batch_stats()
        with open(
            os.path.join(self.log_dir, "training_stats.txt"), encoding="utf-8", mode="a"
        ) as f:
            f.write(f"{title_name}-{title_value}:\n")
            for mode, states in self.batch_stats.items():
                for metric, value in states.items():
                    f.write(
                        f"  {mode.capitalize()}-{metric}: {value['mean']:.4f}+-{value['std']:.4f}\n"
                    )
        f.close()

    def clear(self):
        self.batch_raw_data = {"train": {}, "val": {}, "test": {}}
        self.batch_stats = {"train": {}, "val": {}, "test": {}}
        self.end_of_train_batch = False
        self.end_of_val_batch = False
        self.end_of_test_batch = False

    def print_batch_stats(self, prefix="", fancy=False, print_std=False):
        out_str = prefix
        padded_space = " " * len(prefix)
        self._compute_batch_stats()
        header = ["Mode", "Metric", "Mean", "Std"]
        row = []
        longest_mode = max([len(mode) for mode in self.batch_stats.keys()])
        for mode, states in self.batch_stats.items():
            if states != {}:
                if mode == "train":
                    out_str += f"[{mode.capitalize().ljust(longest_mode)}]"
                else:
                    out_str += (
                        f"\n{padded_space}[{mode.capitalize().ljust(longest_mode)}]"
                    )
                for metric, value in states.items():
                    if print_std:
                        out_str += f" {metric}:{value['mean']:.4f}+-{value['std']:.3f},"
                    else:
                        out_str += f" {metric}:{value['mean']:.4f},"
                    row.append(
                        [
                            mode.capitalize(),
                            metric,
                            f"{value['mean']:.4f}",
                            f"{value['std']:.3f}",
                        ]
                    )
                out_str = out_str[:-1]  # remove the last comma
        if fancy:
            out_str = tabulate(row, headers=header)
        tqdm.write(out_str)

    def print_args(self):
        divider("Current Config")
        for key, value in vars(self.full_args).items():
            print(f"  {key}: {value}")
        plain_divider()
        print("\n")

    def log_config(self, config, overwrite=False):
        config_path = os.path.join(self.log_dir, "config.json")
        if not os.path.exists(config_path) or overwrite:
            with open(config_path, encoding="utf-8", mode="a") as f:
                cfg_dict = dict(config) if isinstance(config, EasyDict) else config
                json.dump(cfg_dict, f, indent=4)
                print(f"[Trainer] Config is backed-up as {config_path}")
            f.close()

    def log_image(self, caption: str, image, save_to_local=False):
        self.writer.add_image(caption, image, self.current_epoch)
        if save_to_local:
            im_path = os.path.join(self.log_dir, "logged_images")
            os.makedirs(im_path, exist_ok=True)
            im_name = (
                f"{caption}.png"
                if caption in ["target", "Target", "GT", "input", "Input"]
                else f"{caption}-epoch-{self.current_epoch}.png"
            )
            ToPILImage()(image).save(os.path.join(im_path, im_name))