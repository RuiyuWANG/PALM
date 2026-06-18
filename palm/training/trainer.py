import glob
import os
import shutil
import random
import numpy as np
from datetime import datetime

import argparse
import torch
from tqdm import tqdm

from palm.data.dataset import PalmDataset
from palm.data.dataset_manager import DatasetManager
from palm.utils.net_utils import Logger
from palm.models.bc_mlp import BCMLP
from palm.evaluation.rollout import evaluate_policy
from palm.utils.print_utils import divider
from palm.utils.rlbench_utils import RLBenchAgent
from palm.utils.net_utils import parse_network_configs

class Trainer:
    def __init__(self, *, args, Net, Dataset):
        self.args = args.train
        self.args_network = args
        assert args.train.experiment is not None, "Experiment Name is not specified"
        self.logger = Logger(args.train)
        self.current_epoch = -1
        self.lowest_loss = float("inf")
        self.highest_succ = 0
        self.last_epoch = -1
        self.n_iter = 0
        self.logger.log_config(args)
        
        # device
        # TODO: add support for multiple GPUs
        assert torch.cuda.is_available(), "CUDA device not found"
        self.device = torch.device("cuda")
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count == 0:
            raise Exception("No GPU Found, Existing ...")
        self.net = Net(args=args).to(self.device)
        
        # Dataset
        dataset = DatasetManager(CustomDataset=Dataset, args=args)
        self.training_dataloader = dataset.train_dataloader()
        self.validation_dataloader = dataset.validation_dataloader()
        self.test_dataloader = dataset.test_dataloader()
        
        # Configure optimizers and load checkpoint
        self.optimizer = None
        self.lr_scheduler = None
        self.configure_optimizers()

        divider("Statistics")
        self.load_ckpt()
        self.logger.current_epoch = self.last_epoch

    def configure_optimizers(self):
        self.optimizer, self.lr_scheduler = self.net.configure_optimizers()

    def training_step(self, batch, batch_idx, to_tb=True):
        batch = self.parse_batch(batch)
        if batch_idx == len(self.training_dataloader) - 1:
            self.logger.end_of_train_batch = True
        self.logger.global_step = self.n_iter
        return self.net.training_step(batch, batch_idx, self.logger, to_tb=to_tb)

    def validation_step(self, batch, batch_idx, to_tb=True):
        batch = self.parse_batch(batch)
        if batch_idx == len(self.validation_dataloader) - 1:
            self.logger.end_of_val_batch = True
        return self.net.validation_step(batch, batch_idx, self.logger, to_tb=to_tb)

    def test_step(self, batch, batch_idx, to_tb=True):
        batch = self.parse_batch(batch)
        if batch_idx == len(self.test_dataloader) - 1:
            self.logger.end_of_test_batch = True
        return self.net.test_step(batch, batch_idx, self.logger, to_tb=to_tb)

    def train(self):
        epoch_progress = tqdm(
            range(self.last_epoch + 1, self.args.num_epochs),
            dynamic_ncols=True,
        )
        for epoch_idx in epoch_progress:
            self.current_epoch = epoch_idx
            self.logger.current_epoch = self.current_epoch
            epoch_progress.set_description(f"Epoch {epoch_idx}/{self.args.num_epochs}")
            self.net = self.net.train()
            for batch_idx, batch in enumerate(self.training_dataloader):
                self.n_iter += 1
                self.logger.prog_bar_dict["Batch"] = (
                    f"{batch_idx:04d}/{len(self.training_dataloader)}"
                )
                epoch_progress.set_postfix(self.logger.prog_bar_dict)
                #
                self.optimizer.zero_grad()
                loss = self.training_step(batch, batch_idx)
                loss.backward()
                if hasattr(self.net, "momentum_update"):
                    self.net.momentum_update()
                if self.args.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip_grad_norm)
                self.optimizer.step()
            if self.lr_scheduler and not isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step()

            # run validation, rollouts, save ckpt
            if (
                epoch_idx % self.args.save_interval == 0 or epoch_idx == self.args.num_epochs - 1
            ) and epoch_idx != 0:
                epoch_progress.set_description("Validating ...")
                rollout_success, validate_loss = None, None
                if self.args.eval.enabled:
                    save_mode = "succ"
                    self.net = self.net.eval()
                    agent = RLBenchAgent(self.net, self.args_network)
                    rollout_success, _ = evaluate_policy(
                        agent, self.args_network.dataset.task_name, self.args.eval
                    )
                    self.validate(print_to_terminal=False)
                    self.net = self.net.train()
                else:
                    save_mode = "val"
                    self.net = self.net.eval()
                    validate_loss = self.validate(print_to_terminal=True)
                    self.net = self.net.train()
                self.save_ckpt(
                    {"loss": validate_loss, "succ": rollout_success},
                    mode=save_mode,
                    print_on_save=False,
                )
                try:
                    self.test(print_to_terminal=False)
                except NotImplementedError:
                    pass

            prefix = f"[{datetime.now().strftime('%H:%M:%S')}] Epoch-{epoch_idx} => "
            self.logger.print_batch_stats(prefix)
            self.logger.log_stats_to_file("Epoch", self.current_epoch)
            self.logger.clear()

    def validate(self, print_to_terminal=True, to_tb=True):
        self.net = self.net.eval()
        mean_validate_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.validation_dataloader):
                batch_val_loss = self.validation_step(batch, batch_idx, to_tb=to_tb)
                mean_validate_loss += (batch_val_loss - mean_validate_loss) / (batch_idx + 1)
        if print_to_terminal:
            self.logger.print_batch_stats()
        return mean_validate_loss

    def test(self, print_to_terminal=True, to_tb=True):
        raise NotImplementedError

    # ---------------------------- Utility Functions ---------------------------- #
    def parse_batch(self, batch):
        assert isinstance(batch, dict), "batch must be a dictionary"
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                self.parse_batch(v)
        return batch

    def cleanup_saved_ckpts(self, dir_path, mode):
        exp_paths = glob.glob(os.path.join(dir_path, "epoch*.tar"))

        def get_value_from_filename(x):
            if mode == "val":
                return float(x.split("-loss=")[1].split("_")[0])
            elif mode == "succ":
                return float(x.split("-rate=")[1].split("_")[0])

        def get_epoch_from_filename(x):
            return int(x.split("epoch_")[1].split("_")[0])

        if mode == "val":
            ckpt_paths_sorted_by_value = sorted(
                exp_paths, key=get_value_from_filename, reverse=False
            )
        elif mode == "succ":
            ckpt_paths_sorted_by_value = sorted(
                exp_paths, key=get_value_from_filename, reverse=True
            )
        ckpt_paths_sorted_by_epoch = sorted(exp_paths, key=get_epoch_from_filename, reverse=True)

        top_ckpts = set(ckpt_paths_sorted_by_value[: self.args.keep_top_k])
        last_ckpts = (
            set(ckpt_paths_sorted_by_epoch[: self.args.keep_last_k])
            if self.args.keep_last_k > 0
            else set()
        )
        protected_ckpts = top_ckpts.union(last_ckpts)
        ckpt_paths_to_delete = [x for x in exp_paths if x not in protected_ckpts]
        for x in ckpt_paths_to_delete:
            os.remove(x)
        return ckpt_paths_sorted_by_value[0] if ckpt_paths_sorted_by_value else None

    def fetch_latest_ckpt(self, dir_path):
        exp_paths = glob.glob(os.path.join(dir_path, "epoch*.tar"))
        if exp_paths == []:
            return None
        ckpt_paths = sorted(exp_paths, key=lambda x: os.path.getmtime(x), reverse=True)
        return ckpt_paths[0]

    def load_ckpt(self):
        ckpt_dir = os.path.join(self.args.experiment_dir, self.args.experiment)
        if self.args.load_checkpoint not in ["", None]:
            ckpt_dir = os.path.join(self.args.experiment_dir, self.args.load_checkpoint)
        ckpt_path = self.fetch_latest_ckpt(ckpt_dir)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"[Trainer] No checkpoint found form {ckpt_dir}, starting from scratch ...")
            return
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
        try:
            self.net.load_state_dict(ckpt["net_params"])
        except RuntimeError:
            self.net.load_state_dict(ckpt["net_params"], strict=False)
            print("\n [Trainer] WARNING!!! Loaded weights with strict=False")
        if not self.args.reset_hparams:
            if "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except ValueError:
                    print("[Trainer] WARNING!!! Could not load optimizer state dict")
            if "lr_scheduler" in ckpt and ckpt["lr_scheduler"] is not None:
                self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if "last_epoch" in ckpt:
                self.last_epoch = ckpt["last_epoch"]
            if "lowest_loss" in ckpt:
                self.lowest_loss = ckpt["lowest_loss"]
            if "highest_succ" in ckpt:
                self.highest_succ = ckpt["highest_succ"]
            if "n_iter" in ckpt:
                self.n_iter = ckpt["n_iter"]
            print(f"[Trainer] Loaded from weights from {os.path.basename(ckpt_path)}")
        else:
            print(f"[Trainer] Loaded from weights from {os.path.basename(ckpt_path)}")
            print("[Trainer] Hyper-parameters are RESET")
        if self.gpu_count > 1:
            raise NotImplementedError
        print("\n")

    def save_ckpt(self, eval_values, mode, print_on_save=False):
        loss = eval_values.get("loss", None)
        succ = eval_values.get("succ", None)
        assert mode in ["val", "succ"]
        if self.gpu_count > 1:
            net_params = self.net.module.state_dict()
        else:
            net_params = self.net.state_dict()

        state = {
            "last_epoch": self.current_epoch,
            "n_iter": self.n_iter,
            "net_params": net_params,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "lowest_loss": self.lowest_loss,
            "highest_succ": self.highest_succ,
        }
        folder_path = os.path.join(self.args.experiment_dir, self.args.experiment)
        os.makedirs(folder_path, exist_ok=True)

        if mode == "val":
            assert loss is not None
            ckpt_fname = f"epoch_{self.current_epoch}_{mode}-loss={loss:.3f}_ckpt.tar"
            ckpt_path = os.path.join(folder_path, ckpt_fname)
            torch.save(state, ckpt_path)
            if loss < self.lowest_loss:
                best_ckpt_fname = "best.ckpt.tar"
                best_file_path = os.path.join(folder_path, best_ckpt_fname)
                shutil.copyfile(ckpt_path, best_file_path)
                if print_on_save:
                    delta = self.lowest_loss - loss
                    tqdm.write(f"=> Best Checkpoint Updated, {delta:.4f} Mean Loss Reduction")
                self.lowest_loss = loss
        elif mode == "succ":
            assert succ is not None
            ckpt_fname = f"epoch_{self.current_epoch}_{mode}-rate={succ:.3f}_ckpt.tar"
            ckpt_path = os.path.join(folder_path, ckpt_fname)
            torch.save(state, ckpt_path)
            if succ > self.highest_succ:
                best_ckpt_fname = "best.ckpt.tar"
                best_file_path = os.path.join(folder_path, best_ckpt_fname)
                shutil.copyfile(ckpt_path, best_file_path)
                if print_on_save:
                    delta = succ - self.highest_succ
                    tqdm.write(f"=> Best Checkpoint Updated, {delta:.4f} Successrate Increase")
                self.highest_succ = succ
        if print_on_save:
            tqdm.write(f"=> Checkpoint Saved to '{ckpt_path}'")

        exp_dir = os.path.join(self.args.experiment_dir, self.args.experiment)
        self.cleanup_saved_ckpts(exp_dir, mode)


def main():
    parser = argparse.ArgumentParser(description="Palm Training")
    parser.add_argument("-c", "--config", type=str, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--val_only", action="store_true", help="Validation only")

    args = parser.parse_args()

    cfg = parse_network_configs(args.config, print_config=True)

    seed = cfg.train.get("seed")
    if seed is None:
        print("[Trainer] Seed not specified, setting to 0")
        seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    trainer = Trainer(
        args=cfg,
        Net=BCMLP,
        Dataset=PalmDataset,
    )

    if args.debug:
        print("[Trainer] Running debug ...")
        cfg.train.num_epochs = 1
        trainer.train()
        trainer.validate()
        trainer.test()
        exit("Debug done")

    if args.val_only:
        print("[Trainer] Running Validation Only ...")
        trainer.validate()
        exit("Validation Done")
    else:
        print("[Trainer] Running Training   ...")
        trainer.train()


if __name__ == "__main__":
    main()
