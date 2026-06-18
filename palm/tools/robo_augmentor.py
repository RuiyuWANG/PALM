import random
import textwrap
import numpy as np

from torch import nn
import torch
from kornia.augmentation import RandomErasing, RandomPerspective, RandomResizedCrop


class RoboAugmentor(nn.Module):
    """
    Given a batch of data, applies data augmentation techniques to the batch.
    """

    def __init__(self, config):
        super(RoboAugmentor, self).__init__()
        self.p = config.prob
        self.enable_perspective = getattr(config, "perspective", False)
        self.enable_zoom = getattr(config, "zoom", False)
        self.enable_erase = getattr(config, "erase", False)

        self.debug = config.debug
        self.debug_dir = config.debug_dir
        self.warmup = config.warmup

        self.E = torch.tensor(np.diag([-1, 1, 1, 1]), dtype=torch.float32).unsqueeze(0)
        self.Rz = torch.tensor(np.diag([-1, -1, 1, 1]), dtype=torch.float32).unsqueeze(
            0
        )
        self.perspective_transform = RandomPerspective(0.2, p=self.p)
        self.zoom_transform = RandomResizedCrop(
            size=(config.input_size[0], config.input_size[1]),
            scale=(0.9, 1.1),
            ratio=(1, 1),
            p=self.p,
            keepdim=True,
        )
        self.erase_transform = RandomErasing(
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value=0.0,
            same_on_batch=False,
            p=self.p,
            keepdim=True,
        )

    def forward(self, batch, epoch_idx=None):
        """
        Apply data augmentation techniques to the batch.

        Args:
            batch (dict): A dictionary containing the batch of data.

        Returns:
            None: The function modifies the input batch in place.
        """

        if not self.training:
            return
        if epoch_idx is not None and epoch_idx < self.warmup:
            return

        augmentations = []
        if self.enable_zoom:
            augmentations.append(self.random_zoom)
        if self.enable_perspective:
            augmentations.append(self.random_perspective)
        if self.enable_erase:
            augmentations.append(self.random_erase)
        if len(augmentations) > 0:
            random.choice(augmentations)(batch)

    def __repr__(self):
        if (
            not self.enable_flipping
            and not self.enable_mirroring
            and not self.enable_perspective
        ):
            return "RoboAugmentor: Disabled\n"
        print_str = "RoboAugmentor:\n"
        if self.enable_perspective:
            print_str += f"  Perspective (p={self.p}): randomly apply perspective transformation to images.\n"
        textwrapper = textwrap.TextWrapper(
            width=80, subsequent_indent=" " * 4, replace_whitespace=False
        )
        return textwrapper.fill(print_str) + "\n"

    def random_perspective(self, batch):
        if not self.enable_perspective:
            return
        for key in self.__get_img_keys(batch):
            batch["obs"]["rgb"][key] = self.perspective_transform(
                batch["obs"]["rgb"][key]
            )

    def random_zoom(self, batch):
        if not self.enable_zoom:
            return
        for key in self.__get_img_keys(batch):
            batch["obs"]["rgb"][key] = self.zoom_transform(batch["obs"]["rgb"][key])

    def random_erase(self, batch):
        if not self.enable_erase:
            return
        for key in self.__get_img_keys(batch):
            batch["obs"]["rgb"][key] = self.erase_transform(batch["obs"]["rgb"][key])

    def __get_img_keys(self, batch):
        img_keys = list(batch["obs"]["rgb"].keys())
        if self.debug:
            print(f"Image keys: {img_keys}")
        return img_keys