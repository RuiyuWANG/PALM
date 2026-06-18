import os
import h5py
from tqdm import tqdm
import numpy as np
from typing import Callable

from PIL import Image
import kornia.augmentation as K

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

from palm.utils.net_utils import reps_to_shapes


class RandomOverlay(nn.Module):
    def __init__(self, args):
        super(RandomOverlay, self).__init__()
        self.enabled = args.get("enabled", False)
        self.p = getattr(args, "p", 0.5)
        self.blend_alpha = getattr(args, "blend_alpha", 0.5)
        self.augment_ood = getattr(args, "augment_ood", False)
        self.ood_transform = K.AugmentationSequential(
            K.RandomAffine(degrees=30, scale=(0.8, 1.2), shear=(-10, 10)),
            K.RandomBrightness(0.5, 1.5),
        )
        self.warmup = getattr(args, "warmup", None)
        self.input_size = args.get("input_size")
        self.backgrounds = None
        self.printed_msg = False
        self.args = args

    def load_all_backgrounds(self, background_dir):
        assert os.path.exists(background_dir), (
            f"Background directory {background_dir} does not exist"
        )
        background_ids = os.listdir(background_dir)
        backgrounds = []
        for bg_id in tqdm(background_ids, desc="[RandomOverlay] Loading backgrounds"):
            bg_path = os.path.join(background_dir, bg_id)
            bg = Image.open(bg_path).convert("RGB")
            bg = T.Resize(self.input_size)(bg)
            bg = T.ToTensor()(bg)
            backgrounds.append(bg)
        self.backgrounds = torch.stack(backgrounds).to("cuda")

    def get_random_coco_background(self, num_backgrounds):
        idx = torch.randint(0, len(self.backgrounds), (num_backgrounds,))
        if self.augment_ood:
            return self.ood_transform(self.backgrounds[idx])
        
        return self.backgrounds[idx]

    def forward(self, x, epoch_idx):
        if not self.training or not self.enabled:
            return x
        if self.backgrounds is None:
            self.load_all_backgrounds(self.args.background_dir)

        assert x.dim() == 4, "Input tensor must be 4D"
        if self.warmup is not None:
            assert epoch_idx is not None, "Epoch index must be provided"
            if epoch_idx < self.warmup:
                return x
            elif epoch_idx == self.warmup and not self.printed_msg:
                self.printed_msg = True
                print("[RandomOverlay] Warmup period ended. Starting random overlay")

        B, _, _, _ = x.shape
        num_augs = int(B * self.p)
        ood_x = self.get_random_coco_background(num_augs)
        idx = torch.randperm(B)[:num_augs]
        x[idx] = self.blend_alpha * x[idx] + (1 - self.blend_alpha) * ood_x
        
        return x


class CropRandomizer(nn.Module):
    def __init__(self, args):
        super(CropRandomizer, self).__init__()
        self.input_size = args.get("input_size")
        assert self.input_size is not None
        self.crop_size = args.crop_size
        self.crop_ratio = args.get("crop_ratio", None)
        self.resize = args.get("resize", False)
        self.shuffle_crop_size = args.get("shuffle_crop_size", False)
        self.return_indices = args.get("return_indices", False)

        assert (self.crop_size is not None) ^ (self.crop_ratio is not None)
        if self.crop_size is not None:
            if isinstance(self.crop_size, int) or isinstance(self.crop_size, tuple):
                self.crop_size = (self.crop_size, self.crop_size)
        assert self.crop_size[0] == self.crop_size[1], "Crop size must be square"

    @staticmethod
    def build_grid(source_height, source_width, target_height, target_width, device):
        k_h = float(target_height) / float(source_height)
        k_w = float(target_width) / float(source_width)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-k_h, k_h, target_height, device=device),
            torch.linspace(-k_w, k_w, target_width, device=device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        return grid

    def bboxes_after_cropping(self, bboxes, delta):
        assert len(delta) == 3, "Delta must be a tuple of 2 elements"
        delta_h, delta_w, crop_ratio = delta
        if self.resize:
            bboxes /= crop_ratio
        # convert delta to crop_vision coordinates (0, 0) at top left corner
        delta_h_cv = (1 - crop_ratio + delta_h) / 2
        delta_w_cv = (1 - crop_ratio + delta_w) / 2
        delta_4pts = torch.stack(
            [delta_h_cv, delta_w_cv, delta_h_cv, delta_w_cv], dim=1
        )
        return torch.clamp(bboxes - delta_4pts, 0, 1)

    def forward(self, x):
        assert x.dim() == 4, "Input tensor must be 4D"
        assert list(x.shape[-2:]) == self.input_size, "Input size mismatch"

        B, _, H, W = x.size()
        if self.crop_size is not None:
            crop_ratio = self.crop_size[0] / H
            crop_h, crop_w = self.crop_size
        else:
            crop_ratio = self.crop_ratio
            crop_h, crop_w = int(H * crop_ratio), int(W * crop_ratio)

        max_crop_h_norm, max_crop_w_norm = 1 - crop_ratio, 1 - crop_ratio
        grid = self.build_grid(H, W, crop_h, crop_w, x.device).repeat(B, 1, 1, 1)
        delta_h = torch.zeros((B,), device=x.device)
        delta_w = torch.zeros((B,), device=x.device)
        if self.training:
            delta_h = (2 * torch.rand(B, device=x.device) - 1) * max_crop_h_norm
            delta_w = (2 * torch.rand(B, device=x.device) - 1) * max_crop_w_norm
            grid[:, :, :, 0] += (
                delta_h.unsqueeze(-1).unsqueeze(-1).expand(-1, crop_h, crop_w)
            )
            grid[:, :, :, 1] += (
                delta_w.unsqueeze(-1).unsqueeze(-1).expand(-1, crop_h, crop_w)
            )

        crops = F.grid_sample(x, grid, align_corners=True)
        if self.resize:
            crops = F.interpolate(crops, (H, W), mode="bilinear", align_corners=True)
        if self.return_indices:
            return crops, (delta_h, delta_w, crop_ratio)

        assert crops.shape[-2:] == (crop_h, crop_w), "Cropped image shape mismatch"
        return crops


class ObsNormalizer(nn.Module):
    def __init__(self, args):
        super(ObsNormalizer, self).__init__()

        self.norm_rgb = args.rgb
        if self.norm_rgb:
            assert self.norm_rgb in ["default", "imagenet"]

        supported_low_dim_modes = ["standardize", "normalize"]
        self.norm_low_dim = args.low_dim

        if self.norm_low_dim is not None:
            assert self.norm_low_dim in supported_low_dim_modes

        self.norm_actions = args.actions
        if self.norm_actions is not None:
            assert self.norm_actions in supported_low_dim_modes

        self.low_dim_noise = (
            getattr(args, "low_dim_noise", 0.0) if self.training else 1.0
        )
        self.dataset_path = args.dataset_path

        # TODO: add support for multiple steps ahead
        self.initialize_parameters()

    def forward(self, batch):

        if self.norm_rgb is not None and "rgb" in batch["obs"]:
            data = batch["obs"]["rgb"]
            for key in data.keys():
                data[key] = data[key] * self.rgb_scale + self.rgb_offset

        if self.norm_low_dim is not None and "low_dim" in batch["obs"]:
            data = batch["obs"]["low_dim"]
            for key in data.keys():
                data[key] = (
                    data[key] * self.low_dim_scale[key] + self.low_dim_offset[key]
                )
                data[key] += torch.randn_like(data[key]) * self.low_dim_noise

        if self.norm_actions is not None and "actions" in batch:
            data = batch["actions"]
            for key in data.keys():
                data[key] = data[key] * self.action_scale[key] + self.action_offset[key]
        return batch

    def denormalize_actions(self, action, keys):
        scale = torch.cat([self.action_scale[key] for key in keys], dim=-1)
        offset = torch.cat([self.action_offset[key] for key in keys], dim=-1)
        return (action - offset) / scale

    def denormalize_images(self, image):
        return (image - self.rgb_offset) / self.rgb_scale

    def initialize_parameters(self):
        assert os.path.exists(self.dataset_path), (
            f"Data path {self.dataset_path} does not exist"
        )

        def to_nn_param(param, is_image=False):
            if param is None:
                return None
            if not isinstance(param, torch.Tensor):
                param = torch.tensor(param)
            if is_image:
                param = param.view(1, 3, 1, 1)
            elif len(param.shape) == 1:
                param = param.view(1, -1)
            return nn.Parameter(param.float(), requires_grad=False)

        if self.norm_rgb == "imagenet":
            rgb_mean = torch.tensor([0.485, 0.456, 0.406])
            rgb_std = torch.tensor([0.229, 0.224, 0.225])
        elif self.norm_rgb == "default":
            rgb_mean = torch.tensor([0.5, 0.5, 0.5])
            rgb_std = torch.tensor([0.5, 0.5, 0.5])

        self.rgb_scale = to_nn_param(1 / rgb_std, is_image=True)
        self.rgb_offset = to_nn_param(-rgb_mean / rgb_std, is_image=True)

        self.low_dim_scale = nn.ParameterDict()
        self.low_dim_offset = nn.ParameterDict()
        self.action_scale = nn.ParameterDict()
        self.action_offset = nn.ParameterDict()

        with h5py.File(self.dataset_path, "r") as f:
            for key in f["stats"]["low_dim"].keys():
                base_key = "_".join(key.split("_")[:-1])
                stats = {
                    "input_mean": f["stats"]["low_dim"][f"{base_key}_mean"][()].reshape(
                        -1
                    ),
                    "input_std": f["stats"]["low_dim"][f"{base_key}_std"][()].reshape(
                        -1
                    ),
                    "input_min": f["stats"]["low_dim"][f"{base_key}_min"][()].reshape(
                        -1
                    ),
                    "input_max": f["stats"]["low_dim"][f"{base_key}_max"][()].reshape(
                        -1
                    ),
                }
                scale, offset = self.get_low_dim_scale_and_offset(
                    self.norm_low_dim, **stats
                )
                self.low_dim_scale[base_key] = to_nn_param(scale)
                self.low_dim_offset[base_key] = to_nn_param(offset)

            for key in f["stats"]["actions"].keys():
                base_key = "_".join(key.split("_")[:-1])
                stats = {
                    "input_mean": f["stats"]["actions"][f"{base_key}_mean"][()].reshape(
                        -1
                    ),
                    "input_std": f["stats"]["actions"][f"{base_key}_std"][()].reshape(
                        -1
                    ),
                    "input_min": f["stats"]["actions"][f"{base_key}_min"][()].reshape(
                        -1
                    ),
                    "input_max": f["stats"]["actions"][f"{base_key}_max"][()].reshape(
                        -1
                    ),
                }
                scale, offset = self.get_low_dim_scale_and_offset(
                    self.norm_actions, **stats
                )
                self.action_scale[base_key] = to_nn_param(scale)
                self.action_offset[base_key] = to_nn_param(offset)

    @staticmethod
    def get_low_dim_scale_and_offset(
        mode,
        input_mean,
        input_std,
        input_min,
        input_max,
        out_min=-1,
        out_max=1,
        range_eps=1e-7,
    ):
        if mode is None:
            return None, None
        assert mode in ["standardize", "normalize"]
        if mode == "standardize":
            ignored_idx = input_std < range_eps
            input_std[ignored_idx] = 1
            scale = 1 / input_std
            offset = -input_mean * scale
            offset[ignored_idx] = 0
        else:
            input_range = input_max - input_min
            ignored_idx = input_range < range_eps
            output_range = out_max - out_min
            input_range[ignored_idx] = output_range
            scale = output_range / input_range
            offset = out_min - input_min * scale
            offset[ignored_idx] = (out_min + out_max) / 2 - input_min[ignored_idx]
        return scale, offset


def make_visual_encoder(args):
    net = nn.ModuleList()

    if args.encoder.type == "resnet18":
        weights = ResNet18_Weights.DEFAULT if args.encoder.pretrained else None
        backbone = resnet18(weights=weights)
    else:
        raise NotImplementedError(f"Backbone {args.encoder.type} not implemented")
    if args.encoder.pooling.type == "spatial_softmax":
        input_size = args.crop_randomizer.crop_size[0]
        last_feat_map_dim = torch.ceil(torch.tensor(input_size) / 32).int()
        spatial_softmax = SpatialSoftmax((512, last_feat_map_dim, last_feat_map_dim))
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.append(spatial_softmax)
    elif args.encoder.pooling.type == "avg":
        backbone = nn.Sequential(*list(backbone.children())[:-1])
    else:
        raise NotImplementedError(
            f"Pooling {args.encoder.pooling.type} not implemented"
        )

    net.append(backbone)
    net.append(nn.Flatten())
    net.append(nn.Linear(args.encoder.pooling.in_dim, args.encoder.pooling.out_dim))

    if args.encoder.coord_conv:
        net.insert(0, CoordConv2d())
    return nn.Sequential(*net)


def make_mlp(args):
    mlp = nn.ModuleList()
    num_rgb = len(reps_to_shapes(args.obs.rgb))
    in_dim = args.low_dim_shape + args.encoder.pooling.out_dim * num_rgb
    if args.mlp.activation == "relu":
        activation = nn.ReLU()
    elif args.mlp.activation == "silu":
        activation = nn.SiLU()
    else:
        raise NotImplementedError(f"Activation {args.mlp.activation} not implemented")
    for hidden_dim in args.mlp.hidden_dims:
        mlp.append(nn.Linear(in_dim, hidden_dim))
        mlp.append(activation)
        in_dim = hidden_dim
    mlp.append(nn.Linear(in_dim, args.action_shape))
    return nn.Sequential(*mlp)


def receptivefield_conv(in_channels, out_channels):
    """
    A simple 2-layer convolutional block with ReLU for receptive field expansion
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
    )


class CoordConv2d(nn.Module):
    def __init__(self, in_channels=5, out_channels=3):
        super(CoordConv2d, self).__init__()

        self.coord_conv_reduction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
        )
        self.coord_grid = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def _make_grid(self, b, h, w, device):
        xx_channel = torch.arange(w).repeat(h, 1).float() / (w - 1) * 2 - 1
        yy_channel = torch.arange(h).repeat(w, 1).t().float() / (h - 1) * 2 - 1
        xx_channel = xx_channel.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w).to(device)
        yy_channel = yy_channel.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w).to(device)
        return torch.cat([xx_channel, yy_channel], dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if self.coord_grid is None or self.coord_grid.shape[0] != b:
            self.coord_grid = self._make_grid(b, h, w, x.device)
        x = torch.cat([x, self.coord_grid.expand(b, 2, h, w)], dim=1)
        return self.coord_conv_reduction(x)


class Resnet18Skip(nn.Module):
    def __init__(self, weights=None):
        super(Resnet18Skip, self).__init__()
        # Load pre-trained lyrn_backend
        backbone = resnet18(weights=weights)
        # print(backbone)
        self.res_conv0 = nn.Sequential(*list(backbone.children())[:-6])
        self.res_conv1 = nn.Sequential(*list(backbone.children())[-6:-5])
        self.res_conv2 = nn.Sequential(*list(backbone.children())[-5:-4])
        self.res_conv3 = nn.Sequential(*list(backbone.children())[-4:-3])
        self.res_conv4 = nn.Sequential(*list(backbone.children())[-3:-2])

        self.up_2x = nn.UpsamplingBilinear2d(scale_factor=2)

        self.top_conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.conv_one_eighth = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, 1, 1, bias=False), nn.ReLU()
        )

        self.conv_quater = nn.Sequential(
            nn.Conv2d(256 + 128, 256, 3, 1, 1, bias=False), nn.ReLU()
        )

    def forward(self, img):
        # bottom-up
        back_bone = self.res_conv0(img)
        c2 = self.res_conv1(back_bone)
        c3 = self.res_conv2(c2)
        c4 = self.res_conv3(c3)
        c5 = self.res_conv4(c4)
        fm = self.top_conv(c5)
        fm = self.conv_one_eighth(torch.cat([self.up_2x(fm), c4], 1))
        fm = self.conv_quater(torch.cat([self.up_2x(fm), c3], 1))
        return fm


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
        self,
        input_shape,
        num_kp=32,
        temperature=1.0,
        learnable_temperature=False,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=True
            )
            self.register_parameter("temperature", temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=False
            )
            self.register_buffer("temperature", temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)
        self.kps = None

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert feature.shape[1] == self._in_c
        assert feature.shape[2] == self._in_h
        assert feature.shape[3] == self._in_w
        if self.nets is not None:
            feature = self.nets(feature)
        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp * 2)
        return feature_keypoints


def frequency_encoding(x, L):
    if L == 0:
        return x
    dims = torch.arange(L).unsqueeze(0).to(x.device)
    freqs = 2.0**dims

    sin_encodings = torch.sin(freqs * x.unsqueeze(-1))
    cos_encodings = torch.cos(freqs * x.unsqueeze(-1))
    enc_x = torch.cat([sin_encodings, cos_encodings], dim=-1)
    return enc_x.reshape(*x.shape[:-1], -1)


def init_mlp_weights(net: nn.Module, init_func=nn.init.xavier_uniform_, zero_bias=True):
    """
    Initializes the weights of an MLP.

    Args:
        mlp (torch.nn.Module): The MLP to initialize.
        init_func (callable, optional): The initialization function to use. Defaults to nn.init.xavier_uniform_.
        zero_bias (bool, optional): Whether to zero the bias. Defaults to True.
    """

    for module in net.modules():
        if isinstance(module, nn.Linear):
            # print("Initializing", module)
            init_func(module.weight)
            if zero_bias:
                nn.init.zeros_(module.bias)


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module