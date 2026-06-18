import os
import pickle
import re
from PIL import Image
import numpy as np
from collections import defaultdict
from typing import Dict, List

import torch
from torchvision import transforms as T
import scipy.spatial.transform.rotation as R

from pyrep.const import RenderMode
from rlbench import ObservationConfig

import palm.utils.palm_utils as PalmUtils
from palm.models.bc_mlp import BCMLP
from palm.utils.config_utils import load_config_to_namespace
from palm.utils.net_utils import parse_network_configs

MISC_SUFFIXES = ["extrinsics", "intrinsics", "near", "far"]
LOW_DIM_FILE = "low_dim_obs.pkl"

class RLBenchDataDirParser:
    def __init__(
        self,
        dataset_dir: str,
        rgb_keys: List[str],
        mask_keys: List[str],
        low_dim_keys: List[str],
    ):
        self.dataset_dir = dataset_dir
        self.eps_dirs = self.parse_data_path()
        self.rgb_keys = rgb_keys
        self.mask_keys = mask_keys
        self.low_dim_keys = low_dim_keys
        self.count = 0

    def __len__(self):
        return len(self.eps_dirs)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.eps_dirs):
            result = self.load_one_episode(self.eps_dirs[self.count])
            result["episode_id"] = os.path.basename(self.eps_dirs[self.count])
            self.count += 1
            return result
        else:
            raise StopIteration

    def parse_data_path(self) -> List[str]:
        eps_path = os.path.join(self.dataset_dir, "episodes")
        if not os.path.exists(eps_path):
            raise FileNotFoundError(f"Episodes directory does not exist: {eps_path}")

        eps = sorted(os.listdir(eps_path), key=_numerical_sort)
        if not eps:
            raise FileNotFoundError("No episodes found in the directory.")

        eps_dirs = [
            os.path.join(eps_path, ep)
            for ep in eps
            if os.path.isdir(os.path.join(eps_path, ep))
        ]

        return eps_dirs

    def load_one_episode(self, episode_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
        obs = {"rgb": {}, "mask": {}, "low_dim": {}}

        for key in self.rgb_keys:
            rgb_dir = os.path.join(episode_dir, key)
            if not os.path.exists(rgb_dir):
                raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")
            obs["rgb"][key] = load_images(rgb_dir)

        for key in self.mask_keys:
            mask_dir = os.path.join(episode_dir, key)
            if not os.path.exists(mask_dir):
                raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
            obs["mask"][key] = load_images(mask_dir)

        low_dim_file = os.path.join(episode_dir, LOW_DIM_FILE)
        if not os.path.exists(low_dim_file):
            raise FileNotFoundError(f"Low dim file does not exist: {low_dim_file}")
        obs["low_dim"] = load_low_dim_obs(low_dim_file, self.low_dim_keys)

        return obs


def load_low_dim_obs(low_dim_pkl_path, obs_keys):
    """
    Load low_dim observation data from a pickle file and return a list of dictionaries containing the selected keys.
    """

    if not os.path.exists(low_dim_pkl_path):
        raise FileNotFoundError(f"Data path does not exist: {low_dim_pkl_path}")
    if not low_dim_pkl_path.endswith(".pkl"):
        raise ValueError("Data path must be a pickle file")

    with open(low_dim_pkl_path, "rb") as file:
        low_dim = pickle.load(file)

    if not low_dim:
        raise ValueError("Loaded low_dim data is empty")

    _low_dim = low_dim[0]

    misc_keys = [
        key for key in obs_keys if any(suffix in key for suffix in MISC_SUFFIXES)
    ]
    missing_misc_keys = [key for key in misc_keys if key not in _low_dim.misc]
    if missing_misc_keys:
        raise KeyError(f"Keys not found in misc: {missing_misc_keys}")

    obs_keys = [key for key in obs_keys if key not in misc_keys]
    missing_keys = [key for key in obs_keys if not hasattr(_low_dim, key)]
    if missing_keys:
        raise KeyError(f"Keys not found in low_dim: {missing_keys}")

    selected_obs = {key: [] for key in obs_keys + misc_keys}

    for obs in low_dim:
        for key in obs_keys:
            selected_obs[key].append(getattr(obs, key))
        for key in misc_keys:
            selected_obs[key].append(obs.misc[key])

    # merge lists into numpy arrays
    for key in selected_obs:
        selected_obs[key] = np.array(selected_obs[key])

    return selected_obs


def load_images(image_dir: str, sort: bool = True):
    """
    Read images from a directory to a batch of images.
    Note that all in range [0, 255].
    """

    # find all images files in the directory
    im_files = (
        sorted(os.listdir(image_dir), key=_numerical_sort)
        if sort
        else os.listdir(image_dir)
    )

    images = []
    for f in im_files:
        if f.endswith(".png") or f.endswith(".jpg"):
            img = Image.open(os.path.join(image_dir, f))
            images.append(np.array(img))
    # check if any images were found
    assert images, "No images found in the directory."
    # check if all images have the same shape
    shape = images[0].shape
    assert all(img.shape == shape for img in images), "Images have different shapes."
    return np.stack(images, axis=0)


class RLBenchAgent:
    def __init__(self, agent=None, args=None, model_dir=None):
        if model_dir is not None:
            self.agent, args = self.model_from_ckpt_path(model_dir)
        else:
            assert (agent is not None) and (args is not None)
            self.agent = agent

        self.args = args.network
        self.input_size = self.args.crop_randomizer.input_size
        self.transform = T.Compose([T.ToTensor()])

        try:
            dataset_path = args.dataset.dataset_path
            dataset_dir = os.path.dirname(dataset_path)
            json_file_paths = [
                f for f in os.listdir(dataset_dir) if f.endswith("config.json")
            ]
            assert len(json_file_paths) == 1, (
                f"Multiple JSON files found in {dataset_dir}"
            )
            dataset_cfg = load_config_to_namespace(
                os.path.join(dataset_dir, json_file_paths[0]), full_path=True
            )
            self.conversion_args = dataset_cfg.conversion
        except Exception as e:
            raise ValueError(f"Error loading dataset config: {e}")

        self.eef_correction_mat = self.get_eef_correction_matrix()

    def model_from_ckpt_path(self, ckpt_path):
        dataset_config_dir = os.path.dirname(ckpt_path)
        dataset_config_path = os.path.join(dataset_config_dir, "config.json")
        cfg = parse_network_configs(dataset_config_path)

        model = BCMLP(cfg)

        assert "*tar" not in ckpt_path, (
            "Please provide the path to the model checkpoint"
        )
        model.load_state_dict(torch.load(ckpt_path)["net_params"], strict=True)

        assert torch.cuda.is_available(), "CUDA device not found"
        model = model.cuda()
        model.eval()
        return model, cfg

    def get_eef_correction_matrix(self):
        correction_mat = np.eye(4)
        if getattr(self.conversion_args, "eef_correction", None) is not None:
            try:
                rot_axis, rot_angle = self.conversion_args.eef_correction
                assert rot_axis in ["x", "y", "z"], "Rotation axis must be x, y, or z"
                assert isinstance(rot_angle, (int, float)), (
                    "Rotation angle must be a number"
                )
                rot_angle = np.deg2rad(rot_angle)
                rot_vec = np.eye(3)[["x", "y", "z"].index(rot_axis)]
                correction_mat[:3, :3] = R.Rotation.from_rotvec(
                    rot_angle * rot_vec
                ).as_matrix()
            except ValueError:
                correction_mat[:3, :3] = self.conversion_args.eef_correction
            print("Rotation matrix:\n", correction_mat[:3, :3])
        return correction_mat

    def get_rgb_obs(self, obs) -> Dict[str, np.ndarray]:
        rgb_obs_keys = self.args.obs.rgb
        rgb_ops = PalmUtils.get_ops_from_obs_keys(rgb_obs_keys)
        processed_obs = {}

        X_H = obs.gripper_matrix
        X_C = obs.misc["front_camera_extrinsics"]
        K = obs.misc["front_camera_intrinsics"]

        crop_kwargs = {
            "eef_poses": X_H,
            "eef_correction": self.eef_correction_mat,
            "X_C": X_C,
            "K": K,
            "tcp_crop_size": getattr(self.conversion_args, "tcp_crop_size", None),
            "jitter_crop": getattr(self.conversion_args, "jitter_crop", 0),
            "tcp_height_offset": getattr(
                self.conversion_args, "tcp_height_offset", None
            ),
            "center_crop": getattr(self.conversion_args, "center_crop", False),
        }

        for cam_mod in rgb_ops.keys():
            rgb = getattr(obs, cam_mod)
            if rgb is None:
                raise KeyError(f"Key not found: {cam_mod}")
            img_reps = PalmUtils.get_image_reps(
                rgb,
                cam_mod,
                ops=rgb_ops[cam_mod],
                img_size=self.input_size,
                **crop_kwargs,
            )
            processed_obs.update(img_reps)
        # Create a list of keys to remove
        keys_to_remove = [k for k in processed_obs if k not in rgb_obs_keys]
        for k in keys_to_remove:
            processed_obs.pop(k)
        return processed_obs

    def get_low_dim_obs(self, obs) -> Dict[str, np.ndarray]:
        eef_poses = getattr(obs, "gripper_matrix")
        X_C = obs.misc["front_camera_extrinsics"]
        low_dim_ops = PalmUtils.get_ops_from_obs_keys(self.args.obs.low_dim)
        eef_ops = low_dim_ops["eef"]
        low_dim_obs = PalmUtils.get_eef_pose_reps(
            eef_poses=eef_poses,
            eef_correction=self.eef_correction_mat,
            X_C=X_C,
            ops=eef_ops,
        )
        for key in self.args.obs.low_dim:
            if "gripper" in key:
                if key == "gripper_open_binary":
                    gripper_state = PalmUtils.get_gripper_open(
                        getattr(obs, "gripper_open")
                    )
                else:
                    gripper_state = getattr(obs, key)
                    if gripper_state is None:
                        raise KeyError(f"Key not found: {key}")
                if gripper_state.ndim == 0:
                    gripper_state = np.array([gripper_state])
                low_dim_obs[key] = gripper_state
        return low_dim_obs

    def obs_to_torch(self, obs) -> Dict[str, np.ndarray]:
        ret = {"obs": defaultdict(dict)}

        ret["obs"]["rgb"] = self.get_rgb_obs(obs)
        self.last_processed_rgb = {
            key: value.copy() for key, value in ret["obs"]["rgb"].items()
        }
        ret["obs"]["low_dim"] = self.get_low_dim_obs(obs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for key in ret["obs"]["rgb"]:
            im_pil = Image.fromarray(ret["obs"]["rgb"][key])
            ret["obs"]["rgb"][key] = self.transform(im_pil).to(device).unsqueeze(0)

        for key in ret["obs"]["low_dim"]:
            ret["obs"]["low_dim"][key] = torch.tensor(
                ret["obs"]["low_dim"][key], device=device
            ).unsqueeze(0)
        return ret

    def predict_action(self, obs) -> torch.Tensor:
        action_keys = self.args.actions
        with torch.no_grad():
            batch = self.obs_to_torch(obs)
            act_pred = self.agent(batch)
            act_pred_denorm = self.agent.obs_normalizer.denormalize_actions(
                act_pred, keys=action_keys
            )
        return PalmUtils.format_action_prediction(
            act_pred_denorm[0], action_keys, eef_correction_mat=self.eef_correction_mat
        )

def _numerical_sort(filename):
    match = re.search(r"\d+", filename)  # Find the first occurrence of digits
    return int(match.group()) if match else float("inf")

def get_default_camera_config(img_size=(512, 512), renderer="opengl"):
    img_size = (512, 512)
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.front_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size

    obs_config.front_camera.as_rgb_and_mask()
    obs_config.wrist_camera.as_rgb_and_mask()

    # We don't need the other cameras
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    obs_config.record_gripper_closing = True

    if renderer == "opengl":
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL
    elif renderer == "opengl3":
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL3
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL3
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL3
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL3
        obs_config.front_camera.render_mode = RenderMode.OPENGL3

    return obs_config


def list_to_noise(noise_xyz, mode="normal"):
    assert len(noise_xyz) == 3
    if isinstance(noise_xyz, list):
        noise_xyz = np.array(noise_xyz)
    if mode == "uniform":
        return np.random.uniform(-noise_xyz, noise_xyz)
    elif mode == "normal":
        stds = noise_xyz / 2.0
        normal_noise = np.random.normal(loc=0, scale=stds)
        return np.clip(normal_noise, -noise_xyz, noise_xyz)
    else:
        raise ValueError("Invalid noise mode %s" % mode)
