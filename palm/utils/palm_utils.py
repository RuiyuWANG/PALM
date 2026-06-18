from collections import defaultdict
from typing import Union

import numpy as np
import torch

import palm.utils.image_utils as ImgUtils
import palm.utils.transform_utils as TUtils
from palm import LABEL_OPS, LOW_DIM_OPS, RGB_OPS
from palm.utils.transform_utils import check_SE3

WORKSPACE_X = np.eye(4)

JOINT_DOF_BY_ROBOT = {
    "franka": 7,
    "franka_panda": 7,
    "panda": 7,
    "panda_tip_only": 7,
    "sawyer": 7,
    "iiwa": 7,
    "jaco": 6,
    "mico": 6,
    "ur5": 6,
    "ur5_tip_only": 6,
    "ur10": 6,
    "xarm6": 6,
    "xarm7": 7,
}

# ------------------------------ Generate Labels ----------------------------- #

def get_labels(
    eef_poses: np.ndarray,
    eef_correction: np.ndarray,
    gripper_state: np.ndarray,
    overwrite_max_n_values: int=0,
) -> dict:
    """
    Generate labels for training by computing various transformations of the end-effector poses and gripper states.

    Parameters:
    - eef_poses: np.ndarray, end-effector poses.
    - gripper_state: np.ndarray, gripper states.

    Returns:
    - dict, labels for training.
    """
    check_SE3(eef_poses)
    eef_poses = eef_poses @ eef_correction

    B = eef_poses.shape[0]
    assert B > 1, "Batch size must be greater than 1 for label generation."
    labels = {}

    def shift_by_n_and_pad(n, arr):
        shifted = arr[n:]
        pad_last = np.tile(arr[-1], (n, 1, 1))
        shifted = np.concatenate([shifted, pad_last], axis=0)
        return shifted

    # delta pose
    eef_poses_t_shifted = shift_by_n_and_pad(1, eef_poses)
    delta_eef_poses = np.linalg.inv(eef_poses) @ eef_poses_t_shifted

    # handle subtask transition
    if overwrite_max_n_values > 0:
        assert overwrite_max_n_values < delta_eef_poses.shape[0]
        pose_norms = np.linalg.norm(delta_eef_poses[:, :3, 3], axis=1)
        pose_indices = np.argsort(pose_norms)[-overwrite_max_n_values:]
        for idx in pose_indices:
            delta_eef_poses[idx] = np.eye(4)
            
    workspace_x = np.array(WORKSPACE_X)
    actions = {
        "delta": delta_eef_poses,
        "relative": np.linalg.inv(eef_poses[0]) @ eef_poses_t_shifted,
        "abs": np.linalg.inv(workspace_x) @ eef_poses_t_shifted,
    }

    for key, act in actions.items():
        labels[get_new_obs_key(key, "6d_xyz")] = TUtils.SE3_to_6D_xyz(act)
        rotvec_xyz = TUtils.SE3_to_rotvec_xyz(act)
        labels[get_new_obs_key(key, "rotvec_xyz")] = rotvec_xyz

    labels["gripper_action"] = get_gripper_actions(gripper_state)
    return labels


def format_action_prediction(
    act: Union[np.ndarray, torch.Tensor], action_keys: tuple, eef_correction_mat: np.ndarray
) -> dict:
    """
    Convert predicted actions into the correct format, i.e. eef actions to the respective SE3 format.

    Parameters:
    - act: Union[np.ndarray, torch.Tensor], the predicted actions.
    - action_keys: tuple, keys representing the action types.

    Returns:
    - dict, the converted actions, with the respective action types and values.
    """
    assert len(action_keys) > 0, "No action keys provided."

    act = act.cpu().numpy() if isinstance(act, torch.Tensor) else act
    act = act.reshape(1, -1) if len(act.shape) == 1 else act

    assert len(action_keys) == 2, "shoule be in the form of (eef, gripper)"

    def to_action(key):
        if "6d_xyz" in key:
            return TUtils.SE3_from_6D_xyz
        elif "rotvec_xyz" in key and "scaled" not in key:
            return TUtils.SE3_from_rotvec_xyz
        elif "gripper" in key:
            return lambda x: x
        raise ValueError("Invalid action key")

    res = {}
    act_dims = [rep_to_shape(key) for key in action_keys]
    action_slices = np.cumsum(act_dims)
    action_slices = np.concatenate([[0], action_slices])
    workspace_x = np.array(WORKSPACE_X)
    for i, key in enumerate(action_keys):
        start, end = action_slices[i], action_slices[i + 1]
        action = act[:, start:end]
        act_type = key.split("_")[0]
        assert act_type in ["delta", "relative", "abs", "gripper"], "Invalid action type"
        if "gripper" in key:
            res["gripper"] = {"action": action}
        else:
            action_mat = to_action(key)(action)
            X_inv = np.linalg.inv(eef_correction_mat)
            X = eef_correction_mat
            if act_type != "abs":
                action_mat = X @ action_mat @ X_inv
            else:
                action_mat = workspace_x @ action_mat @ X
            res["eef"] = {"type": act_type, "action": action_mat}

    return res


# ------------------------- Representation Conversion ------------------------ #

def get_gripper_open(gripper_state: np.ndarray, open_threshold=0.95) -> dict:
    """
    Get gripper representations.
    """
    if isinstance(gripper_state, list):
        gripper_state = np.array(gripper_state)
    assert gripper_state.shape[-1] == 2, "Gripper state must have 2 dimensions."
    has_batch_dim = len(gripper_state.shape) == 2
    gripper_state = gripper_state.reshape(-1, 2)
    gripper_open_ratio = np.sum(gripper_state, axis=1) / 2
    gripper_open = (gripper_open_ratio > open_threshold).astype(int)
    return gripper_open if has_batch_dim else gripper_open[0]


def get_gripper_actions(gripper_state: np.ndarray, delta=0.02) -> dict:
    """
    Get gripper representations.
    """
    if isinstance(gripper_state, list):
        gripper_state = np.array(gripper_state)
    assert gripper_state.ndim == 2, "Gripper state must have 2 dimensions."
    assert gripper_state.shape[-1] == 2, "Gripper state must have 2 dimensions."
    gripper_state = gripper_state.reshape(-1, 2)
    gripper_open_ratio = np.sum(gripper_state, axis=1) / 2
    #
    num_states = gripper_state.shape[0]
    gripper_actions = np.zeros((num_states, 1))
    # HARDCODE: threshold for gripper open/close
    prev_state = int(gripper_open_ratio[0] > 0.9)
    for i in range(num_states - 1):
        delta_state = gripper_open_ratio[i + 1] - gripper_open_ratio[i]
        if delta_state > delta:
            gripper_actions[i] = 1
            prev_state = 1
        elif delta_state < -delta:
            gripper_actions[i] = 0
            prev_state = 0
        else:
            gripper_actions[i] = prev_state
    gripper_actions[-1] = gripper_actions[-2]
    return gripper_actions


def get_eef_pose_reps(
    eef_poses: np.ndarray,
    eef_correction: np.ndarray,
    X_C: np.ndarray,
    ops: list = None,
    all: bool = False,
) -> dict:
    """
    Get end-effector pose representations.
    """
    if (all and ops is not None) or (not all and ops is None):
        raise ValueError("Either 'all' must be True or 'ops' must be provided, but not both.")

    check_SE3(eef_poses)

    eef_poses = eef_poses @ eef_correction

    has_batch_dim = len(eef_poses.shape) == 3
    eef_poses = eef_poses.reshape(-1, 4, 4)
    B = eef_poses.shape[0]
    eef_reps = {}

    if X_C is not None:
        check_SE3(X_C)
        X_C = X_C.reshape(-1, 4, 4)
        if X_C.shape[0] == 1:
            X_C = X_C.repeat(B, axis=0)
        assert X_C.shape[0] == B, "Number of camera poses must match number of eef poses."

        C_X = np.linalg.inv(X_C)
        C_X_H = C_X @ eef_poses
        X_Hybrid = C_X_H.copy()
        X_Hybrid[:, :3, 3] = eef_poses[:, :3, 3]

        cam_ops = {
            "6d_in_cam_xyz": lambda: TUtils.SE3_to_6D_xyz(X_Hybrid),
            "6d_in_cam_z": lambda: TUtils.SE3_to_6D_z(X_Hybrid),
        }

        for key, func in cam_ops.items():
            if all or (ops and key in ops):
                eef_reps[get_new_obs_key("eef", key)] = func()

    general_ops = {
        "9d_xyz": lambda: TUtils.SE3_to_9D_xyz(eef_poses),
        "9d_z": lambda: TUtils.SE3_to_9D_z(eef_poses),
        "6d_xyz": lambda: TUtils.SE3_to_6D_xyz(eef_poses),
        "6d_z": lambda: TUtils.SE3_to_6D_z(eef_poses),
        "z_only": lambda: eef_poses[:, 2, 3].reshape(-1, 1),
    }

    for key, func in general_ops.items():
        if all or (ops and key in ops):
            eef_reps[get_new_obs_key("eef", key)] = func()

    for key in eef_reps:
        eef_reps[key] = eef_reps[key] if has_batch_dim else eef_reps[key][0]
        eef_reps[key] = eef_reps[key].astype(np.float32)
    return eef_reps


def get_image_reps(
    images: np.ndarray,
    img_key: str,
    ops: list,
    img_size: Union[tuple, list],
    debug=False,
    **kwargs,
) -> dict:
    """
    Get image representations based on the representation keys.

    Parameters:
    - images: np.ndarray, input images with at least 3 dimensions.
    - img_key: str, base key for the image representations.
    - ops: list, operations to perform on the images.
    - img_size: tuple, size to resize the images to (default is (84, 84)).
    - debug: bool, whether to enable debug mode.
    - kwargs: additional keyword arguments required for certain operations.

    Returns:
    - dict: dictionary of image representations.
    """
    assert len(images.shape) >= 3, "Images must have at least 3 dimensions."
    all_ops_valid = all([op in RGB_OPS + LOW_DIM_OPS for op in ops])
    assert all_ops_valid, "All operations must be supported by the listed ops."
    
    def check_kwargs(op, kwargs):
        base_keys = ["K", "X_C"]
        tcp_keys = ["eef_poses"]
        op_required_keys = {
            "tcp_crop": ["tcp_crop_size"] + base_keys + tcp_keys,
            "overlay_tcp_crop": ["tcp_crop_size"] + base_keys + tcp_keys,
        }

        if op not in op_required_keys:
            raise ValueError(f"Unknown operation: {op}")

        missing_keys = [key for key in op_required_keys[op] if key not in kwargs]
        if missing_keys:
            raise ValueError(f"Missing required keys for '{op}': {missing_keys}")

    res = {}

    if "front" in img_key:
        for op in ops:
            if "overlay" in op:
                check_kwargs(op, kwargs)
                # generate full overlay images, crop latter if needed
                overlay_key = get_new_obs_key(img_key, "overlay")
                eef_correction = kwargs.get("eef_correction", np.eye(4))
                if overlay_key not in res:
                    res[overlay_key] = ImgUtils.overlay_poses(
                        images,
                        kwargs["eef_poses"] @ eef_correction,
                        kwargs["K"],
                        kwargs["X_C"],
                        im_prefix=img_key + "_overlay",
                        jitter_crop=kwargs["jitter_crop"],
                        debug=debug,
                    )

            if "crop" in op:
                check_kwargs(op, kwargs)
                eef_correction = kwargs.get("eef_correction", np.eye(4))

                is_tcp = "tcp" in op
                is_overlay = "overlay" in op
    
                poses = kwargs["eef_poses"] @ eef_correction if is_tcp else kwargs["target_poses"]
                coords = (
                    ImgUtils.project_points(
                        K=kwargs["K"],
                        X_C=kwargs["X_C"],
                        poses=poses,
                    ) if is_tcp else None
                )

                _images = res[get_new_obs_key(img_key, "overlay")] if is_overlay else images
                crop_suffix = "_overlay" if is_overlay else ""
                if "center" in op:
                    crop_suffix += "_center"
                elif is_tcp:
                    crop_suffix += "_tcp"

                center_crop = kwargs.get("center_crop", False)
                if center_crop:
                    coords = None
                res[get_new_obs_key(img_key, op)] = ImgUtils.crop_at_coords(
                    _images,
                    coords,
                    crop_size=kwargs["tcp_crop_size"],
                    jitter_crop=kwargs["jitter_crop"],
                    img_prefix=img_key + crop_suffix,
                    debug=debug,
                    allow_out_of_bounds=True,
                    height_offset=kwargs.get("tcp_height_offset", 0),
                )

    # add original images
    res[img_key] = images
    # Resize all images
    for key, img in res.items():
        res[key] = ImgUtils.resize_images(img, img_size)
    
    return res

# ------------------------- Name and Shape Helper ------------------------ #

def get_new_obs_key(key, op):
    """
    Generate a new observation key based on the operation and robot name.
    """
    all_ops = RGB_OPS + LOW_DIM_OPS + LABEL_OPS
    assert op in all_ops, f"Operation {op} not supported."
    if op in key or "crop" in key or "resize" in key:
        return key
    return f"{key}_{op}"


def get_ops_from_obs_keys(obs_keys):
    """
    Get the operation from the observation keys.
    """
    all_ops = RGB_OPS + LOW_DIM_OPS + LABEL_OPS
    # sort by length to avoid partial matches
    all_ops = sorted(all_ops, key=lambda x: len(x), reverse=True)
    ops = defaultdict(list)
    for key in obs_keys:
        key_has_op = False
        for op in all_ops:
            if op in key:
                mod = key.replace(op, "")
                mod = mod.rstrip("_")
                ops[mod].append(op)
                key_has_op = True
                break
        if not key_has_op:
            ops[key] = []
    return ops


def _joint_dof_from_key(state_key: str, key_chunks: list) -> int:
    for chunk in key_chunks:
        if chunk.endswith("dof") and chunk[:-3].isdigit():
            return int(chunk[:-3])
        if chunk.startswith("dof") and chunk[3:].isdigit():
            return int(chunk[3:])

    padded_key = f"_{state_key.lower()}_"
    for robot_name, dof in JOINT_DOF_BY_ROBOT.items():
        if f"_{robot_name}_" in padded_key:
            return dof

    return 7


def rep_to_shape(state_key: str) -> int:
    """
    Get the shape of the representation based on the key.
    """
    low_dim_shape = 0
    if "gripper" in state_key:
        low_dim_shape += 1
        return low_dim_shape
    key_chunks = state_key.split("_")
    if "6d" in key_chunks:
        low_dim_shape += 6
    elif "joint" in key_chunks:
        low_dim_shape += _joint_dof_from_key(state_key, key_chunks)
    elif "3d" in key_chunks:
        low_dim_shape += 3
    elif "9d" in key_chunks:
        low_dim_shape += 9
    elif "rotvec" in key_chunks:
        low_dim_shape += 3
    if "xyz" in key_chunks:
        low_dim_shape += 3
    elif "z" in key_chunks:
        low_dim_shape += 1
    elif "vel" in key_chunks:
        low_dim_shape += 6
    return low_dim_shape


def reps_to_shapes(reps: list) -> list:
    """
    Get the shapes of the representations based on the keys.
    """
    return [rep_to_shape(rep) for rep in reps]