import os
from tqdm import tqdm
import numpy as np
import imageio
import torch
from PIL import Image

from palm.utils.config_utils import load_config_to_namespace
from palm.utils.ood_utils import CameraHelper
from palm.utils.rlbench_utils import RLBenchAgent, get_default_camera_config

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import DeltaEndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import DiscreteWithoutChecking
from rlbench.backend.const import *
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from pyrep.errors import IKError


def _save_debug_crops(agent, debug_crop_dir, traj_i, step_i):
    processed_rgb = getattr(agent, "last_processed_rgb", {})
    crop_items = [(key, img) for key, img in processed_rgb.items() if "crop" in key]
    if len(crop_items) == 0:
        return

    os.makedirs(debug_crop_dir, exist_ok=True)
    for key, img in crop_items:
        img_name = f"traj_{traj_i:04d}_step_{step_i:04d}_{key}.png"
        Image.fromarray(img).save(os.path.join(debug_crop_dir, img_name))


def evaluate_policy(agent, task_str, args):
    agent.agent.eval()

    robot_name = getattr(args, "robot", "panda")
    # init env, change robot if needed
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(DeltaEndEffectorPoseViaIK(), DiscreteWithoutChecking()),
        obs_config=get_default_camera_config(),
        headless=args.headless,
        robot_setup=robot_name,
    )
    rlbench_env.launch()

    task_env = rlbench_env.get_task(task_file_to_task_class(task_str))
    camera_mode = getattr(args, "camera_mode", None)
    camera_helper = CameraHelper(task_env, rot_center="workspace")
    X_C = camera_helper.X_C
    
    if getattr(args, "workspace", False):
        task_env._task.static_scene = False
        
    statistics = {}
    video_dir = getattr(args, "video_dir", "../videos")
    record_video = getattr(args, "record", False)
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        record_first_n = getattr(args, "record_first_n", args.num_rollouts)
        all_video_frames = []

    debug_crop = getattr(args, "debug_crop", False)
    debug_crop_first_n = getattr(args, "debug_crop_first_n", 1)
    debug_crop_every = max(1, getattr(args, "debug_crop_every", 1))
    debug_crop_dir = getattr(
        args,
        "debug_crop_dir",
        os.path.join(getattr(args, "save_path", "../results"), "debug_crops"),
    )
        
    try:
        num_success = 0
        traj_i = 0
        progress_bar = tqdm(range(args.num_rollouts))

        while traj_i < args.num_rollouts:
            
            task_env.reset()
            task_id = task_env._task.subtask_id
            obj_height = task_env._task.target_objects[task_id].get_matrix()[2, 3]
            object_position_from_mask = None

            # ood: workspace, align robot ee pose w.r.t. object position
            if getattr(args, "workspace", False):
                front_mask = task_env.get_observation().front_mask
                
                object_positions_from_mask = [
                    camera_helper.get_object_position_from_mask(
                        front_mask, obj_height, id=object_id
                    )
                    for object_id in task_env._task.target_handles
                ]

                if any(pos is None for pos in object_positions_from_mask):
                    print(f"[Retry] Failed to get all object positions at rollout {traj_i}, retrying...")
                    continue

                object_id = task_env._task.target_handles[task_id]
                object_position_from_mask = object_positions_from_mask[task_id]

            try:
                task_env._task.set_hand_wrt_target(object_position=object_position_from_mask)
            
            except IKError:
                print(f"[Retry] IK error for {traj_i}, retrying...")
                continue

            # ood: camera, rotate robot ee w.r.t. camera
            X_C_prime, _ = camera_helper.move_camera(mode=camera_mode)
            if camera_mode is not None and args.align_cam:
                task_env._task.align_eef_rotation_to_camera(X_Cs=(X_C, X_C_prime))
                
            task_env._task.pyrep.step()
            observation = task_env.get_observation()
            initial_gripper_matrix = observation.gripper_matrix

            for step_i in range(args.horizon):
                with torch.no_grad():
                    current_gripper_matrix = observation.gripper_matrix.copy()
                    action_prediction = agent.predict_action(observation)
                    if (
                        debug_crop
                        and traj_i < debug_crop_first_n
                        and step_i % debug_crop_every == 0
                    ):
                        _save_debug_crops(agent, debug_crop_dir, traj_i, step_i)
                    gripper_action = action_prediction["gripper"]["action"][0]
                    eef_action_type = action_prediction["eef"]["type"]
                    eef_action = action_prediction["eef"]["action"][0]
                    
                    if eef_action_type == "delta":
                        base_matrix = current_gripper_matrix
                    elif eef_action_type == "relative":
                        base_matrix = initial_gripper_matrix
                    else:
                        base_matrix = np.eye(4)

                    eef_action_transformed = base_matrix @ eef_action
                    action_for_env = np.concatenate(
                        [eef_action_transformed.reshape(16), gripper_action]
                    )
                    try:
                        observation, success, terminate = task_env.step(action_for_env)
                        
                        if record_video and traj_i < record_first_n:
                            frame_rgb = observation.front_rgb
                            all_video_frames.append(frame_rgb)

                    except InvalidActionError as e:
                        success = False
                        print(e)
                        break

                    if success:
                        num_success += 1
                        break
                    if terminate:
                        break
                    
                    if not task_env._task.is_local():
                        task_id = task_env._task.get_subtask_id()
                        task_env._task.subtask_id = task_id
                        obj_height = task_env._task.target_objects[task_id].get_matrix()[2, 3]
                        
                        if getattr(args, "workspace", False):
                            front_mask = task_env.get_observation().front_mask
                            object_id = task_env._task.target_handles[task_id]
                            object_position_from_mask = camera_helper.get_object_position_from_mask(
                                front_mask, obj_height, id=object_id
                            )
                            if object_position_from_mask is None:
                                print(f"[Retry] Failed to get object position at rollout {traj_i}, retrying...")
                                continue
                        
                        try:
                            task_env._task.set_hand_wrt_target(object_position=object_position_from_mask)
                        
                        except IKError:
                            print(f"[Retry] IK error for {traj_i}, retrying...")
                            continue
                        observation = task_env.get_observation()

            statistics[traj_i] = {
                "robot": robot_name,
                "success": success,
                "X_C": X_C,
                "X_C_prime": X_C_prime,
            }
            
            traj_i += 1
            progress_bar.set_description(f"Success Rate: {num_success / (traj_i):.2f}")
            progress_bar.update(1)

            
    finally:
        rlbench_env.shutdown()

    success_rate = num_success / args.num_rollouts
    if record_video and len(all_video_frames) > 0:
        prefix = task_str
        if args.workspace:
            prefix += "_workspace_shifted"
        if camera_mode is not None:
            prefix += f"_camera_{camera_mode}"
        if robot_name != "panda":
            prefix += f"_{robot_name}"
        out_path = os.path.join(video_dir, f"{prefix}_rollouts.mp4")
        with imageio.get_writer(out_path, fps=10, codec='libx264', quality=8) as writer:
            for frame in all_video_frames:
                writer.append_data(frame)

    return success_rate, statistics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="../results")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--num_rollouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("-w", "--workspace", action="store_true")
    parser.add_argument("-c", "--camera_mode", type=str, default=None)
    parser.add_argument("-r", "--robot", type=str, default="panda")
    parser.add_argument("--align_cam", action="store_true")

    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record_first_n", type=int, default=1)
    parser.add_argument("--video_dir", type=str, default="../videos")
    parser.add_argument("--debug_crop", action="store_true")
    parser.add_argument("--debug_crop_first_n", type=int, default=1)
    parser.add_argument("--debug_crop_every", type=int, default=1)
    parser.add_argument("--debug_crop_dir", type=str, default="../results/debug_crops")
    parser.add_argument("--task", type=str, default=None)

    cli_args = parser.parse_args()

    seed = cli_args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = RLBenchAgent(model_dir=cli_args.model_path)
    parent_folder = os.path.dirname(cli_args.model_path)
    model_config_path = os.path.join(parent_folder, "config.json")
    model_config = load_config_to_namespace(model_config_path, full_path=True)
    task_name = model_config.dataset.task_name
    
    if cli_args.task is not None:
        task_name = cli_args.task
    evaluate_policy(agent, task_name, cli_args)


if __name__ == "__main__":
    main()
