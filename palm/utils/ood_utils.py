import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from palm.utils.image_utils import project_points

class CameraHelper:
    def __init__(self, task_env, rot_center):
        assert rot_center in ["workspace", "task_base", None]
        self.env = task_env
        self.movement_center = self.get_movement_center(rot_center)
        self.cam = self.env._scene._cam_front
        self.X_C = self.get_extrinsic()

    def get_movement_center(self, rot_center=None):
        if rot_center == "workspace":
            X_center = self.env._scene._workspace
        elif rot_center == "task_base":
            X_center = self.env._task.get_base()
        else:
            return None
        return X_center

    def get_extrinsic(self):
        camera_extrinsic = self.cam.get_matrix()
        return np.array(camera_extrinsic)

    def get_intrinsic(self):
        camera_intrinsic = self.cam.get_intrinsic_matrix()
        return np.array(camera_intrinsic)

    def get_intrinsic_extrinsic(self):
        return self.get_intrinsic(), self.get_extrinsic()

    def set_extrinsic(self, X_C, relative_to=None):
        X_C = np.array(X_C)
        if relative_to is None:
            relative_to = None
        elif relative_to == "workspace":
            relative_to = self.env._scene._workspace
        elif relative_to == "task_base":
            relative_to = self.env._task.get_base()
        else:
            raise NotImplementedError(
                "The relative frame for set extrinsic is not implemented yet!"
            )
        self.cam.set_matrix(X_C, relative_to=relative_to)

    def move_camera_in_arc(self, rad):
        try:
            X_M = self.movement_center.get_matrix()
        except AttributeError:
            X_M = np.eye(4)

        M_X_C = np.linalg.inv(X_M) @ self.X_C

        # Rotate the camera around the movement center
        delta_X = np.eye(4)
        delta_X[:3, :3] = R.from_rotvec(rad * np.array([0, 0, 1])).as_matrix()
        M_X_C = delta_X @ M_X_C

        X_C_prime = X_M @ M_X_C
        self.cam.set_matrix(X_C_prime)

        return X_C_prime

    def move_camera_in_cone(self, rad, scale_factor, xy_noise):
        try:
            X_M = self.movement_center.get_matrix()
        except AttributeError:
            X_M = np.eye(4)

        M_X_C = np.linalg.inv(X_M) @ self.X_C

        # Rotate the camera around the movement center
        delta_rad = np.eye(4)
        delta_rad[:3, :3] = R.from_rotvec(rad * np.array([0, 0, 1])).as_matrix()
        M_X_C = delta_rad @ M_X_C

        # adjust radius
        cam_pos = M_X_C[:3, 3]
        r = np.sqrt(cam_pos[0] ** 2 + cam_pos[1] ** 2)
        theta = np.arctan2(cam_pos[1], cam_pos[0])
        r_new = r * scale_factor

        new_x = r_new * np.cos(theta) + np.random.uniform(-xy_noise, xy_noise)
        new_y = r_new * np.sin(theta) + np.random.uniform(-xy_noise, xy_noise)

        gamma = np.arcsin(cam_pos[2] / r)
        new_z = r_new * np.sin(gamma)

        M_X_C[:3, 3] = [new_x, new_y, new_z]

        X_C_prime = X_M @ M_X_C
        self.cam.set_matrix(X_C_prime)

        return X_C_prime

    def cam_pt_to_table_pt(self, cam_pt, table_height):
        K, X_C = self.get_intrinsic_extrinsic()
        R, t = X_C[:3, :3], X_C[:3, 3]

        p_image = np.array([cam_pt[0], cam_pt[1], 1])
        p_camera = np.linalg.inv(K) @ p_image
        p_world_ray = R @ p_camera

        s = (table_height - t[2]) / p_world_ray[2]
        p_world = t + s * p_world_ray
        return p_world

    def move_camera(self, mode):
        assert mode in ["mild", "median", "max", "cone", "random", None]
        if mode is None:
            return None, None
        elif mode == "mild":
            rad = np.random.uniform(-1 / 8 * np.pi, 1 / 8 * np.pi)
            X_C_prime = self.move_camera_in_arc(rad)
        elif mode == "median":
            rad = np.random.uniform(-1 / 6 * np.pi, 1 / 6 * np.pi)
            X_C_prime = self.move_camera_in_arc(rad)
        elif mode == "max":
            rad = np.random.uniform(-1 / 4 * np.pi, 1 / 4 * np.pi)
            X_C_prime = self.move_camera_in_arc(rad)
        elif mode == "cone":
            rad = np.random.uniform(-1 / 8 * np.pi, 1 / 8 * np.pi)
            scale_factor = np.random.uniform(0.8, 1.2)
            X_C_prime = self.move_camera_in_cone(rad, scale_factor, xy_noise=0.0)
        elif mode == "random":
            rad = np.random.uniform(-1 / 8 * np.pi, 1 / 8 * np.pi)
            scale_factor = np.random.uniform(0.8, 1.2)
            X_C_prime = self.move_camera_in_cone(rad, scale_factor, xy_noise=0.05)
        return X_C_prime, rad

    def get_object_position_from_mask(self, mask_rgb, table_height, id):
        mask_gray = mask_rgb[:, :, 0]
        mask_gray = (mask_gray * 255).astype(np.uint8)

        object_coords = np.argwhere(mask_gray == id)
        if object_coords.size == 0:
            print("Occlusion or missing object — unable to get object mask!")
            return None
        object_centroid = object_coords.mean(axis=0)
        x, y = int(object_centroid[1]), int(object_centroid[0])

        object_position_in_pixel = np.array([x, y])
        object_position_in_world = self.cam_pt_to_table_pt(object_position_in_pixel, table_height)
        return object_position_in_world
    
    def get_robot_object_masks(self, front_mask):
        robot_visual_names = self.env._task.robot.arm.get_visuals() + self.env._task.robot.gripper.get_visuals()
        robot_handles = [obj.get_handle() for obj in robot_visual_names]
        obj_handles = self.env._task.visible_handles
        handles = robot_handles + obj_handles
        
        mask_gray = front_mask[:, :, 0]
        mask_gray = (mask_gray * 255).astype(np.uint8)
        masks = np.isin(mask_gray, handles)
        return masks
    
    def get_pixel_space_bbx(self, obj, table_center=np.array([0.25, 0, 0.752]), offset=0):
        position_world = obj.get_matrix()[:3, 3]
        position_table = position_world
        size_world = obj.get_bounding_box()
        x1, y1 = size_world[0] + offset, size_world[2] + offset
        x0, y0, z0 = position_table[0], position_table[1], table_center[2]
        corners_world = [
            [x0 + x1, y0 + y1, z0],
            [x0 - x1, y0 + y1, z0],
            [x0 - x1, y0 - y1, z0],
            [x0 + x1, y0 - y1, z0]
        ]

        K, XC = self.get_intrinsic_extrinsic()
        pixel_corners = [
            project_points(K, XC, np.array([pt]), to_int=True)[0]
            for pt in corners_world
        ]

        return pixel_corners
            
    def visualise_mask(self, mask, position, object_position_gt, img_path):
        plt.figure(figsize=(6, 6))
        plt.imshow(mask[:, :, 0], cmap="gray")
        plt.scatter(
            position[0], position[1], color="red", marker="+", s=100, label="Spam Position"
        )
        plt.scatter(
            object_position_gt[0],
            object_position_gt[1],
            color="green",
            marker="+",
            s=100,
            label="Ground Truth",
        )
        plt.title("Detected Spam Position")
        plt.legend()
        plt.axis("off")
        plt.savefig(img_path, bbox_inches="tight")
