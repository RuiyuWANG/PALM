import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

from palm import PROJ_DIR
from palm.utils.transform_utils import check_pts, check_SE3


# ------------------- Camera Projection and Transformation ------------------- #


def get_camera_matrices(K, X_C):
    """Gets the camera matrix.

    :param K: camera intrinsics.
    :param X_C: camera extrinsics, i.e. the camera pose.
    :return: The camera matrix. K @ X_C^-1
    """
    check_cam_intrinsics(K)
    check_SE3(X_C)
    has_batch_dim = len(K.shape) == 3
    if not has_batch_dim:
        K = np.expand_dims(K, axis=0)
        X_C = np.expand_dims(X_C, axis=0)

    assert K.shape[0] == X_C.shape[0], "Batch size mismatch."

    # add a column of zeros if K is bx3x3 to make it bx3x4
    if K.shape[-1] == 3:
        K = np.concatenate([K, np.zeros((K.shape[0], 3, 1))], axis=-1)
    mat = K @ np.linalg.inv(X_C)
    return mat if has_batch_dim else mat[0]


def project_points(K, X_C, pts=None, poses=None, to_int=False):
    """
    Transforms world points or poses to camera coordinates.

    Parameters:
    - K: np.array of camera intrinsics of shape (3, 3) or (3, 4)
    - X_C: np.array of SE(3) matrix of shape (4, 4)
    - pts: np.array of shape (N, 3) or (N, 4), optional
    - poses: np.array of SE(3) matrix of shape (4, 4), optional

    Returns:
    - cam_pts: np.array of transformed camera coordinates
    """
    # pts and poses should only have one being not None
    assert (pts is not None) ^ (poses is not None), (
        "Either pts or poses should be provided."
    )

    if poses is not None:
        check_SE3(poses)
        has_batch_dim = len(poses.shape) == 3
        poses = poses.reshape(-1, 4, 4)
        pts_homog = poses[:, :, 3]
    else:
        is_homog = check_pts(pts)
        has_batch_dim = len(pts.shape) == 2
        if is_homog:
            pts_homog = pts
        else:
            pts_homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)
        pts_homog = pts_homog.reshape(-1, 4)
    pts_homog = pts_homog[:, :, np.newaxis]  # (N, 4, 1)
    camera_matrices = get_camera_matrices(K, X_C).reshape(-1, 3, 4)  # (N, 3, 4)
    assert camera_matrices.shape[0] == pts_homog.shape[0], "Batch size mismatch."

    cam_pts = (camera_matrices @ pts_homog).squeeze(-1)  # (N, 3)
    cam_pts /= cam_pts[:, 2].reshape(-1, 1)  # (N, 3)
    cam_pts = cam_pts[:, :2]  # (N, 2)
    if to_int:
        cam_pts = cam_pts[:, :2].astype(int)
    return cam_pts if has_batch_dim else cam_pts[0]


# -------------- Shape, Channel Ordering and Backend Conversion -------------- #


class ShapeHelper:
    def __init__(self, debug=False, debug_save_dir="../data/debug_images"):
        self.debug_save_dir = debug_save_dir
        self.debug = debug
        self.reset_meta()
        if not os.path.exists(debug_save_dir) and debug:
            os.makedirs(debug_save_dir)

    def reset_meta(self):
        self.meta = {
            "is_set": False,
            "dtype": None,
            "input_order": None,
            "current_order": None,
            "input_ndim": None,
            "current_ndim": None,
            "input_shape": None,
        }

    def set_meta(self, img):
        if self.meta["is_set"]:
            return

        assert isinstance(img, np.ndarray), "Input image must be a numpy array!"
        assert img.ndim in [3, 4], "Input must have 3 or 4 dimensions!"
        channel_order = "bhwc" if img.shape[-1] in [3, 4] else "bchw"
        self.meta.update(
            {
                "is_set": True,
                "input_shape": img.shape,
                "dtype": img.dtype,
                "input_ndim": img.ndim,
                "current_ndim": img.ndim,
                "input_order": channel_order,
                "current_order": channel_order,
            }
        )

    def reorder_channels(self, img, order, backend="torch"):
        assert order in ["bchw", "bhwc"], "Order must be either 'bchw' or 'bhwc'!"
        assert backend in ["torch", "numpy"], (
            "Backend must be either 'torch' or 'numpy'!"
        )
        self.set_meta(img)
        if self.meta["current_ndim"] == 3:
            img = np.expand_dims(img, axis=0)
            self.meta["current_ndim"] = 4
        # Rearrange the channels
        if self.meta["current_order"] != order:
            pattern = f"{' '.join(self.meta['current_order'])} -> {' '.join(order)}"
            img = rearrange(img, pattern)
            self.meta["current_order"] = order
        return torch.tensor(img, dtype=torch.float32) if backend == "torch" else img

    def restore_channel_orders(self, img, backend="numpy"):
        """
        Restore the original shape and dtype of the image.
        """
        assert self.meta["is_set"], "Meta data must be set before restoring the image!"
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if self.meta["input_order"] != self.meta["current_order"]:
            img = self.reorder_channels(img, self.meta["input_order"], backend=backend)
        if self.meta["input_ndim"] == 3:
            img = np.squeeze(img, axis=0)
        return img.astype(self.meta["dtype"])

    def save_debug_images(self, np_img, img_name):
        if not self.debug:
            return
        assert isinstance(np_img, np.ndarray), "Input image must be a numpy array!"
        self.reset_meta()
        np_img = self.reorder_channels(np_img, "bhwc", backend="numpy")
        for i in range(np_img.shape[0]):
            img_ = np_img[i]
            # Scale the image to 0-255 if it's in a 0-1 range
            img_ = img_ * 255 if np.amax(img_) <= 1 else img_
            img_ = Image.fromarray(img_.astype(np.uint8))
            img_.save(f"{self.debug_save_dir}/{img_name}_{i}.png")


# --------------------------- Image Transformations -------------------------- #


def crop_at_coords(
    images,
    coords,
    crop_size,
    jitter_crop,
    height_offset=0,
    allow_out_of_bounds=True,
    img_prefix=None,
    debug=False,
):
    """
    Crops the image at specified coordinates.

    Parameters:
    - images: np.ndarray, input images  [B, C, H, W], [B, H, W, C]
    - coords: np.ndarray, coordinates for cropping, set to None for center cropping
    - crop_size: int or tuple, size of the crop
    - height_offset: int, vertical offset of the coords for cropping
    - allow_out_of_bounds: bool, whether to allow out of bounds cropping, outside regions are filled with zeros
    - img_prefix: str, prefix for debug image names
    - debug: bool, whether to enable debug mode

    Returns:
    - crops: np.ndarray, cropped images
    """
    assert isinstance(images, np.ndarray), "Input image must be a numpy array!"
    assert isinstance(coords, np.ndarray) or coords is None, (
        "Image coordinates must be a numpy array!"
    )

    if isinstance(crop_size, tuple) or isinstance(crop_size, list):
        assert len(crop_size) == 2, "Crop size must be a tuple of two integers!"
        crop_size = crop_size[0]
    assert crop_size > 0, "Crop size must be a positive!"
    assert np.amax(images.shape[1:]) >= crop_size, (
        "Image size must be larger than crop size!"
    )
    #
    img_helper = ShapeHelper(debug=debug)
    _images = img_helper.reorder_channels(images, "bchw", backend="torch")
    B = _images.shape[0]
    if coords is None:
        # default to center of the image
        coords = np.array([_images.shape[2] // 2, _images.shape[3] // 2])
        coords = np.tile(coords, (B, 1))
    else:
        coords = coords.reshape(-1, 2)
    assert coords.shape[-1] == 2, "Coordinates must be in (x, y) format!"
    coords = torch.tensor(coords, dtype=torch.float32)
    if jitter_crop > 0:
        coords = torch.round(coords + (torch.rand_like(coords) - 0.5) * 2 * jitter_crop)

    b, c, h, w = _images.shape
    crops = torch.zeros((b, c, crop_size, crop_size), dtype=torch.float32)
    for i in range(b):
        h_ = int(coords[i, 1] + height_offset - crop_size // 2)
        w_ = int(coords[i, 0] - crop_size // 2)
        if not allow_out_of_bounds:
            h_ = max(0, h_)
            w_ = max(0, w_)
            h_ = min(h - crop_size, h_)
            w_ = min(w - crop_size, w_)
        else:
            h_ = min(max(0, h_), h - crop_size)
            w_ = min(max(0, w_), w - crop_size)
        crop = _images[i, :, h_ : h_ + crop_size, w_ : w_ + crop_size]
        crops[i] = crop

    crops = img_helper.restore_channel_orders(crops, backend="numpy")
    img_helper.save_debug_images(crops, img_name=f"{img_prefix}_crop")

    return crops


def resize_images(img, resize_size, img_prefix=None, debug=False):
    assert isinstance(img, np.ndarray), "Input image must be a numpy array!"
    if isinstance(resize_size, tuple) or isinstance(resize_size, list):
        assert len(resize_size) == 2, "Resize size must be a tuple of two integers!"
        resize_size = resize_size[0]
    assert resize_size > 0, "Resize size must be a positive!"
    resize_size = int(resize_size)
    if img.shape[-1] == resize_size:
        return img
    img_helper = ShapeHelper(debug=debug)

    img = img_helper.reorder_channels(img, "bchw", backend="torch")
    img_resized = F.interpolate(
        img, size=(resize_size, resize_size), mode="bilinear", align_corners=False
    )
    img_resized = img_helper.restore_channel_orders(img_resized, backend="numpy")
    img_helper.save_debug_images(img_resized, img_name=f"{img_prefix}_resized")
    return img_resized


def overlay_background(res, bg_path, img_size):
    H, W = img_size

    # Load and resize background
    bg = Image.open(bg_path).convert("RGB").resize((W, H))
    bg_np = np.array(bg)

    out = {}
    for key, img in res.items():
        # Convert to numpy RGB
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"))
        else:
            img_np = img
            if img_np.ndim == 2:  # grayscale to RGB
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.shape[-1] == 4:  # remove alpha if needed
                img_np = img_np[..., :3]

        # Resize to target size
        img_np = resize_images(img_np, (H, W))

        # Normalize and blend
        fg = img_np.astype(np.float32) / 255.0
        bgf = bg_np.astype(np.float32) / 255.0

        # α-blend with mask (nonzero pixels kept)
        mask = (fg.mean(axis=-1, keepdims=True) > 0.05).astype(np.float32)
        blended = fg * mask + bgf * (1 - mask)

        out[key] = (blended * 255).astype(np.uint8)

    return out


def generate_grids(x_min, x_max, y_min, y_max, img_h, img_w, crop_size):
    c_x = (x_min + x_max) / 2
    c_y = (y_min + y_max) / 2

    c_x_normalized = (c_x / img_w) * 2 - 1
    c_y_normalized = (c_y / img_h) * 2 - 1

    # Create a base grid for a square crop
    k_w, k_h = crop_size / img_w, crop_size / img_h
    grid_x = (
        torch.linspace(-k_w, k_w, crop_size)
        .unsqueeze(0)
        .repeat(crop_size, 1)
        .unsqueeze(-1)
    )
    grid_y = (
        torch.linspace(-k_h, k_h, crop_size)
        .unsqueeze(0)
        .repeat(crop_size, 1)
        .unsqueeze(-1)
    )
    base_grid = torch.cat([grid_x, grid_y.transpose(1, 0)], dim=-1).unsqueeze(
        0
    )  # [1, c, c, 2]

    # Adjust grid for each center point
    n = c_x.shape[0]
    grid = base_grid.repeat(n, 1, 1, 1)  # [n, crop_size, crop_size, 2]
    grid[..., 0] += c_x_normalized.view(-1, 1, 1).expand(-1, crop_size, crop_size)
    grid[..., 1] += c_y_normalized.view(-1, 1, 1).expand(-1, crop_size, crop_size)

    return grid


# ------------------------------- Visualisation ------------------------------ #


def overlay_poses(
    images,
    poses,
    K,
    X_C,
    thickness=2,
    axis_length=0.1,
    dot_size=5,
    im_prefix="pose",
    jitter_crop=0,
    debug=False,
):
    assert isinstance(images, np.ndarray), "Input images must be a numpy array."
    assert isinstance(poses, np.ndarray), "Input poses must be a numpy array."
    check_SE3(poses)
    has_batch_dim = poses.ndim == 3

    pose = poses.reshape(-1, 4, 4) if not has_batch_dim else poses
    _images = images.copy()

    B = pose.shape[0]
    axes_homog = np.concatenate([np.eye(3) * axis_length, np.ones((1, 3))], axis=0)
    axes_homog = axes_homog.reshape(1, 4, 3).repeat(B, axis=0)
    axes_in_pose = poses @ axes_homog
    pts = project_points(K, X_C, poses=poses, to_int=True)
    pts = pts.reshape(1, -1) if not pts.ndim == 2 else pts
    #
    x_axis = project_points(K, X_C, pts=axes_in_pose[:, :, 0], to_int=True)
    y_axis = project_points(K, X_C, pts=axes_in_pose[:, :, 1], to_int=True)
    z_axis = project_points(K, X_C, pts=axes_in_pose[:, :, 2], to_int=True)

    if jitter_crop > 0:
        noise = (np.random.rand(*pts.shape) - 0.5) * (2 * jitter_crop)
        pts = pts + noise

        x_axis = x_axis + noise
        y_axis = y_axis + noise
        z_axis = z_axis + noise

    helper = ShapeHelper(debug=debug)
    _images = helper.reorder_channels(_images, "bhwc", backend="numpy")
    max_range = 1 if np.amax(_images) <= 1 else 255
    _images = _images * 255 if max_range == 1 else _images
    overlayed_imgs = []

    # draw each axis as a line
    for i in range(B):
        im_ = Image.fromarray(_images[i].astype(np.uint8))
        draw = ImageDraw.Draw(im_)

        start = (pts[i][0], pts[i][1])
        x = (x_axis[i][0], x_axis[i][1])
        y = (y_axis[i][0], y_axis[i][1])
        z = (z_axis[i][0], z_axis[i][1])

        # draw arrows
        draw_arrow(draw, start, x, (255, 0, 0), thickness)
        draw_arrow(draw, start, y, (0, 255, 0), thickness)
        draw_arrow(draw, start, z, (0, 0, 255), thickness)

        # draw dot
        draw.ellipse(
            [
                start[0] - dot_size,
                start[1] - dot_size,
                start[0] + dot_size,
                start[1] + dot_size,
            ],
            fill=(255, 255, 255),
            outline=None,
        )
        overlayed_imgs.append(np.array(im_))

    overlayed_imgs = np.stack(overlayed_imgs, axis=0)
    overlayed_imgs = overlayed_imgs / 255 if max_range == 1 else overlayed_imgs
    overlayed_imgs = helper.restore_channel_orders(overlayed_imgs, backend="numpy")
    helper.save_debug_images(overlayed_imgs, im_prefix)

    return overlayed_imgs


def overlay_points(images, pts, K, X_C, thickness=2, im_prefix="pts", debug=False):
    assert isinstance(images, np.ndarray), "Input images must be a numpy array."

    check_pts(pts)
    has_batch_dim = pts.ndim == 2
    pts = pts.reshape(-1, 3)
    _images = images.copy()

    B = pts.shape[0]
    coords = project_points(K, X_C, pts=pts, to_int=True)

    helper = ShapeHelper(debug=debug)
    _images = helper.reorder_channels(_images, "bhwc", backend="numpy")
    max_range = 1 if np.amax(_images) <= 1 else 255
    _images = _images * 255 if max_range == 1 else _images
    overlayed_imgs = []

    # draw each axis as a line
    for i in range(B):
        im_ = Image.fromarray(images[i].astype(np.uint8))
        draw = ImageDraw.Draw(im_)
        x, y = coords[i][0], coords[i][1]
        draw.ellipse(
            [x - thickness, y - thickness, x + thickness, y + thickness],
            fill=(255, 0, 0),
        )
        overlayed_imgs.append(np.array(im_))

    overlayed_imgs = np.stack(overlayed_imgs, axis=0)
    overlayed_imgs = overlayed_imgs / 255 if max_range == 1 else overlayed_imgs
    overlayed_imgs = helper.restore_channel_orders(overlayed_imgs, backend="numpy")
    helper.save_debug_images(overlayed_imgs, im_prefix)

    return overlayed_imgs if has_batch_dim else overlayed_imgs[0]


def draw_arrow(
    draw: ImageDraw.ImageDraw, start: tuple, end: tuple, color: str, width: int = 1
):
    """
    Draw an arrow from start to end.
    """
    arrow_length = 10  # Length of the arrow head
    arrow_width = 5  # Width of the arrow head

    # Calculate the direction of the arrow
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    if length == 0:
        return
    direction = direction / length

    # Calculate the points for the arrow head
    left = np.array([-direction[1], direction[0]]) * arrow_width
    right = np.array([direction[1], -direction[0]]) * arrow_width
    head_base = np.array(end) - direction * arrow_length

    arrow_head = [tuple(end), tuple(head_base + left), tuple(head_base + right)]

    # Draw the arrow line
    draw.line([tuple(start), tuple(head_base)], fill=color, width=width)
    # Draw the arrow head
    draw.polygon(arrow_head, fill=color)


def overlay_text(image, text, position, font_size=20, font_color=(255, 255, 255)):
    """
    Overlay text on an image.

    Parameters:
    - image: PIL.Image, input image
    - text: str, text to overlay
    - position: tuple, position of the text
    - font_size: int, font size
    - font_color: tuple, font color
    """
    assert isinstance(image, Image.Image), "Input image must be a PIL image."
    assert isinstance(text, str), "Text must be a string."
    assert isinstance(position, tuple), "Position must be a tuple."
    assert isinstance(font_size, int), "Font size must be an integer."
    assert isinstance(font_color, tuple), "Font color must be a tuple."

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=font_color)


def save_torch_images(
    images,
    obs_normaliser=None,
    save_dir=os.path.join(PROJ_DIR, "debug_imgs"),
    prefix="img",
):
    """
    Save a batch of torch images to disk.

    Parameters:
    - images: torch.Tensor, input images of shape [B, C, H, W]
    - save_dir: str, directory to save the images
    - prefix: str, prefix for the image names
    """
    assert isinstance(images, torch.Tensor), "Input images must be a torch tensor."
    assert images.ndim == 4, "Input images must have 4 dimensions."
    assert isinstance(save_dir, str), "Save directory must be a string."
    assert isinstance(prefix, str), "Prefix must be a string."

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if obs_normaliser is not None:
        images = obs_normaliser.denormalize(images)

    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"{save_dir}/{prefix}_{i}.png")


# ------------------------------ Sanity Checkers ----------------------------- #


def check_cam_intrinsics(mat: Union[np.ndarray, torch.Tensor], atol=1e-5):
    """
    Check if the input matrix is a valid camera intrinsics matrix.

    Parameters:
    - mat: Union[np.ndarray, torch.Tensor], input matrix of shape (3, 3) or (3, 4) or batch of such matrices.
    - atol: float, absolute tolerance for numerical checks.

    Raises:
    - AssertionError: If the input matrix is not a valid camera intrinsics matrix.
    """
    assert isinstance(mat, (np.ndarray, torch.Tensor)), (
        "Input must be a numpy array or torch tensor."
    )

    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().numpy()

    has_batch_dim = mat.ndim == 3
    if not has_batch_dim:
        mat = np.expand_dims(mat, axis=0)

    shape_check = mat.shape[-2:] in [(3, 3), (3, 4)]
    assert shape_check, "Invalid camera intrinsics matrix shape."

    if mat.shape[-1] == 4:
        last_col_check = np.allclose(mat[:, :, -1], np.array([0, 0, 0]), atol=atol)
        last_row_check = np.allclose(mat[:, -1, :], np.array([0, 0, 0, 1]), atol=atol)
        assert last_col_check and last_row_check, (
            "Last column and row of camera intrinsics matrix must be [0, 0, 0] and [0, 0, 0, 1]."
        )

    if mat.shape[-1] == 3:
        last_row_check = np.allclose(mat[:, -1, :], np.array([0, 0, 1]), atol=atol)
        assert last_row_check, "Last row of camera intrinsics matrix must be [0, 0, 1]."

    if not has_batch_dim:
        mat = mat.squeeze(0)

    return mat


def check_cam_extrinsics(mat):
    return check_SE3(mat)
