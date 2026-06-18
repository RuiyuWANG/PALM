from typing import Union, Tuple, Literal
import numpy as np

import torch
import torch.functional as F
from scipy.spatial.transform import Rotation as R

# -------------------------------- Lie Group Operation -------------------------------- #


def vee(X: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 matrix (or batch of such matrices) in se(3) Lie algebra form
    to a 6D twist vector (or batch of 6D vectors).

    Args:
        X (np.ndarray): A single (4, 4) or batched (N, 4, 4) se(3) matrix/matrices.

    Returns:
        np.ndarray: A (6,) or (N, 6) twist vector or batch.
    """
    assert isinstance(X, np.ndarray), f"Expected np.ndarray, got {type(X)}"
    assert X.ndim in [2, 3], (
        f"Input must be (4, 4) or (N, 4, 4), but got shape {X.shape}"
    )
    check_se3(X)

    is_single = X.ndim == 2
    if is_single:
        X = X[None, ...]

    assert X.shape[1:] == (4, 4), f"Input must be (N, 4, 4), got {X.shape}"

    v = X[:, :3, 3]  # translation part
    w = np.stack(
        [X[:, 2, 1], X[:, 0, 2], X[:, 1, 0]], axis=1
    )  # angular part from skew-symmetric

    twist = np.concatenate([v, w], axis=1)
    return twist[0] if is_single else twist


def wedge(xi: np.ndarray) -> np.ndarray:
    """
    Map 6D twist vector(s) to se(3) matrix/matrices.
    Args:
        xi: (6,) or (N, 6) NumPy array
    Returns:
        (4, 4) or (N, 4, 4) se(3) matrix/matrices
    """
    assert isinstance(xi, (np.ndarray, torch.Tensor)), (
        f"Input must be a numpy array, torch tensor, or list, but got {type(xi)}"
    )

    has_batch_dim = xi.ndim == 2
    if not has_batch_dim:
        xi = xi[None, :]

    assert xi.shape[1] == 6, f"Input must have shape (6,) or (N, 6), but got {xi.shape}"

    v, w = xi[:, :3], xi[:, 3:]
    N = xi.shape[0]

    omega_hat = np.zeros((N, 3, 3))
    omega_hat[:, [0, 0, 1], [1, 2, 2]] = -w[:, [2, 1, 0]]
    omega_hat[:, [1, 2, 2], [0, 0, 1]] = w[:, [2, 1, 0]]

    mat = np.zeros((N, 4, 4))
    mat[:, :3, :3] = omega_hat
    mat[:, :3, 3] = v
    check_se3(mat)
    return mat[0] if not has_batch_dim else mat


def get_skew_symmetric_matrix(w: np.ndarray) -> np.ndarray:
    """
    Convert a 3D angular velocity vector or a batch of such vectors to a 3x3 (or N×3×3) skew-symmetric matrix.

    For a vector w = [w_x, w_y, w_z], returns the matrix:
        [  0  -w_z  w_y ]
        [ w_z   0  -w_x ]
        [-w_y  w_x   0  ]

    Args:
        w (np.ndarray or torch.Tensor): Shape (3,) or (N, 3)

    Returns:
        np.ndarray or torch.Tensor: Shape (3, 3) or (N, 3, 3)
    """
    assert isinstance(w, (np.ndarray, torch.Tensor)), (
        f"Input must be a numpy array or torch tensor, got {type(w)}"
    )

    is_single = w.ndim == 1
    if is_single:
        w = w[None, :]
    assert w.shape[1] == 3, f"Expected shape (3,) or (N, 3), but got {w.shape}"

    N = w.shape[0]
    S = np.zeros((N, 3, 3), dtype=w.dtype)

    S[:, 0, 1] = -w[:, 2]
    S[:, 0, 2] = w[:, 1]
    S[:, 1, 0] = w[:, 2]
    S[:, 1, 2] = -w[:, 0]
    S[:, 2, 0] = -w[:, 1]
    S[:, 2, 1] = w[:, 0]

    return S[0] if is_single else S


def adjoint(X):
    """
    Compute the adjoint matrix of an SE(3) transformation or batch of transformations.

    The adjoint matrix Ad_X ∈ R^{6x6} (or batch of them) maps body-frame twists to
    spatial-frame twists: v_spatial = Ad_X @ v_body

    For a transformation matrix X = [R | p] ∈ SE(3),
    the adjoint matrix is:
        Ad_X = | R    [p]×R |
               | 0      R   |

    Args:
        X (np.ndarray): SE(3) transformation matrix or batch of matrices, shape (..., 4, 4)

    Returns:
        np.ndarray: Adjoint matrix/matrices, shape (..., 6, 6)
    """
    assert isinstance(X, np.ndarray)
    assert X.shape[-2:] == (4, 4), "Expected shape (..., 4, 4)"
    is_single = X.ndim == 2
    if is_single:
        X = X[None, :]

    R = X[..., :3, :3]
    p = X[..., :3, 3]
    pxR = get_skew_symmetric_matrix(p) @ R

    if X.ndim == 2:
        Ad = np.zeros((6, 6), dtype=X.dtype)
        Ad[:3, :3] = R
        Ad[:3, 3:] = pxR
        Ad[3:, 3:] = R
    else:
        batch_shape = X.shape[:-2]
        Ad = np.zeros(batch_shape + (6, 6), dtype=X.dtype)
        Ad[..., :3, :3] = R
        Ad[..., :3, 3:] = pxR
        Ad[..., 3:, 3:] = R

    return Ad[0] if is_single else Ad


def log_SO3(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the logarithm map of a rotation matrix (SO(3)) to its Lie algebra (so(3)).

    For a rotation matrix R ∈ SO(3), the logarithm map gives:
        ω̂ = (θ / (2 sinθ)) * (R - Rᵀ)
    and the rotation vector ω ∈ ℝ³ is extracted from ω̂.

    Args:
        R (np.ndarray): A (3, 3) or (N, 3, 3) rotation matrix or batch.

    Returns:
        omega_hat (np.ndarray): Skew-symmetric matrix or batch of shape (3, 3) or (N, 3, 3)
        omega (np.ndarray): Rotation vector(s) of shape (3,) or (N, 3)
    """
    assert isinstance(R, np.ndarray), f"Expected np.ndarray, got {type(R)}"
    assert R.ndim in [2, 3], f"Expected (3,3) or (N,3,3), got {R.shape}"
    check_SO3(R)

    is_single = R.ndim == 2
    if is_single:
        R = R[None, ...]

    assert R.shape[1:] == (3, 3), f"Expected shape (N,3,3), got {R.shape}"

    N = R.shape[0]
    tr = np.trace(R, axis1=1, axis2=2)
    cos_theta = np.clip((tr - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    omega_hat = np.zeros((N, 3, 3))
    omega = np.zeros((N, 3))

    for i in range(N):
        if np.isclose(theta[i], 0.0):
            continue  # No rotation, log = 0
        A = theta[i] / (2 * np.sin(theta[i]))
        R_diff = R[i] - R[i].T
        omega_hat[i] = A * R_diff
        omega[i] = [omega_hat[i][2, 1], omega_hat[i][0, 2], omega_hat[i][1, 0]]

    check_so3(omega_hat)
    return (omega_hat[0], omega[0]) if is_single else (omega_hat, omega)


def log_SE3(T: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute the logarithm map of SE(3) transformation(s) into 6D twist vector(s).

    Given T = [R t; 0 1] ∈ SE(3), the logarithm is:
        log(T) = [ω̂ v; 0 0], where:
            ω̂ = log(R) ∈ so(3)
            v = V⁻¹ t, with:
                V⁻¹ = I - 0.5[ω̂] + ((1/θ² - (1+cosθ)/(2θsinθ)) [ω̂]^2

    Args:
        T (np.ndarray): SE(3) matrix of shape (4,4) or batch (N,4,4)
        eps (float): small threshold for handling near-zero angles

    Returns:
        np.ndarray: 6D twist vector(s) of shape (6,) or (N,6)
    """
    assert isinstance(T, np.ndarray), f"Expected np.ndarray, got {type(T)}"
    assert T.ndim in [2, 3], f"Expected (4,4) or (N,4,4), got {T.shape}"
    check_SE3(T)

    is_single = T.ndim == 2
    if is_single:
        T = T[None, ...]

    assert T.shape[1:] == (4, 4), f"Expected shape (N,4,4), got {T.shape}"
    N = T.shape[0]

    R, t = T[:, :3, :3], T[:, :3, 3]
    from numpy.linalg import norm

    _, omega = log_SO3(R)
    theta = norm(omega, axis=1)

    V_inv = np.zeros((N, 3, 3))
    for i in range(N):
        if theta[i] < eps:
            V_inv[i] = np.eye(3)
        else:
            w = omega[i]
            wx = get_skew_symmetric_matrix(w)
            wx2 = wx @ wx
            c = 1.0 / theta[i] ** 2 - (1 + np.cos(theta[i])) / (
                2 * theta[i] * np.sin(theta[i])
            )
            V_inv[i] = np.eye(3) - 0.5 * wx + c * wx2

    v = np.einsum("nij,nj->ni", V_inv, t)

    twist = np.concatenate([v, omega], axis=1)
    return twist[0] if is_single else twist


def so3_exp(omega: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute the exponential map from so(3) to SO(3), converting rotation vectors
    into rotation matrices using Rodrigues' formula:

        R = I + sin(θ)[ω̂] + (1 - cos(θ))[ω̂]^2
    where:
        - ω ∈ ℝ³ is the rotation vector
        - θ = ||ω|| is the rotation angle
        - [ω̂] ∈ so(3) is the skew-symmetric matrix of ω

    Args:
        omega (np.ndarray): (3,) or (N, 3) rotation vectors
        eps (float): threshold for small-angle approximation

    Returns:
        np.ndarray: (3, 3) or (N, 3, 3) rotation matrices
    """
    assert isinstance(omega, np.ndarray), f"Expected np.ndarray, got {type(omega)}"
    assert omega.ndim in [1, 2], f"Expected (3,) or (N,3), got {omega.shape}"

    is_single = omega.ndim == 1
    if is_single:
        omega = omega[None, :]  # (1, 3)

    theta = np.linalg.norm(omega, axis=1, keepdims=True)
    theta_safe = np.where(theta < eps, 1.0, theta)
    wx = get_skew_symmetric_matrix(omega / theta_safe)

    wx2 = np.einsum("nij,njk->nik", wx, wx)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    R = np.eye(3) + sin_theta[:, None, None] * wx + (1 - cos_theta[:, None, None]) * wx2
    check_SO3(R)
    return R[0] if is_single else R


def se3_exp(xi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Exponential map from se(3) to SE(3), mapping a 6D twist vector to a 4x4 SE(3) transformation.

    Given xi = [v, w], the exponential map is:
        T = [exp(w^)   V v]
            [  0       1 ]

    Where:
        - exp(w^) is the SO(3) matrix from so(3) using Rodrigues' formula
        - V = I + (1 - cosθ)/θ w^ + (θ - sinθ)/θ w^2
        - θ = ||w||, w^ is the skew-symmetric matrix of w

    Args:
        xi (np.ndarray): (6,) or (N, 6) twist vector(s)

    Returns:
        np.ndarray: (4,4) or (N,4,4) SE(3) matrix/matrices
    """
    assert isinstance(xi, np.ndarray), f"Expected np.ndarray, got {type(xi)}"
    assert xi.ndim in [1, 2], f"Expected shape (6,) or (N,6), got {xi.shape}"

    is_single = xi.ndim == 1
    if is_single:
        xi = xi[None, :]  # (1, 6)

    v, w = xi[:, :3], xi[:, 3:]
    N = xi.shape[0]
    theta = np.linalg.norm(w, axis=1, keepdims=True)
    theta_safe = np.where(theta < eps, 1.0, theta)

    # Skew symmetric [w^] and [w^]^2
    w_hat = get_skew_symmetric_matrix(w / theta_safe)
    w_hat_sq = np.einsum("nij,njk->nik", w_hat, w_hat)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Rotation part
    R = so3_exp(w)

    # V matrix
    V = (
        np.eye(3)
        + (1 - cos_theta) / theta_safe * w_hat
        + (theta - sin_theta) / theta_safe * w_hat_sq
    )

    t = np.einsum("nij,nj->ni", V, v)

    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t

    check_SE3(T)
    return T[0] if is_single else T


def pose_to_twist(
    X1: np.ndarray,
    X2: np.ndarray,
    dt: float = 1.0,
    type: Literal["spatial", "body"] = "spatial",
) -> np.ndarray:
    """
    Compute the 6D twist (v; ω) that moves pose X1 to pose X2 over time dt.

    The twist is computed from the relative transformation:
        - 'body' twist: expressed in the frame of X1 (at time t)
                       uses X_rel = inv(X1) @ X2
        - 'spatial' twist: expressed in the world/base frame
                          transforms the body twist via Adjoint(X1)

    Args:
        X1 (np.ndarray): Initial SE(3) pose at time t, shape (4,4) or (N,4,4)
        X2 (np.ndarray): Target SE(3) pose at time t + dt, same shape as X1
        dt (float): Time interval between the two poses (must be > 0)
        type (str): Frame in which the twist is expressed — 'body' or 'spatial'

    Returns:
        np.ndarray: 6D twist vector(s) (linear; angular), shape (6,) or (N,6)
    """
    assert isinstance(X1, np.ndarray) and isinstance(X2, np.ndarray)
    assert X1.shape[-2:] == (4, 4) and X2.shape == X1.shape
    assert dt > 0
    check_SE3(X1)
    check_SE3(X2)

    is_single = X1.ndim == 2
    if is_single:
        X1 = X1[None, ...]
        X2 = X2[None, ...]

    X_T = X2 @ np.linalg.inv(X1)
    twist = log_SE3(X_T) / dt  # Spatial vel

    if type == "body":
        Ad = adjoint(X1)
        twist = np.einsum("nij,nj->ni", Ad, twist)

    return twist[0] if is_single else twist


def twist_to_pose(
    vel: np.ndarray,
    X1: np.ndarray,
    dt: float = 1.0,
    type: Literal["spatial", "body"] = "spatial",
) -> np.ndarray:
    """
    Integrate a 6D twist (v; ω) to compute the next SE(3) pose from a current pose.

    The twist represents motion over a time interval `dt` starting from pose `X1`.

    - If `type == 'body'`:
        Assumes `vel` is a body-frame twist (expressed in the local frame at X1),
        and integrates as:    X2 = X1 @ exp(se3)

    - If `type == 'spatial'`:
        Assumes `vel` is a spatial-frame twist (expressed in the world/base frame),
        and integrates by first converting to a body twist:
            X2 = exp(vel) @ X1

    Args:
        vel (np.ndarray): 6D twist vector(s) (linear; angular), shape (6,) or (N,6)
        X1 (np.ndarray): Initial SE(3) pose(s), shape (4,4) or (N,4,4)
        dt (float): Time step for integration (must be > 0)
        type (str): Frame in which the twist is expressed — 'spatial' or 'body'

    Returns:
        np.ndarray: Integrated SE(3) pose(s), shape (4,4) or (N,4,4)
    """
    assert isinstance(vel, np.ndarray) and vel.shape[-1] == 6
    assert isinstance(X1, np.ndarray) and X1.shape[-2:] == (4, 4)
    assert dt > 0
    check_SE3(X1)

    is_single = vel.ndim == 1
    if is_single:
        vel = vel[None, :]
        X1 = X1[None, ...]

    if type == "body":
        Ad = adjoint(np.linalg.inv(X1))
        vel = np.einsum("nij,nj->ni", Ad, vel)

    T_delta = se3_exp(vel * dt)
    X2 = T_delta @ X1

    check_SE3(X2)

    return X2[0] if is_single else X2


# -------------------------------- To Methods -------------------------------- #
def rotation_angle_and_axis_to_SO3(angle_rad: float, axis: str) -> np.ndarray:
    """Generate a 3x3 rotation matrix for a given angle and rotation axis.

    Implements the Rodrigues' rotation formula for canonical axes (x/y/z).

    Args:
        angle_rad: Rotation angle in radians.
        axis: Rotation axis, must be 'x', 'y', or 'z' (case-sensitive).

    Returns:
        3x3 special orthogonal rotation matrix (SO(3)).
    """
    valid_axes = {"x", "y", "z"}
    if axis not in valid_axes:
        raise ValueError(f"Invalid axis: '{axis}'. Must be one of {valid_axes}.")

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rotation_matrix = np.eye(3)

    if axis == "x":
        rotation_matrix[1:, 1:] = [[cos_a, -sin_a], [sin_a, cos_a]]
    elif axis == "y":
        rotation_matrix[[0, 0, 2, 2], [0, 2, 0, 2]] = [cos_a, sin_a, -sin_a, cos_a]
    elif axis == "z":
        rotation_matrix[:2, :2] = [[cos_a, -sin_a], [sin_a, cos_a]]

    check_SO3(rotation_matrix)
    return rotation_matrix


def SO3_to_6D(X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert an orthonormal SO(3) matrix to a 6-dimensional vector.

    Args:
        X: A numpy array or torch tensor of shape (3, 3) or (N, 3, 3).

    Returns:
        A numpy array or torch tensor of shape (6,) or (N, 6).
    """
    check_SO3(X)
    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 3, 3)

    vec_6D = _float32(_transpose(X[:, :, :2]).reshape(-1, 6))
    return vec_6D if has_batch_dim else vec_6D[0]


def SO3_to_9D(X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert an orthonormal SO(3) matrix to a 9-dimensional vector.

    Args:
        X: A numpy array or torch tensor of shape (3, 3) or (N, 3, 3).

    Returns:
        A numpy array or torch tensor of shape (9,) or (N, 9). [x1, x2, x3, y1, y2, y3, z1, z2, z3]
    """
    check_SO3(X)
    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 3, 3)

    X = _float32(_transpose(X).reshape(-1, 9))
    return X if has_batch_dim else X[0]


def SE3_to_6D_xyz(
    X: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert SE(3) 4x4 transformation matrices to 9D vectors.

    Args:
        X: A numpy array or torch tensor of shape (4, 4) or (N, 4, 4).

    Returns:
        A numpy array or torch tensor of shape (9,) or (N, 9).
    """
    check_SE3(X)
    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4)

    vec_6D = SO3_to_6D(X[:, :3, :3])
    t = X[:, :3, 3]
    vec_6D_xyz = _float32(_cat(vec_6D, t, dim=-1))

    return vec_6D_xyz if has_batch_dim else vec_6D_xyz[0]


def SE3_from_6D_z(
    vec: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 7) if has_batch_dim else vec

    rot = SO3_from_6D(vec[:, :6])

    if isinstance(vec, np.ndarray):
        t = np.zeros((vec.shape[0], 3))
    elif isinstance(vec, torch.Tensor):
        t = torch.zeros((vec.shape[0], 3), device=vec.device)
    t[:, -1] = vec[:, -1]
    X = _make_SE3(rot, t)
    return X if has_batch_dim else X[0]


def SE3_to_6D_z(X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert SE(3) 4x4 transformation matrices to 7D vectors.

    Args:
        X: A numpy array or torch tensor of shape (4, 4) or (N, 4, 4).

    Returns:
        A numpy array of shape (7,) or (N, 7).
    """
    check_SE3(X)

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4) if not has_batch_dim else X

    vec_6D = SO3_to_6D(X[:, :3, :3])
    t = X[:, 2, 3].reshape(-1, 1)
    vec_6D_z = _float32(_cat(vec_6D, t, dim=-1))

    return vec_6D_z if has_batch_dim else vec_6D_z[0]


def SE3_to_9D_xyz(
    X: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert SE(3) 4x4 transformation matrices to 9D vectors.

    Args:
        X: A numpy array or torch tensor of shape (4, 4) or (N, 4, 4).

    Returns:
        Array of shape (9,) or (N, 9) of the same type as the input.
    """
    check_SE3(X)

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4) if not has_batch_dim else X

    rot = SO3_to_9D(X[:, :3, :3])
    t = X[:, :3, 3]
    vec_9D_xyz = _float32(_cat(rot, t, dim=-1))
    return vec_9D_xyz if has_batch_dim else vec_9D_xyz[0]


def SE3_to_9D_z(X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert SE(3) 4x4 transformation matrices to 9D vectors.

    Args:
        X: A numpy array or torch tensor of shape (4, 4) or (N, 4, 4).

    Returns:
        Array of shape (9,) or (N, 9) of the same type as the input.
    """

    check_SE3(X)

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4) if not has_batch_dim else X

    rot = X[:, :3, :3].reshape(-1, 9)
    t = X[:, 2, 3].reshape(-1, 1)
    vec_9D_z = _float32(_cat(rot, t, dim=-1))
    return vec_9D_z if has_batch_dim else vec_9D_z[0]


def SE3_to_rotvec_xyz(
    X: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert SE(3) matrix to rotation vector and translation vector.

    Parameters:
    - X: np.ndarray, SE(3) matrix of shape (4, 4) or batch of SE(3) matrices of shape (B, 4, 4)

    Returns:
    - np.ndarray: rotation vector and translation vector of shape (6,) or (B, 6)
    """
    check_SE3(X)
    has_batch_dim = X.ndim == 3
    if not has_batch_dim:
        X = X.reshape(1, 4, 4)

    backend = "numpy"
    device = None
    if isinstance(X, torch.Tensor):
        backend = "torch"
        device = X.device
        X = X.cpu().numpy()

    rot = X[:, :3, :3]
    rotvec = R.from_matrix(rot).as_rotvec()
    t = X[:, :3, 3]
    vec_rotvec_xyz = _cat(rotvec, t, dim=-1)

    if backend == "torch":
        vec_rotvec_xyz = torch.tensor(vec_rotvec_xyz, device=device)

    vec_rotvec_xyz = _float32(vec_rotvec_xyz)

    return vec_rotvec_xyz if has_batch_dim else vec_rotvec_xyz[0]


def center_SO3(
    rot_mat: Union[np.ndarray, torch.Tensor], rot_home: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Center SO(3) matrices by expressing them in a home frame.

    Args:
        rot_mat (Union[np.ndarray, torch.Tensor]): Input rotation matrices of shape (3, 3) or (N, 3, 3).
        rot_home (Union[np.ndarray, torch.Tensor]): Home frame rotation matrix of shape (3, 3).

    Returns:
        Union[np.ndarray, torch.Tensor]: Centered rotation matrices of shape (3, 3) or (N, 3, 3).
    """
    check_SO3(rot_mat)
    check_SO3(rot_home)

    has_batch_dim = rot_mat.ndim == 3
    rot_mat = rot_mat.reshape(-1, 3, 3)

    centered_rot = _inv(rot_home) @ rot_mat
    centered_rot = _float32(centered_rot)
    return centered_rot if has_batch_dim else centered_rot[0]


def center_SE3(X: np.ndarray, X_home: np.ndarray) -> np.ndarray:
    """
    Center SE(3) matrices by expressing them in a home frame.

    Args:
        X (np.ndarray): Input transformation matrices of shape (4, 4) or (N, 4, 4).
        X_home (np.ndarray): Home frame transformation matrix of shape (4, 4).

    Returns:
        np.ndarray: Centered transformation matrices of shape (4, 4) or (N, 4, 4).
    """
    check_SE3(X)
    check_SE3(X_home)

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4)

    centered_X = _inv(X_home) @ X
    centered_X = _float32(centered_X)
    return centered_X if has_batch_dim else centered_X[0]


def Eular_to_SE3(pose):
    """
    Convert (x, y, z, roll, pitch, yaw) to SE(3) 4x4 matrix.
    Supports both single pose and batch of poses.

    Args:
        pose: numpy array of shape (6,) or (N, 6)

    Returns:
        T: numpy array of shape (4, 4) or (N, 4, 4)
    """
    assert isinstance(pose, np.ndarray)
    assert pose.shape[-1] == 6, f"Expected shape (N,6), got {pose.shape}"

    is_single = pose.ndim == 1
    if is_single:
        pose = pose[None, :]

    assert np.all((pose[:, 3:6] >= -np.pi) & (pose[:, 3:6] <= np.pi)), (
        "Roll, pitch, and yaw must be in [-π, π]"
    )

    rot = R.from_euler("xyz", pose[:, 3:6]).as_matrix()
    T = np.tile(np.eye(4), (pose.shape[0], 1, 1))
    T[:, :3, :3] = rot
    T[:, :3, 3] = pose[:, :3]
    return T[0] if is_single else T


def SE3_to_Eular(T):
    """
    Convert SE(3) 4x4 matrix to (x, y, z, roll, pitch, yaw)
    Supports both single matrix and batch of matrices.

    Args:
        T: numpy array of shape (4, 4) or (N, 4, 4)

    Returns:
        pose: numpy array of shape (6,) or (N, 6)
    """
    assert isinstance(T, np.ndarray)
    assert T.ndim in [2, 3], f"Input T must be (4,4) or (N,4,4), got shape {T.shape}"
    assert T.shape[-2:] == (4, 4), f"Last two dims must be (4,4), got {T.shape[-2:]}"
    is_single = T.ndim == 2
    if is_single:
        T = T[None, :]

    trans = T[:, :3, 3]
    euler = R.from_matrix(T[:, :3, :3]).as_euler("xyz")
    pose = np.hstack([trans, euler])

    assert np.all((pose[:, 3:6] >= -np.pi) & (pose[:, 3:6] <= np.pi)), (
        "Roll, pitch, and yaw must be in [-π, π]"
    )
    return pose[0] if is_single else pose


# ----------------------------------- From ----------------------------------- #


def SO3_from_6D(
    vec: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Recover an orthonormal SO(3) matrix from a 6-dimensional vector using the Gram-Schmidt process.

    Args:
        vec: A numpy array or torch tensor of shape (6,) or (N, 6) where the first 3 elements are the x axis and the next 3 elements are the y axis.

    Returns:
        A numpy array or torch tensor of shape (3, 3) or (N, 3, 3) representing the orthonormal SO(3) matrix.
    """
    if isinstance(vec, np.ndarray):
        backend = "numpy"
    elif isinstance(vec, torch.Tensor):
        backend = "torch"
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")

    assert vec.shape[-1] == 6, "Input must be a 6D vector or a batch of 6D vectors"

    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 6) if not has_batch_dim else vec

    x_axis = vec[:, :3]
    y_axis = vec[:, 3:]

    if backend == "numpy":
        # Normalize the x axis
        x_axis /= np.linalg.norm(x_axis, axis=-1, keepdims=True)

        # Make the y axis orthogonal to the x axis
        y_axis -= np.sum(x_axis * y_axis, axis=-1, keepdims=True) * x_axis
        y_axis /= np.linalg.norm(y_axis, axis=-1, keepdims=True)

        z_axis = np.cross(x_axis, y_axis)

        # Form the orthonormal SO(3) matrix
        rot = np.stack([x_axis, y_axis, z_axis], axis=-1)

    elif backend == "torch":
        # Normalize the x axis
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

        # Make the y axis orthogonal to the x axis
        y_axis = y_axis - torch.sum(x_axis * y_axis, dim=-1, keepdim=True) * x_axis
        y_axis = y_axis / torch.norm(y_axis, dim=-1, keepdim=True)

        z_axis = torch.cross(x_axis, y_axis, dim=-1)

        # Form the orthonormal SO(3) matrix
        rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    check_SO3(rot)
    rot = _float32(rot)
    return rot if has_batch_dim else rot[0]


def SO3_from_9D(
    vec: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert a 9-dimensional vector to an orthonormal SO(3) matrix.

    Args:
        vec: A numpy array or torch tensor of shape (9,) or (N, 9), with [x1, x2, x3, y1, y2, y3, z1, z2, z3].

    Returns:
        A numpy array or torch tensor of shape (3, 3) or (N, 3, 3).
    """
    assert isinstance(vec, np.ndarray) or isinstance(vec, torch.Tensor), (
        "Input must be a numpy array or a torch tensor"
    )

    assert vec.shape[-1] == 9, "Input must be a 9D vector or a batch of 9D vectors"

    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 9)

    # Transpose must be done before reshaping as vec is of [x1, x2, x3, y1, y2, y3, z1, z2, z3]
    rot = _float32(_transpose(vec.reshape(-1, 3, 3)))

    return rot if has_batch_dim else rot[0]


def SE3_from_6D_xyz(
    vec: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 9D vectors to SE(3) 4x4 transformation matrices.

    Args:
        vec: A numpy array or torch tensor of shape (9,) or (N, 9).

    Returns:
        A numpy array or torch tensor of shape (4, 4) or (N, 4, 4).
    """

    assert isinstance(vec, np.ndarray) or isinstance(vec, torch.Tensor), (
        "Input must be a numpy array or a torch tensor"
    )
    assert vec.shape[-1] == 9, "Input must be a 9D vector or a batch of 9D vectors"
    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 9)

    X = _make_SE3(SO3_from_6D(vec[:, :6]), vec[:, 6:])

    return X if has_batch_dim else X[0]


def SE3_from_9D_xyz(
    vec: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 9D vectors to SE(3) 4x4 transformation matrices.

    Args:
        vec: A numpy array or torch tensor of shape (9,) or (N, 9), i.e. [R1.T, R2.T, R3.T, t1, t2, t3].

    Returns:
        A numpy array or torch tensor of shape (4, 4) or (N, 4, 4).
    """

    assert isinstance(vec, np.ndarray) or isinstance(vec, torch.Tensor), (
        "Input must be a numpy array or a torch tensor"
    )
    assert vec.shape[-1] == 12, "Input must be a 12D vector or a batch of 12D vectors"

    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 12)

    X = _make_SE3(SO3_from_9D(vec[:, :9]), vec[:, 9:])
    return X if has_batch_dim else X[0]


def SE3_from_rotvec_xyz(
    vec: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert rotation vector and translation vector to SE(3) matrix.

    Args:
        vec: np.ndarray or torch.Tensor, rotation vector and translation vector of shape (6,) or (N, 6)
    Returns:
        np.ndarray or torch.Tensor, SE(3) matrix of shape (4, 4) or (N, 4, 4)
    """

    assert vec.shape[-1] == 6, "Input must be a 6D vector or a batch of 6D vectors"
    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 6)

    backend = "numpy"
    device = None
    if isinstance(vec, torch.Tensor):
        backend = "torch"
        device = vec.device
        vec = vec.cpu().numpy()

    X = _make_SE3(R.from_rotvec(vec[:, :3]).as_matrix(), vec[:, 3:])
    if backend == "torch":
        X = torch.tensor(X, device=device)
    X = _float32(X)
    return X if has_batch_dim else X[0]


def sixd_to_rotmat(sixd):
    assert sixd.shape[-1] == 6
    # sixd: (..., 6)
    a1 = sixd[..., :3]
    a2 = sixd[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def sixd_xyz_to_se3(sixd_xyz):
    assert sixd_xyz.shape[-1] == 9
    # sixd_xyz: (..., 9)
    R = sixd_to_rotmat(sixd_xyz[..., :6])
    t = sixd_xyz[..., 6:]
    return _make_SE3(R, t)


# ---------------------------------------------------------------------------- #
#                                Sanity Checkers                               #
# ---------------------------------------------------------------------------- #


def check_SO3(mat: Union[np.ndarray, torch.Tensor], atol=1e-5) -> bool:
    """
    Check if the input matrix is a valid SO(3) matrix.

    Parameters:
    - mat: Union[np.ndarray, torch.Tensor], input matrix of shape (3, 3) or batch of such matrices.
    - atol: float, absolute tolerance for numerical checks.

    Returns:
    - bool: True if the input matrix is a valid SO(3) matrix, raises AssertionError otherwise.
    """
    assert isinstance(mat, (np.ndarray, torch.Tensor)), (
        "Input must be a numpy array or torch tensor."
    )
    assert mat.shape[-2:] == (3, 3), "Invalid shape for SO(3) matrix."

    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().numpy()

    mat = mat.reshape(-1, 3, 3)

    # Check if the rotation matrix is orthonormal
    ortho_check = np.allclose(mat @ mat.transpose(0, 2, 1), np.eye(3), atol=atol)
    assert ortho_check, "Invalid SO(3) matrix, rotation matrix must be orthonormal."

    # Check if the determinant is 1
    det_check = np.allclose(np.linalg.det(mat), 1, atol=atol)
    assert det_check, "Invalid SO(3) matrix, rotation matrix determinant must be 1."

    return True


def check_SE3(mat: Union[np.ndarray, torch.Tensor], atol=1e-5, fast=False) -> bool:
    """
    Check if the input matrix is a valid SE(3) matrix.

    Parameters:
    - mat: Union[np.ndarray, torch.Tensor], input matrix of shape (4, 4) or batch of such matrices.
    - atol: float, absolute tolerance for numerical checks.
    - fast: bool, if True, only perform shape and last row checks.

    Returns:
    - bool: True if the input matrix is a valid SE(3) matrix, raises AssertionError otherwise.
    """
    assert isinstance(mat, (np.ndarray, torch.Tensor)), (
        "Input must be a numpy array or torch tensor."
    )

    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().numpy()

    has_batch_dim = mat.ndim == 3
    if not has_batch_dim:
        mat = np.expand_dims(mat, axis=0)

    shape_check = mat.shape[-2:] == (4, 4)
    last_row_check = np.allclose(mat[:, -1, :], np.array([0, 0, 0, 1]), atol=atol)
    fast_check = shape_check and last_row_check

    if fast:
        assert fast_check, (
            f"Shape Check: {shape_check}, Last Row Check: {last_row_check}"
        )
        return True

    # Check the rotation matrix part
    check_SO3(mat[:, :3, :3], atol=atol)

    return True


def check_pts(pts: Union[np.ndarray, torch.Tensor]) -> bool:
    """
    Check if the input points are valid.

    Parameters:
    - pts: Union[np.ndarray, torch.Tensor], input points of shape (N, 3) or (N, 4).

    Returns:
    - bool: True if the input points are valid, raises AssertionError otherwise.
    """
    assert isinstance(pts, (np.ndarray, torch.Tensor)), (
        "Input must be a numpy array or torch tensor."
    )

    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()

    assert pts.ndim <= 2, "Input must be a 1D or 2D array."
    assert pts.shape[-1] in [3, 4], "Invalid shape for points."

    if pts.shape[-1] == 4:
        last_col_check = np.allclose(pts[:, -1], 1)
        assert last_col_check, "Invalid homogeneous coordinates for points."

    is_homog = pts.shape[-1] == 4
    return is_homog


def check_so3(X: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Check whether a matrix or batch of matrices belongs to the so(3) Lie algebra.

    A matrix is in so(3) if it is 3x3 and skew-symmetric:
        X^T = -X

    Args:
        X (np.ndarray): A (3,3) or (N,3,3) matrix or batch of matrices.
        atol (float): Absolute tolerance for skew-symmetry check.

    Returns:
        bool: True if all matrices in the batch are valid so(3) elements.
    """
    assert isinstance(X, np.ndarray), f"Input must be np.ndarray, got {type(X)}"
    if X.ndim == 2:
        X = X[None, ...]
    assert X.shape[1:] == (3, 3), f"Expected shape (3,3) or (N,3,3), but got {X.shape}"

    return np.allclose(X, -X.transpose(0, 2, 1), atol=atol)


def check_se3(X: Union[np.ndarray, torch.Tensor], atol=1e-5) -> bool:
    """
    Check if a 4x4 matrix is a valid element of se(3) Lie algebra
    Args:
        X: 4x4 matrices or batch of such matrices
        atol: tolerance for numerical comparisons
    Returns:
        True if X ∈ se(3), else False
    """
    if not isinstance(X, (np.ndarray, torch.Tensor)):
        return False

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    if X.ndim == 2 and X.shape == (4, 4):
        X = X[None, ...]  # make batch of 1
    elif X.ndim != 3 or X.shape[1:] != (4, 4):
        return False

    # 1. Check so3 of top-left 3x3 block
    skew_block = X[:, :3, :3]
    is_skew = check_so3(skew_block)

    # 2. Check last row is all zeros
    last_row = X[:, 3, :]
    is_last_row_zero = np.all(np.abs(last_row) < atol)

    return is_skew and is_last_row_zero


# ---------------------------------------------------------------------------- #
#                         Backend-compatible Operations                        #
# ---------------------------------------------------------------------------- #


def _cat(
    x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], dim: int
):
    """
    Concatenate two numpy arrays or torch tensors along a specified dimension.
    """
    if isinstance(x, np.ndarray):
        return np.concatenate([x, y], axis=dim)
    elif isinstance(x, torch.Tensor):
        return torch.cat([x, y], dim=dim)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")


def _transpose(x: Union[np.ndarray, torch.Tensor]):
    """
    Transpose the last two dimensions of a numpy array or torch tensor.
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            return x.T
        elif x.ndim == 3:
            return x.transpose(0, 2, 1)
    elif isinstance(x, torch.Tensor):
        return x.transpose(-2, -1)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")


def _float32(x: Union[np.ndarray, torch.Tensor]):
    """
    Convert a numpy array or torch tensor to float32.
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        return x.float()
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")


def _inv(x: Union[np.ndarray, torch.Tensor]):
    """
    Compute the inverse of a numpy array or torch tensor.
    """
    if isinstance(x, np.ndarray):
        return np.linalg.inv(x)
    elif isinstance(x, torch.Tensor):
        return x.inverse()
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")


def _make_SE3(rot: Union[np.ndarray, torch.Tensor], t: Union[np.ndarray, torch.Tensor]):
    """
    Create an SE(3) matrix from rotation matrix and translation vector.

    Args:
        rot: Rotation matrix of shape (3, 3) or (N, 3, 3).
        t: Translation vector of shape (3,) or (N, 3).

    Returns:
        SE(3) matrix of shape (4, 4) or (N, 4, 4).
    """
    B1 = rot.shape[0]
    B2 = t.shape[0]
    assert B1 == B2, "Batch sizes must match for rotation and translation."

    if isinstance(rot, np.ndarray):
        X = np.eye(4).reshape(1, 4, 4).repeat(B1, axis=0)
    elif isinstance(rot, torch.Tensor):
        X = torch.eye(4).reshape(1, 4, 4).repeat(B1, 1, 1).to(rot.device)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")

    has_batch_dim = rot.ndim == 3
    X[:, :3, :3] = rot
    X[:, :3, 3] = t
    check_SE3(X)
    X = _float32(X)
    return X if has_batch_dim else X[0]


def _all_close(
    x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], atol=1e-5
):
    """
    Check if two numpy arrays or torch tensors are element-wise equal within a tolerance.

    Args:
        x: First input array or tensor.
        y: Second input array or tensor.
        atol: Absolute tolerance.

    Returns:
        bool: True if the arrays or tensors are element-wise equal within the tolerance, False otherwise.
    """
    if isinstance(x, np.ndarray):
        return np.allclose(x, y, atol=atol)
    elif isinstance(x, torch.Tensor):
        return torch.allclose(x, y, atol=atol)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    atol = 1e-5
    print("Running tests...")
    B = 1000
    # Test SO3_to_6D
    rot = Rotation.random(B).as_matrix().astype(np.float32)
    X = _make_SE3(rot, np.random.randn(B, 3))

    test_backend = ["numpy", "torch"]

    for backend in test_backend:
        print(f"Testing backend: {backend}")
        if backend == "torch":
            rot = torch.tensor(rot).float()
            X = torch.tensor(X).float()

        rot6d = SO3_to_6D(rot)
        # Test SO3_from_6D
        rot_recovered = SO3_from_6D(rot6d)
        assert _all_close(rot, rot_recovered, atol=atol), (
            "SO3_to_6D and SO3_from_6D are not consistent"
        )
        print("[Passed] SO3_to_6D and SO3_from_6D are consistent")
        # test torch and numpy consistency
        if isinstance(rot6d, np.ndarray):
            rot6d_torch = torch.tensor(rot6d)
            rot_recovered_torch = SO3_from_6D(rot6d_torch)
            assert torch.allclose(
                rot_recovered_torch, torch.tensor(rot_recovered), atol=atol
            ), "SO3_from_6D is not consistent with numpy"
            print("[Passed] SO3_from_6D torch and numpy are consistent")
        # Test SE3_to_6D_xyz

        vec = SE3_to_6D_xyz(X)
        # Test SE3_from_6D_xyz
        X_recovered = SE3_from_6D_xyz(vec)
        assert _all_close(X, X_recovered, atol=atol), (
            "SE3_to_6D_xyz and SE3_from_6D_xyz are not consistent"
        )
        print("[Passed] SE3_to_6D_xyz and SE3_from_6D_xyz are consistent")
        # Test SE3_to_9D_xyz
        vec = SE3_to_9D_xyz(X)
        # Test SE3_from_9D_xyz
        X_recovered = SE3_from_9D_xyz(vec)
        assert _all_close(X, X_recovered, atol=atol), (
            "SE3_to_9D_xyz and SE3_from_9D_xyz are not consistent"
        )
        print("[Passed] SE3_to_9D_xyz and SE3_from_9D_xyz are consistent")

        # Test SE3_to_rotvec_xyz
        vec = SE3_to_rotvec_xyz(X)
        # Test SE3_from_rotvec_xyz
        X_recovered = SE3_from_rotvec_xyz(vec)
        assert _all_close(X, X_recovered, atol=atol), (
            "SE3_to_rotvec_xyz and SE3_from_rotvec_xyz are not consistent"
        )
        print("[Passed] SE3_to_rotvec_xyz and SE3_from_rotvec_xyz are consistent")
