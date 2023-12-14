from pathlib import Path
from typing import Any, Generator, List, Tuple, Union
import numpy as np
import re
import numpy as np
import torch
import cv2


def opengl_projection_transform(opengl_projection_matrix, width, height, points):
    if isinstance(points, np.ndarray):
        points = np.concatenate([points, np.ones((points.shape[0], 1))], -1)
    elif isinstance(points, torch.Tensor):
        shape = points.shape[:-1]
        device = points.device
        
        points = torch.cat([points, torch.ones(*shape, 1).to(device)], -1)

    shape = points.shape[:-1]

    if isinstance(points, np.ndarray):
        points = np.reshape(points, (-1, 4))
    else:
        points = points.view(-1, 4)

    h_points = (opengl_projection_matrix @ points.T).T 
    h_points = h_points / h_points[:, -1:]

    h_points = (1 - h_points) * 0.5
    h_points[:, 0] *= width
    h_points[:, 1] *= height

    if isinstance(h_points, np.ndarray):
       h_points = np.reshape(h_points, (*shape, 4))
    else:
        h_points = h_points.view(*shape, 4)

    return h_points[..., :2]


def opencv_projection_transform(cv2_projection_matrix, points):
    points = np.concatenate([points, np.ones_like(points)[..., :1]], -1)
    shape = points.shape[:-1]

    points = np.reshape(points, (-1, 4))

    h_points = (cv2_projection_matrix @ points.T).T 
    h_points /= h_points[:, -1:]

    h_points = np.reshape(h_points, (*shape, 3))
    h_points = h_points[..., :2]

    return h_points


def opencv_camera_view_to_screen_space_transform(camera_matrix, camera_view_points):
    points = camera_view_points

    # points = np.concatenate([points, np.ones_like(points)[..., :1]], -1)
    shape = points.shape[:-1]

    points = np.reshape(points, (-1, 3))

    ss_points = (camera_matrix @ points.T).T 
    ss_points /= ss_points[:, -1:]

    ss_points = np.reshape(ss_points, (*shape, 3))
    ss_points = ss_points[..., :2]

    return ss_points


def opencv_global_view_to_camera_view_transform(extrinsic, global_3d_points):
    points = global_3d_points

    points = np.concatenate([points, np.ones_like(points)[..., :1]], -1)
    shape = points.shape[:-1]

    points = np.reshape(points, (-1, 4))

    camera_view_points = np.matmul(extrinsic, points.T).T
    camera_view_points = np.reshape(camera_view_points, (*shape, 3))
    # ss_points = ss_points[..., :3]

    return camera_view_points


def extract_params(
    lines: List, idx: int, resolution: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the extrinsic, intrinsic, and distortion parameters of a camera.

    Returns:
        R_mat (np.ndarray): 3x3 matrix of rotation extrinsic parameters.
        T_vec (np.ndarray): 3x3 matrix of translation extrinsic parameters.
        intrinsics (np.ndarray): 5 vector of instric parameters and distortion coefficients.
            1-2th elements are focal lengths.
            3-4th elements are distortion coeffs.
            5-6th elements are principal point.
    """
    distortion_coeffs = np.fromstring(lines[idx + 11][15:], dtype=np.float64, sep="\t")
    extrinsic = np.array(
        [
            np.fromstring(lines[idx + j][1:], dtype=np.float64, sep="\t")
            for j in [16, 17, 18]
        ],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [
            np.fromstring(lines[idx + j][1:], dtype=np.float64, sep="\t")
            for j in [20, 21, 22]
        ],
        dtype=np.float64,
    )
        
    w, h = resolution

    focals = np.diag(intrinsic)[:2] * w  # in px
    dist = distortion_coeffs
    principal_pt = intrinsic[:2, 2].ravel() * w  # in px

    pixelAspect = np.fromstring(lines[idx + 8][15:15+11], dtype=np.float64, sep="\t")[0]
    
    fx, fy = focals
    cx, cy = principal_pt

    mtx = np.array([[fx, 0 , cx],
                    [0 , fy * pixelAspect, cy],
                    [0 , 0 , 1 ]])
    
    return mtx, dist, extrinsic 


def create_cv2_camera(file_path: Union[str, Path], image_shape: Union[np.ndarray, Tuple], camera_index: int):
    with open(file_path) as f:
        lines = f.readlines()

    starting_idx = []
    camera_idx = []
    for idx, line in enumerate(lines):
        match = re.search(r'camera.+(\d{1,})\s.*\.avi', line)
        if match:
            starting_idx.append(idx)
            camera_idx.append(int(match.group(1)))

    starting_idx = starting_idx[camera_index]

    camera_matrix, dist, extrinsic = extract_params(lines, starting_idx, image_shape)
    
    projection_matrix = np.matmul(camera_matrix, extrinsic)

    return camera_matrix, dist, extrinsic, projection_matrix


def undistort(xy, mtx, dist, width, height):
    und = cv2.undistortPoints(xy, mtx, dist)
    und = und.reshape(-1, 2)
    und = np.c_[und, np.ones_like(und[:,0])] @ mtx.T
    assert(np.allclose(und[:, 2], np.ones_like(und[:, 2])))
    und = und[:, :2]

    und[:, 0] = np.clip(und[:, 0], 0, width-1)
    und[:, 1] = np.clip(und[:, 1], 0, height-1)
    assert(np.all(0 <= und) and np.all(und[:, 0] < width) and np.all(und[:, 1] < height))
    
    return und
