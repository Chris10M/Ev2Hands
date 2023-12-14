import torch
import os
import numpy as np
import traceback

from mesh_intersection.bvh_search_tree import BVH
from .logger import Logger, set_model_logger



def word_to_screen_transform(cameras, points, img_w, img_h, **kwargs):
    verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(points)
    # view to NDC transform
    to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
    projection_transform = cameras.get_projection_transform(**kwargs).compose(
        to_ndc_transform
    )
    verts_ndc = projection_transform.transform_points(verts_view)
    verts_ndc[..., 2] = verts_view[..., 2]

    verts_ndc = torch.clamp_(verts_ndc, -1, 1)
    N = verts_ndc.shape[0]

    device = verts_ndc.device
    if isinstance(img_h, (int, float)):
        img_h = torch.tensor(img_h, device=device).repeat(N, 1)           
    if isinstance(img_w, (int, float)):
        img_w = torch.tensor(img_w, device=device).repeat(N, 1)           

    screen_space = torch.zeros_like(verts_ndc)
    screen_space[:, :, 0] = (1.0 - verts_ndc[:, :, 0]) * 0.5 * img_w.reshape(-1, 1)
    screen_space[:, :, 1] = (1.0 - verts_ndc[:, :, 1]) * 0.5 * img_h.reshape(-1, 1)

    return screen_space[:, :, :2]


def visualize_lnes(lnes):
    lnes = lnes.permute(1, 2, 0).cpu().numpy()
    h, w = lnes.shape[:2]

    g = np.zeros((h, w, 1), dtype=lnes.dtype)
    b = lnes[:, :, :1]
    r = lnes[:, :, 1:]

    lnes = np.concatenate([b, g, r], axis=2) * 255
    lnes = np.clip(lnes, 0, 255)
    lnes = lnes.astype(dtype=np.uint8)
    
    return lnes


def get_gpu_memory_usage():
    import nvidia_smi
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    message = f"Total memory: {info.total * 1e-6}MB | Free memory: {info.free * 1e-6}MB | Used memory: {info.used * 1e-6}MB"
    nvidia_smi.nvmlShutdown()

    return message


def load_network(saved_path, net, optim, device, print_fn):
    max_eval_score = 0
    start_it = 0
    
    if os.path.isfile(saved_path):
            saved_model = torch.load(saved_path, map_location=device)
            if 'state_dict' in saved_model: 
                state_dict = saved_model['state_dict']

            try:
                net.load_state_dict(state_dict, strict=False)
            except RuntimeError as e:
                traceback.print_exc()
            
            try:
                start_it = 0
                start_it = saved_model['start_it'] + 2
            except KeyError:
                start_it = 0

            try:
                max_eval_score = saved_model['max_eval_score']
            except KeyError as e: 
                print_fn(e)

            try:
                optim.load_state_dict(saved_model['optimize_state'])
                ...
            except Exception as e: 
                traceback.print_exc()


            print_fn(f'Model Loaded: {saved_path} @ start_it: {start_it}')
    else:
        print_fn(f'No Checkpoint path, training model from scratch.')

    return net, optim, start_it, max_eval_score


def compute_collision_percentage(mesh):
    max_collisions = 8 
    search_tree = BVH(max_collisions=max_collisions)

    verts_tensor = torch.tensor(mesh.vertices)[None, ...].cuda()
    face_tensor = torch.tensor(mesh.faces)[None, ...].cuda()
                          
    triangles = verts_tensor.view([-1, 3])[face_tensor]

    with torch.no_grad():                 
        outputs = search_tree(triangles)

    outputs = outputs.squeeze()
    collisions = outputs[outputs[:, 0] >= 0, :]

    n_collisions = collisions.shape[0]
    percentange_of_collisions = n_collisions / triangles.shape[1] * 100

    return round(percentange_of_collisions, 2)
