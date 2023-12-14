import numpy as np
import torch
from dataset.utils import interpolate_sequence, mano_data_to_mano_sequence
from settings import MANO_PATH, DATA_PATH, MANO_CMPS, OUTPUT_WIDTH, OUTPUT_HEIGHT
import smplx
import warnings
import trimesh
import random
from concurrent.futures import ProcessPoolExecutor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment_mano_sequence(mano_data):
    x = (2 * np.random.rand(3) - 1) * 0.1

    keys = list(mano_data['mano_sequence'].keys())

    for key in keys:
        mano_params = mano_data['mano_sequence'][key]

        transformed_mano_params = list()
        for mano_param in mano_params:
            mano_param['trans'] = x + np.array(mano_param['trans'], dtype=np.float32)
            transformed_mano_params.append(mano_param)

        mano_data['mano_sequence'][key] = transformed_mano_params

    return mano_data


def intersection_worker_fn(hands):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        layer = {
                'right': smplx.create(MANO_PATH, 'mano', use_pca=True, num_pca_comps=MANO_CMPS, is_rhand=True), 
                'left': smplx.create(MANO_PATH, 'mano', use_pca=True, num_pca_comps=MANO_CMPS, is_rhand=False)
                }

    if len(hands) != 2: return hands

    meshes = list()
    for mano_param in hands:
        hand_type = mano_param['hand_type']

        pose = mano_param['pose']
        shape = mano_param['shape']
        trans = mano_param['trans']

        pose = torch.tensor(pose[None, :], dtype=torch.float32) 
        shape = torch.tensor(shape[None, :], dtype=torch.float32) 
        trans = torch.tensor(trans[None, :], dtype=torch.float32)

        global_orient = pose[:, :3] 
        hand_pose = pose[:, 3:3+MANO_CMPS]

        output = layer[hand_type](global_orient=global_orient, hand_pose=hand_pose, betas=shape, transl=trans)
        verts = output.vertices[0].cpu().numpy() * 1000
        faces = layer[hand_type].faces

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        meshes.append(mesh)
        
    intersection = trimesh.boolean.intersection(meshes)
    if intersection.area < 1000:
        ...
    else:
        hands.pop(random.choice([0, 1]))

    return hands


def clean_intersections(mano_sequence):
    frame_indices = list(mano_sequence.keys())
 
    with ProcessPoolExecutor() as workers:
        for fdx, hands in zip(frame_indices, workers.map(intersection_worker_fn, [mano_sequence[fdx] for fdx in frame_indices])):
            mano_sequence[fdx] = hands
    
    return mano_sequence

