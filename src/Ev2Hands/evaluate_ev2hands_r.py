import sys; sys.path.append('..') # add settings.
import os
os.environ['ERPC'] = '1'
import torch
import pickle
import numpy as np
import trimesh

import arg_parser 

from sklearn import metrics as skmetrics
from torch.utils.data import DataLoader
from model import TEHNetWrapper    
from evaluate import right_root_relative_pck3d_frame, absolute_pck3d_frame, relative_pck3d_frame
from camera import *
from mesh_intersection.bvh_search_tree import BVH

from dataset import ERPCParser
from settings import REAL_TEST_DATA_PATH


SAVE_NAME = 'Ev2Hands'
aedat_input_paths = [
    ['1', f'{REAL_TEST_DATA_PATH}/subject_1_event.pickle'],
    ['2', f'{REAL_TEST_DATA_PATH}/subject_2_event.pickle'],
    ['5', f'{REAL_TEST_DATA_PATH}/subject_5_event.pickle'],
    ['6', f'{REAL_TEST_DATA_PATH}/hands_2_event.pickle'],
    ['7', f'{REAL_TEST_DATA_PATH}/hands_4_event.pickle']

]

os.makedirs('outputs', exist_ok=True)


def get_auc(pck3d):
    auc = skmetrics.auc(range(pck3d.shape[0]), pck3d) / pck3d.shape[0]
    auc = round(auc, 3)

    return auc



def mepj_frame(joints_pred, joints_gt, num_steps=100, dist_max_mm=100):    
    # # relative to root joint
    joints_pred = joints_pred - joints_pred[:, :1, :]
    joints_gt = joints_gt - joints_gt[:, :1, :]

    joints_pred = torch.cat([joints_pred[0], joints_pred[1]], 0) 
    joints_gt = torch.cat([joints_gt[0], joints_gt[1]], 0) 
    
    # compute distances
    dists = torch.norm(joints_pred - joints_gt, p=2, dim=1).mean()

    return dists



def evaluate_joints_real(j3d_pred, j3d_gts, num_steps):    
    aucs = list()
    for idx, j3d_gt in enumerate(j3d_gts):
        absolute_pck3d = absolute_pck3d_frame(j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=100)
        relative_pck3d = relative_pck3d_frame(j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=100)
        right_root_relative_pck3d = right_root_relative_pck3d_frame(j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=100)

        absolute_auc = get_auc(absolute_pck3d)
        relative_auc = get_auc(relative_pck3d)
        right_root_relative_auc = get_auc(right_root_relative_pck3d)

        aucs.append(right_root_relative_auc)
    
    
    idx = np.argmax(aucs)

    j3d_gt = j3d_gts[idx]
    absolute_pck3d = absolute_pck3d_frame(j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=100)
    relative_pck3d = relative_pck3d_frame(j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=100)
    right_root_relative_pck3d = right_root_relative_pck3d_frame(j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=100)
    
    joint_loss = mepj_frame(j3d_pred, j3d_gt).item()
        
    root_distance = [torch.norm((j3d_gt[0] - j3d_gt[1]), p=2, dim=-1).min(-1)[0].cpu().numpy().tolist()]

    return {
        'root_distance': root_distance,
        'joint_loss': joint_loss,
        'absolute_pck3d': absolute_pck3d,
        'relative_pck3d': relative_pck3d,
        'right_root_relative_pck3d': right_root_relative_pck3d,
    }

def evaluate_real(net, device, batch, num_steps):
    net.eval()

    data = batch['data'].to(device=device, dtype=torch.float32)

    with torch.no_grad():
        outputs = net(data)

    N = data.shape[0] 

       
    frames = list()
    for idx in range(N):
        hands = dict()

        hands['left'] = {
            'vertices': outputs['left']['vertices'][idx].cpu(),
            'j3d': outputs['left']['j3d'][idx].cpu(),
        }

        hands['right'] = {
            'vertices': outputs['right']['vertices'][idx].cpu(),
            'j3d': outputs['right']['j3d'][idx].cpu(),
        }

        j3d_pred = torch.cat([hands['left']['j3d'][None, ...], hands['right']['j3d'][None, ...]])
        j3d_gts = batch['j3d'][idx]
        
        scores = evaluate_joints_real(j3d_pred * 1000, # meter to mm 
                                      j3d_gts * 1000, # meter to mm 
                                      num_steps)
        hands['scores'] = scores 
        frames.append(hands)

    return frames

    
def compute_non_collision_score(verts_left_pred, faces_left, verts_right_pred, faces_right):
    B = verts_left_pred.shape[0]

    max_collisions = 8 
    search_tree = BVH(max_collisions=max_collisions)
    
    meshes = list()
    non_collision_scores = list()
    for i in range(B):
        ml = trimesh.Trimesh(verts_left_pred[i].cpu().numpy() * 1000, faces_left)
        mr = trimesh.Trimesh(verts_right_pred[i].cpu().numpy() * 1000, faces_right)

        mesh = trimesh.util.concatenate([ml, mr])
        meshes.append(mesh)

        verts_tensor = torch.tensor(mesh.vertices)[None, ...].cuda()
        face_tensor = torch.tensor(mesh.faces)[None, ...].cuda()
                            
        triangles = verts_tensor.view([-1, 3])[face_tensor]

        with torch.no_grad():                 
            outputs = search_tree(triangles)

        outputs = outputs.squeeze()
        collisions = outputs[outputs[:, 0] >= 0, :]

        n_collisions = collisions.shape[0]
        percentange_of_collisions = n_collisions / triangles.shape[1] * 100
        percentange_of_collisions = round(percentange_of_collisions, 2)

        non_collision_scores.append(100 - percentange_of_collisions)
        
    return non_collision_scores, meshes


def main():
    arg_parser.evaluate()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = TEHNetWrapper(device=device)

    save_path = os.environ['CHECKPOINT_PATH']
    batch_size = os.environ['BATCH_SIZE'] 

    print(f"Loading checkpoint from {save_path}")

    checkpoint = torch.load(save_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
            
    mano_hands = net.hands

    batch_size = 128
    num_workers = os.cpu_count()
    
    num_steps = 100
    
    for subject_idx, aedat_input_path in aedat_input_paths:
        demo_dataset = ERPCParser(aedat_input_path)
        validation_loader = DataLoader(demo_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        joint_loss = 0
        absolute_pck3d = np.zeros(num_steps + 1)
        relative_pck3d = np.zeros(num_steps + 1)
        right_root_relative_pck3d = np.zeros(num_steps + 1)
        non_collision_score = []
        root_distance = []

        frame_index = 1
        subject_scores = []


        for bdx, batch in enumerate(validation_loader):
            frames = evaluate_real(net=net, device=device, batch=batch, num_steps=num_steps)

            for idx, frame in enumerate(frames):
                jframe_idx = batch['frame_index'][idx]

                scores = frame['scores']

                root_distance += scores['root_distance']

                absolute_pck3d += scores['absolute_pck3d']
                relative_pck3d += scores['relative_pck3d']
                right_root_relative_pck3d += scores['right_root_relative_pck3d']
                joint_loss += scores['joint_loss']

                faces_left = mano_hands['left'].faces
                faces_right = mano_hands['right'].faces

                verts_left_pred = frame['left']['vertices'][None, ...]
                verts_right_pred = frame['right']['vertices'][None, ...]

                ncs, meshes = compute_non_collision_score(verts_left_pred, faces_left, verts_right_pred, faces_right)
                non_collision_score += ncs

                subject_scores.append([scores, ncs, jframe_idx])

                absolute_auc = get_auc(absolute_pck3d / frame_index)
                relative_auc = get_auc(relative_pck3d / frame_index)
                right_root_relative_auc = get_auc(right_root_relative_pck3d / frame_index)

                print(f'frame index: {frame_index}', f'relative_auc: {relative_auc}', f'right_root_relative_auc: {right_root_relative_auc} joint_loss: {joint_loss / frame_index} non_collision_score: {np.mean(non_collision_score)}')
            
                frame_index += 1

        print()

        with open(f'outputs/{SAVE_NAME}_subject_{subject_idx}_scores.pickle', 'wb') as pickle_file:
            pickle.dump(subject_scores, pickle_file)


        joint_loss /= frame_index
        absolute_pck3d /= frame_index
        relative_pck3d /= frame_index
        right_root_relative_pck3d /= frame_index

        absolute_auc = get_auc(absolute_pck3d)
        relative_auc = get_auc(relative_pck3d)
        right_root_relative_auc = get_auc(right_root_relative_pck3d)
        
        auc = relative_auc

        metrics = {
            'joint_loss': joint_loss,
            'pck3d': {
                        'absolute': absolute_pck3d,
                        'relative': relative_pck3d,
                        'right_root_relative': right_root_relative_pck3d,
            },
            'auc': {
                        'relative': relative_auc,
                        'absolute': absolute_auc,
                        'right_root_relative': right_root_relative_auc,
            },
            'non_collision_score': non_collision_score,
            'root_distance': root_distance,
            'frame_index': frame_index
        }
    
        np.save(f'./outputs/{SAVE_NAME}_real_{subject_idx}_metrics.npy', metrics)

        print(f'absolute_auc: {absolute_auc}', f'relative_auc: {relative_auc}', f'right_root_relative_auc: {right_root_relative_auc}')



if __name__ == '__main__':
    main()
