import sys; sys.path.append('..') # add settings.
import os
import multiprocessing

from sklearn import metrics
import trimesh
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import arg_parser
from model import TEHNetWrapper
from dataset import Ev2HandSDataset
from settings import ROOT_TRAIN_DATA_PATH


def batch_registration_transform(points, matrix, translate=True):
    """
    Returns points rotated by a homogeneous
    transformation matrix.
    If points are (n, 2) matrix must be (3, 3)
    If points are (n, 3) matrix must be (4, 4)
    Parameters
    ----------
    points : (n, D) float
      Points where D is 2 or 3
    matrix : (3, 3) or (4, 4) float
      Homogeneous rotation matrix
    translate : bool
      Apply translation from matrix or not
    Returns
    ----------
    transformed : (n, d) float
      Transformed points
    """
    device = points.device
    
    points = points.double().to(device)
    matrix = matrix.double().to(device)
    
    n_batch = matrix.shape[0]
    n_points = points.shape[1]
    
    # check to see if we've been passed an identity matrix
    identity = torch.abs(matrix - torch.eye(matrix.shape[1], device=device).repeat(n_batch, 1, 1)).max()
    if identity < 1e-8:
        return points.contiguous()
        
    dimension = points.shape[2]
    
    column = torch.zeros(n_batch, n_points, 1, device=device) + int(bool(translate))    
    stacked = torch.cat((points, column), 2)
    
    transformed = torch.bmm(matrix, stacked.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :dimension]
    transformed = transformed.contiguous()
    
    return transformed.float()


def registration_transform(points, matrix, translate=True):
    """
    Returns points rotated by a homogeneous
    transformation matrix.
    If points are (n, 2) matrix must be (3, 3)
    If points are (n, 3) matrix must be (4, 4)
    Parameters
    ----------
    points : (n, D) float
      Points where D is 2 or 3
    matrix : (3, 3) or (4, 4) float
      Homogeneous rotation matrix
    translate : bool
      Apply translation from matrix or not
    Returns
    ----------
    transformed : (n, d) float
      Transformed points
    """
    device = points.device
    
    points = points.double().to(device)
    # points = np.asanyarray(points, dtype=np.double64)
    
    # no points no cry
    if len(points) == 0:
        return points.clone()

    matrix = matrix.double().to(device)
    # matrix = np.asanyarray(matrix, dtype=np.float64)

    if (len(points.shape) != 2 or
            (points.shape[1] + 1 != matrix.shape[1])):
        raise ValueError('matrix shape ({}) doesn\'t match points ({})'.format(
            matrix.shape,
            points.shape))
    
    # check to see if we've been passed an identity matrix
    identity = torch.abs(matrix - torch.eye(matrix.shape[0], device=device)).max()
    if identity < 1e-8:
        return points.contiguous()
        
    dimension = points.shape[1]
    
    column = torch.zeros(len(points), device=device) + int(bool(translate))    
    stacked = torch.column_stack((points, column))
    
    transformed = (matrix @ stacked.T).T[:, :dimension]
    transformed = transformed.contiguous()
    
    return transformed.float()


def register_mano_mesh_to_template(mano_kpts, template_kpts):
    device = mano_kpts.device

    # The transformation matrix sending a to b
    a = mano_kpts.detach().cpu().numpy() 
    b = template_kpts.detach().cpu().numpy()

    initial = np.identity(4)
    threshold = 1e-5
    max_iterations = 20


    # # transform a under initial_transformation
    # a = transform_points(a, initial)
    total_matrix = initial

    # start with infinite cost
    old_cost = np.inf

    # avoid looping forever by capping iterations
    for n_iteration in range(max_iterations):
        # align a with closest points
        matrix, transformed, cost = trimesh.registration.procrustes(a=a, b=b)

        # update a with our new transformed points
        a = transformed
        total_matrix = np.dot(matrix, total_matrix)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    total_matrix = torch.from_numpy(total_matrix).to(device)
    mano_kpts = registration_transform(mano_kpts, total_matrix)
        
    return mano_kpts, template_kpts, cost


# palm-normalized 2d-pck
def pck2dp_frame(joints_pred, joints_gt, num_steps=100):
    # palm lengths
    len_palm_right_gt = np.linalg.norm(joints_gt[9, :] - joints_gt[0, :])
    len_palm_left_gt = np.linalg.norm(joints_gt[30, :] - joints_gt[21, :])

    # relative to root joints
    joints_pred[0:21, :] = joints_pred[0:21, :] - joints_pred[0, :]
    joints_pred[21:42, :] = joints_pred[21:42, :] - joints_pred[21, :]
    joints_gt[0:21, :] = joints_gt[0:21, :] - joints_gt[0, :]
    joints_gt[21:42, :] = joints_gt[21:42, :] - joints_gt[21, :]

    # remove root joints
    joints_pred = np.concatenate((joints_pred[1:21, :], joints_pred[22:42, :]), 0)
    joints_gt = np.concatenate((joints_gt[1:21, :], joints_gt[22:42, :]), 0)

    # compute distances
    dists_right = np.linalg.norm(joints_pred[0:20, :] - joints_gt[0:20, :], axis=1)
    dists_left = np.linalg.norm(joints_pred[20:40, :] - joints_gt[20:40, :], axis=1)
    pck = np.zeros(num_steps + 1)

    for s in range(num_steps + 1):
        dist_right_s = len_palm_right_gt * s / num_steps
        dist_left_s = len_palm_left_gt * s / num_steps
        pck[s] += (dists_right < dist_right_s).sum()
        pck[s] += (dists_left < dist_left_s).sum()

    pck /= 40

    return pck

def absolute_pck3d_frame(joints_pred, joints_gt, num_steps=100, dist_max_mm=100):
    joints_pred = torch.cat([joints_pred[0], joints_pred[1]], 0) 
    joints_gt = torch.cat([joints_gt[0], joints_gt[1]], 0) 
    
    # compute distances
    dists = torch.norm(joints_pred - joints_gt, p=2, dim=1)
    
    pck = np.zeros(num_steps + 1)
    for s in range(num_steps + 1):
        dist_s = (dist_max_mm / num_steps) * s 
        pck[s] = (dists < dist_s).float().mean()

    return pck


def relative_pck3d_frame(joints_pred, joints_gt, num_steps=100, dist_max_mm=100):        
    # # relative to root joint
    joints_pred = joints_pred - joints_pred[:, :1, :]
    joints_gt = joints_gt - joints_gt[:, :1, :]

    joints_pred = torch.cat([joints_pred[0], joints_pred[1]], 0) 
    joints_gt = torch.cat([joints_gt[0], joints_gt[1]], 0) 
    
    # compute distances
    dists = torch.norm(joints_pred - joints_gt, p=2, dim=1)
    
    pck = np.zeros(num_steps + 1)
    for s in range(num_steps + 1):
        dist_s = (dist_max_mm / num_steps) * s 
        pck[s] = (dists < dist_s).float().mean()

    return pck


def right_root_relative_pck3d_frame(joints_pred, joints_gt, num_steps=100, dist_max_mm=100):
    joints_pred = joints_pred - joints_pred[1:, :1, :] # 1: - select right hand
    joints_gt = joints_gt - joints_gt[1:, :1, :] # 1: - select right hand

    joints_pred = torch.cat([joints_pred[0], joints_pred[1]], 0) 
    joints_gt = torch.cat([joints_gt[0], joints_gt[1]], 0) 
    
    # compute distances
    dists = torch.norm(joints_pred - joints_gt, p=2, dim=1)
    
    pck = np.zeros(num_steps + 1)
    for s in range(num_steps + 1):
        dist_s = (dist_max_mm / num_steps) * s 
        pck[s] = (dists < dist_s).float().mean()

    return pck


def get_auc(pck3d):
    auc = metrics.auc(range(pck3d.shape[0]), pck3d) / pck3d.shape[0]
    auc = round(auc, 2)

    return auc

# evaluate network
def evaluate_net(net, validation_loader, device, max_iterations=np.inf):
    net.eval()
    
    hands = net.hands

    frame_count = 0
    num_steps = 50
    absolute_pck3d = np.zeros(num_steps + 1)
    relative_pck3d = np.zeros(num_steps + 1)
    right_root_relative_pck3d = np.zeros(num_steps + 1)
    for idx, batch in enumerate(tqdm(validation_loader)):
        batch = Ev2HandSDataset.to_device(batch, device)

        events = batch['events']

        with torch.no_grad():
            outputs = net(events)
    
        batch['mano_gt'] = torch.mean(batch['mano_gt'])

        j3d_pred = list()
        j3d_gt = list()
        for hand_type in ['left', 'right']:  
            if batch['mano_gt']:      
                batch[hand_type]['j3d'] = hands[hand_type](global_orient=batch[hand_type]['global_orient'], 
                                                        hand_pose=batch[hand_type]['hand_pose'], 
                                                        betas=batch[hand_type]['shape'], 
                                                        transl=batch[hand_type]['trans']).joints

            j3d_pred.append(outputs[hand_type]['j3d'][:, None, ...] * 1000) # meters to millimeters
            j3d_gt.append(batch[hand_type]['j3d'][:, None, ...] * 1000) # meters to millimeters

        j3d_pred = torch.cat(j3d_pred, 1)
        j3d_gt = torch.cat(j3d_gt, 1)
    
        for i in range(0, j3d_pred.shape[0]):
            absolute_pck3d += absolute_pck3d_frame(j3d_pred[i], j3d_gt[i], num_steps=num_steps, dist_max_mm=num_steps)
            relative_pck3d += relative_pck3d_frame(j3d_pred[i], j3d_gt[i], num_steps=num_steps, dist_max_mm=num_steps)
            right_root_relative_pck3d += right_root_relative_pck3d_frame(j3d_pred[i], j3d_gt[i], num_steps=num_steps, dist_max_mm=num_steps)
            
            frame_count += 1
        
        if frame_count >= max_iterations:
            break
        
        print(f"L1 Distance: {torch.abs(j3d_pred - j3d_gt).mean().item()}")

    absolute_pck3d /= frame_count
    relative_pck3d /= frame_count
    right_root_relative_pck3d /= frame_count

    net.train()
     
    absolute_auc = get_auc(absolute_pck3d)
    relative_auc = get_auc(relative_pck3d)
    right_root_relative_auc = get_auc(right_root_relative_pck3d)
    
    auc = relative_auc

    return {
        'pck3d': {
                    'absolute': absolute_pck3d,
                    'relative': relative_pck3d,
                    'right_root_relative': right_root_relative_pck3d,
        },
        'auc': {
                    'relative': relative_auc,
                    'absolute': absolute_auc,
                    'right_root_relative': right_root_relative_auc,
        }
    }, auc


def main():
    arg_parser.evaluate()

    os.makedirs('./outputs', exist_ok=True)

    save_path = os.environ['CHECKPOINT_PATH']
    batch_size = int(os.environ['BATCH_SIZE'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TEHNetWrapper(device=device)

    print(f"Loading checkpoint from {save_path}")

    checkpoint = torch.load(save_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    
    validation_dataset = Ev2HandSDataset(f'{ROOT_TRAIN_DATA_PATH}/test', augment=False)

    num_workers = 8
    num_workers = multiprocessing.cpu_count()
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    metrics, eval_score = evaluate_net(net, validation_loader, device)
    print(f"Eval score: {eval_score}")
    
    np.save('./outputs/metrics.npy', metrics)

    print(f"Absolute AUC: {metrics['auc']['absolute']}")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics['pck3d']['absolute'])
    ax.figure.savefig('./outputs/pck3d_absolute.png')
    fig.clf()
    
    print(f"Relative AUC: {metrics['auc']['relative']}")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics['pck3d']['relative'])
    ax.figure.savefig('./outputs/pck3d_relative.png')
    fig.clf()

    print(f"right root Relative AUC: {metrics['auc']['right_root_relative']}")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics['pck3d']['right_root_relative'])
    ax.figure.savefig('./outputs/pck3d_right_root_relative.png')
    fig.clf()


if __name__ == '__main__':
    main()

    