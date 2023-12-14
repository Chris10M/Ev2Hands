import collections
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from mesh_intersection.loss import DistanceFieldPenetrationLoss
from mesh_intersection.bvh_search_tree import BVH

from settings import MANO_CMPS, PROJECTION_MATRIX, OUTPUT_WIDTH, OUTPUT_HEIGHT
from camera import opengl_projection_transform


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)


def rotmat_to_rot6d(x):
    rotmat = x.reshape(-1, 3, 3)
    rot6d = rotmat[:, :, :2].reshape(x.shape[0], -1)
    return rot6d


class CollisionLoss:
    def __init__(self, device) -> None:
        max_collisions = 16 
        self.search_tree = BVH(max_collisions=max_collisions)
        
        self.collision_weight = 1e2
        print(f"collision_weight: {self.collision_weight}")

        sigma = 0.5
        point2plane = False
        self.pen_distance = DistanceFieldPenetrationLoss(sigma=sigma, point2plane=point2plane, vectorized=True, penalize_outside=False)

        self.device = device

    def __call__(self, outs):
        B, V, C = outs['left']['vertices'].shape
        
        verts = list()
        faces = list()
        for i in range(B):           
            hand_verts = torch.cat([outs['left']['vertices'][i], outs['right']['vertices'][i]])
            hand_faces = torch.cat([outs['left']['faces'], outs['right']['faces'] + V])
                        
            verts.append(hand_verts[None, ...])          
            faces.append(hand_faces[None, ...])
        
        verts_tensor = torch.cat(verts, dim=0)
        face_tensor = torch.cat(faces, dim=0)

        triangles = verts_tensor.view([-1, 3])[face_tensor]

        with torch.no_grad():                 
            collision_idxs = self.search_tree(triangles)
        
        loss = self.pen_distance(triangles, collision_idxs)
        indices = torch.nonzero(loss)
    
        if len(indices):
            loss = loss[indices].mean() * self.collision_weight
        else:
            loss = 0
                    
        return loss


class Loss(nn.Module):
    def __init__(self, hands, device):
        super(Loss, self).__init__()
        self.device = device
        self.hands = hands
        self.collision_loss = CollisionLoss(device)

        self.previous_out = dict()
        self.PROJECTION_MATRIX = torch.tensor(PROJECTION_MATRIX).to(device).float()

    def quant_error_loss(self, inp, target):
        c1 = target[:, 0]
        v1 = target[:, 1:]
        c2 = inp[:, 0]
        v2 = inp[:, 1:]
        
        tmp = torch.clamp(c1 * c2 + torch.sum(v1 * v2, 1), -1, 1)
        cosine_similarity = torch.abs(tmp)

        cosine_distance = 1 - cosine_similarity
            
        return cosine_distance.mean(-1)

    def index_losss(self, loss_fn, outs, targets, indices):
        indices = indices.int()
        
        if indices.sum() == 0: return 0

        loss = loss_fn(outs, targets, reduction='none')
        B = loss.shape[0]
        loss = loss.reshape(B, -1)
        D = loss.shape[1]
        indices = indices[:, None].repeat(1, D)
            
        index_loss = loss * indices
        index_mean_loss = index_loss.sum() / indices.sum()

        return index_mean_loss


    def forward(self, outs, targets, losses=None):
        targets['mano_gt'] = torch.mean(targets['mano_gt'])

        if targets['mano_gt']:
            return self.forward_mano_data(outs, targets, losses)

        return self.forward_non_mano_data(outs, targets, losses)

    def forward_mano_data(self, outs, targets, losses=None):
        if losses is None:
            losses = collections.defaultdict(float)

        for hand_type in ['left', 'right']:
            outputs = self.hands[hand_type](global_orient=targets[hand_type]['global_orient'], 
                                            hand_pose=targets[hand_type]['hand_pose'], 
                                            betas=targets[hand_type]['shape'], 
                                            transl=targets[hand_type]['trans'])
                                            
            targets[hand_type]['j3d'] = outputs.joints
            targets[hand_type]['vertices'] = outputs.vertices

            outs[hand_type]['faces'] = targets[hand_type]['faces'] = torch.tensor(self.hands[hand_type].faces).to(self.device) 

        losses['loss_interpen'] += self.collision_loss(outs)


        interacting_indices = torch.sum(targets['handedness'], 1) == 2
        
        losses['loss_inter_shape'] += self.index_losss(F.mse_loss, outs['left']['betas'], 
                                                                   outs['right']['betas'], 
                                                                   interacting_indices)

        losses['loss_inter_transl'] += self.index_losss(F.mse_loss, outs['left']['transl'] - outs['right']['transl'], 
                                                                    (targets['left']['trans'] - targets['right']['trans']), 
                                                                    interacting_indices) * 100

        losses['loss_inter_j3d'] += self.index_losss(F.mse_loss, outs['left']['j3d'] - outs['right']['j3d'], 
                                                                 (targets['left']['j3d'] - targets['right']['j3d']), 
                                                                 interacting_indices) * 100            

        for hand_type in ['left', 'right']:
            indices = targets[hand_type]['valid']

            losses['loss_global_orient'] += self.index_losss(F.mse_loss, outs[hand_type]['global_orient'], targets[hand_type]['global_orient'], indices) * 10

            targets[hand_type]['hand_pose'] = targets[hand_type]['hand_pose'][:, :MANO_CMPS]
            losses['loss_hand_pose'] += self.index_losss(F.mse_loss, outs[hand_type]['hand_pose'], targets[hand_type]['hand_pose'], indices) * 10

            losses['loss_rj3d'] += self.index_losss(F.l1_loss, (outs[hand_type]['j3d'][:, 1:, :] - outs[hand_type]['j3d'][:, :1, :]) * 1000, (targets[hand_type]['j3d'][:, 1:, :] - targets[hand_type]['j3d'][:, :1, :]) * 1000, indices)  * 0.01
            losses['loss_j3d'] += self.index_losss(F.l1_loss, outs[hand_type]['j3d'] * 1000, targets[hand_type]['j3d'] * 1000, indices) * 0.01

            losses['loss_shape'] += self.index_losss(F.mse_loss, outs[hand_type]['betas'], targets[hand_type]['shape'], indices) * 10
            losses['loss_transl'] += self.index_losss(F.l1_loss, outs[hand_type]['transl'], targets[hand_type]['trans'], indices) * 10


            losses['regularizer_loss'] += 0.1 * self.index_losss(F.mse_loss, outs[hand_type]['betas'], outs[hand_type]['betas'], indices)
            losses['regularizer_loss'] += self.index_losss(F.mse_loss, outs[hand_type]['hand_pose'], outs[hand_type]['hand_pose'], indices)
        
        losses['loss_class_logits'] = F.cross_entropy(outs['class_logits'], targets['class_logits'], weight=torch.tensor([1, 30, 30, 10]).float().to(self.device),
                                                      ignore_index=0)

        return losses     

    def forward_non_mano_data(self, outs, targets, losses=None):
        if losses is None:
            losses = collections.defaultdict(float)

        for hand_type in ['left', 'right']:
            outs[hand_type]['faces'] = torch.tensor(self.hands[hand_type].faces).to(self.device) 
            outs[hand_type]['j2d'] = opengl_projection_transform(self.PROJECTION_MATRIX, OUTPUT_WIDTH, OUTPUT_HEIGHT, outs[hand_type]['j3d'] * 1000)

        losses['loss_interpen'] += self.collision_loss(outs)

        interacting_indices = torch.sum(targets['handedness'], 1) == 2

        losses['loss_inter_shape'] += self.index_losss(F.mse_loss, outs['left']['betas'], 
                                                                   outs['right']['betas'], 
                                                                   interacting_indices) * 1e3

        losses['loss_inter_j3d'] += self.index_losss(F.l1_loss, (outs['left']['j3d'] - outs['right']['j3d']) * 1000, 
                                                                 (targets['left']['j3d'] - targets['right']['j3d']) * 1000, 
                                                                 interacting_indices)            

        for hand_type in ['left', 'right']:
            indices = targets[hand_type]['valid']

            losses['regularizer_loss'] += torch.mean(outs[hand_type]['betas'] ** 2) * 1e3
            losses['regularizer_loss'] += torch.mean(outs[hand_type]['hand_pose'] ** 2)

            losses['regularizer_loss'] *= 0.025

            losses['loss_rj3d'] += self.index_losss(F.l1_loss, (outs[hand_type]['j3d'][:, 1:, :] - outs[hand_type]['j3d'][:, :1, :]) * 1000, (targets[hand_type]['j3d'][:, 1:, :] - targets[hand_type]['j3d'][:, :1, :]) * 1000, indices)  * 10
            losses['loss_j2d'] += self.index_losss(F.mse_loss, outs[hand_type]['j2d'], targets[hand_type]['j2d'][..., :2], indices)

        
        return losses     
