import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from manopth.manolayer import ManoLayer



def create_mano_layers(mano_path, device, n_cmps):
    class Output:
        def __init__(self, vertices, joints):
            self.vertices = vertices
            self.joints = joints

    class SmplxAdapter:
        def __init__(self, side):
            self.m = ManoLayer(mano_root=f'{mano_path}/mano', use_pca=True, ncomps=n_cmps, side=side, flat_hand_mean=False, robust_rot=True).to(device)
            self.faces = self.m.th_faces.cpu().numpy()
            self.shapedirs = self.m.th_shapedirs

        def __call__(self, global_orient, hand_pose, betas, transl): 
            vertices, joints = self.m(torch.cat([global_orient, hand_pose], 1), betas, transl)
            
            vertices /= 1000
            joints /= 1000

            return Output(vertices, joints)
                
    mano_layer = {
            'left': SmplxAdapter(side='left'),
            'right': SmplxAdapter(side='right')
            }
    
    if torch.sum(torch.abs(mano_layer['left'].m.th_shapedirs[:,0,:] - mano_layer['right'].m.th_shapedirs[:,0,:])) < 1:
        print('Fix th_shapedirs bug of MANO')
        mano_layer['left'].m.th_shapedirs[:,0,:] *= -1

    return mano_layer
