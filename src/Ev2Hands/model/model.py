import torch
import torch.nn as nn
import trimesh
import numpy as np
from .TEHNet import TEHNet
from .utils import create_mano_layers
from settings import MANO_PATH, MANO_CMPS


class TEHNetWrapper():
    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, params, *args, **kwargs):
        modified_params = dict()

        for k, v in params.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
        
            modified_params[k] = v               
        
        self.net.load_state_dict(modified_params, *args, **kwargs)

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.training = True
        return self.net.train()

    def eval(self):
        self.training = False
        return self.net.eval()

    def P3dtoP2d(self, j3d, scale, translation):
        B, N = j3d.shape[:2]

        homogeneous_j3d = torch.cat([j3d, torch.ones(B, N, 1, device=j3d.device)], 2)  
        homogeneous_j3d = homogeneous_j3d @ self.rot.detach()

        translation = translation.unsqueeze(1) 
        scale = scale.unsqueeze(1)

        j2d = torch.zeros(B, N, 2, device=j3d.device)
        j2d[:, :, 0] = translation[:, :, 0] + scale[:, :, 0] * homogeneous_j3d[:, :, 0]
        j2d[:, :, 1] = translation[:, :, 1] + scale[:, :, 1] * homogeneous_j3d[:, :, 1]

        return j2d

    def __init__(self, device):
        net = TEHNet(n_pose_params=MANO_CMPS).to(device)

        self.net = net
        self.training = False
        
        self.hands = create_mano_layers(MANO_PATH, device, MANO_CMPS)

        self.rot = torch.tensor(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]), device=device).float()

    def __call__(self, inp):
        outputs = self.net(inp, self.hands)

        return outputs
