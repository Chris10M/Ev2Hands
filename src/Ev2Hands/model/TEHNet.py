import numpy as np
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()

    def forward(self, key, value, query):
        query = query.permute(0, 2, 1)
        N, KC = key.shape[:2]
        key = key.view(N, KC, -1)

        N, KC = value.shape[:2]
        value = value.view(N, KC, -1)
       
        sim_map = torch.bmm(key, query)
        sim_map = (KC ** -.5 ) * sim_map
        sim_map = F.softmax(sim_map, dim=1)

        context = torch.bmm(sim_map, value)

        return context


class MANORegressor(nn.Module):
    def __init__(self, n_inp_features=4, n_pose_params=6, n_shape_params=10):
        super(MANORegressor, self).__init__()
        
        normal_channel = True

        if normal_channel:
            additional_channel = n_inp_features
        else:
            additional_channel = 0

        self.normal_channel = normal_channel
        
        self.sa1 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], additional_channel, [[128, 128, 256], [128, 196, 256]])
        self.sa2 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512], group_all=True)

        self.n_pose_params = n_pose_params
        self.n_mano_params = n_pose_params + n_shape_params

        self.mano_regressor = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 3 + self.n_mano_params + 3),   
        )


    def J3dtoJ2d(self, j3d, scale):
        B, N = j3d.shape[:2]
        device = j3d.device

        j2d = torch.zeros(B, N, 2, device=device)
        j2d[:, :, 0] = scale[:, :, 0] * j3d[:, :, 0]
        j2d[:, :, 1] = scale[:, :, 1] * j3d[:, :, 1]

        return j2d

    def forward(self, xyz, features, mano_hand, previous_mano_params=None):
        device = xyz.device
        batch_size = xyz.shape[0]        

        l0_xyz = xyz
        l0_points = features

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l2_xyz = l2_xyz.squeeze(-1)
        l2_points = l2_points.squeeze(-1)

        if previous_mano_params is None:
            previous_mano_params = torch.zeros(self.n_mano_params).unsqueeze(0).expand(batch_size, -1).to(device)
            previous_rot_trans_params = torch.zeros(6).unsqueeze(0).expand(batch_size, -1).to(device)
        
        mano_params = self.mano_regressor(l2_points)
        
        global_orient = mano_params[:, :3]
        hand_pose = mano_params[:, 3:3+self.n_pose_params]
        betas = mano_params[:, 3+self.n_pose_params:-3]
        transl = mano_params[:, -3:]
                        
        device = mano_hand.shapedirs.device

        mano_args = {
            'global_orient': global_orient.to(device),
            'hand_pose'    : hand_pose.to(device),
            'betas'        : betas.to(device),
            'transl'       : transl.to(device),
        } 
        
        mano_outs = dict()

        output = mano_hand(**mano_args)
        mano_outs['vertices'] = output.vertices
        mano_outs['j3d'] = output.joints

        mano_outs.update(mano_args)

        if not self.training:            
            mano_outs['faces'] = np.tile(mano_hand.faces, (batch_size, 1, 1))

        return mano_outs


class TEHNet(nn.Module):
    def __init__(self, n_pose_params, num_classes=4):
        super(TEHNet, self).__init__()
        
        normal_channel = True

        if normal_channel:
            additional_channel = 1 + int(os.getenv('ERPC', 0))
        else:
            additional_channel = 0

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])

        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 256])

        self.classifier = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Conv1d(256, num_classes, 1)
        )
    
        self.attention_block = AttentionBlock()

        self.left_mano_regressor = MANORegressor(n_pose_params=n_pose_params)
        self.right_mano_regressor = MANORegressor(n_pose_params=n_pose_params)

        self.mhlnes = int(os.getenv('MHLNES', 0))

        self.left_query_conv = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 3//2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, 3, 1, 3//2),
            nn.BatchNorm1d(256),
        )

        self.right_query_conv = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 3//2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, 3, 1, 3//2),
            nn.BatchNorm1d(256),
        )

    def forward(self, xyz, mano_hands):
        device = xyz.device
        
        # Set Abstraction layers
        l0_points = xyz
        
        l0_xyz = xyz[:, :3, :]

        if self.mhlnes:
            l0_xyz[:, -1, :] = xyz[:, 3:, :].mean(1)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        seg_out = self.classifier(l0_points)
        feat_fuse = l0_points

        left_hand_features = self.attention_block(seg_out, feat_fuse, self.left_query_conv(feat_fuse))
        right_hand_features = self.attention_block(seg_out, feat_fuse, self.right_query_conv(feat_fuse))

        left = self.left_mano_regressor(l0_xyz, left_hand_features, mano_hands['left'])
        right = self.right_mano_regressor(l0_xyz, right_hand_features, mano_hands['right'])

        return {'class_logits': seg_out, 'left': left, 'right': right}


def main():

    net = TEHNet(n_pose_params=6)
    points = torch.rand(4, 4, 128)
    net(points)


if __name__ == '__main__':
    main()
