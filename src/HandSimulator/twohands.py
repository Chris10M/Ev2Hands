import numpy as np
import random
import smplx
import torch
import trimesh
from torch.utils.data import Dataset
from mano_texture import ManoTexture
from manotosmplx import MANOTOSMPLX, Forearms
from manopth.manolayer import ManoLayer


from settings import MANO_PATH, DATA_PATH, SEGMENTAION_COLOR


class TwoHands(Dataset):
    def __init__(self, mano_sequence, device, compute_smplx=False):        
        mano_layer = {
                'left': ManoLayer(mano_root=f'{MANO_PATH}/mano', use_pca=True, ncomps=45, side='left', flat_hand_mean=False).to(device),
                'right': ManoLayer(mano_root=f'{MANO_PATH}/mano', use_pca=True, ncomps=45, side='right', flat_hand_mean=False).to(device)}
        for hand_type in ['left', 'right']: mano_layer[hand_type].faces = mano_layer[hand_type].th_faces.cpu().numpy()

        if torch.sum(torch.abs(mano_layer['left'].th_shapedirs[:,0,:] - mano_layer['right'].th_shapedirs[:,0,:])) < 1:
            print('Fix th_shapedirs bug of MANO')
            mano_layer['left'].th_shapedirs[:,0,:] *= -1

        self.mano_hands = mano_layer

        self.seq_dict = mano_sequence
        self.keys = list(self.seq_dict.keys())
        self.n_sequences = len(self.keys)

        self.device = device

        self.segmentation_color = {k: trimesh.visual.color.to_rgba(np.array(v) * 255) for k, v in SEGMENTAION_COLOR.items()}       
        self.mano_texture = ManoTexture(DATA_PATH, device=device)

        self.compute_smplx = compute_smplx
        
        if self.compute_smplx:
            self.manotosmplx = MANOTOSMPLX(device=device)

        self.forearms = Forearms(radius=0.0275, num_vecs_circle=36, mano_texture=self.mano_texture)

    def infer_mano(self, hand_type, global_orient, hand_pose, shape, trans):
        global_orient = torch.tensor(global_orient[None, :], dtype=torch.float32, device=self.device) 
        hand_pose = torch.tensor(hand_pose[None, :], dtype=torch.float32, device=self.device)
        shape = torch.tensor(shape[None, :], dtype=torch.float32, device=self.device) 
        trans = torch.tensor(trans[None, :], dtype=torch.float32, device=self.device)

        vertices, joints = self.mano_hands[hand_type](torch.cat([global_orient, hand_pose], 1), shape, trans)
        verts, j3d = vertices[0].cpu().numpy() / 1000, joints[0].cpu().numpy() / 1000
        faces = self.mano_hands[hand_type].faces

        return verts, j3d, faces

    def generate_mesh(self, two_hands_output, texture_type='uv'):
        meshes = list()

        if self.compute_smplx:
            smplx_model = two_hands_output['smplx_model']
            verts = smplx_model['verts']
            faces = smplx_model['faces'] 
            meshes.append(trimesh.Trimesh(vertices=verts, faces=faces))

        hand_info = two_hands_output['hand_info']
        for hand_type in hand_info.keys():
            trans_jitter = 5 * np.random.random(3) / 1000 # 5 mm jitter
            rotation_jitter = 0#np.random.random(3) / 1000 
                        
            verts, j3d, faces = self.infer_mano(hand_type, hand_info[hand_type]['global_orient'] + rotation_jitter, 
                                                           hand_info[hand_type]['hand_pose'], 
                                                           hand_info[hand_type]['shape'], 
                                                           hand_info[hand_type]['trans'] + trans_jitter)

            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            
            if texture_type == 'segmentation':    
                mesh.visual.vertex_colors = self.segmentation_color[hand_type]
            else:
                mesh.visual.vertex_colors = self.mano_texture(hand_type)


            if texture_type == 'segmentation':   
                meshes.append(self.forearms(hand_type, j3d, texture_mesh=False))
            else:
                meshes.append(self.forearms(hand_type, j3d, texture_mesh=True))

            meshes.append(mesh)
            
        return meshes

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, index):
        key = self.keys[index]

        mano_params = self.seq_dict[key]
    
        hand_info = dict()
        for mano_param in mano_params:
            hand_type = mano_param['hand_type']

            pose = mano_param['pose']
            shape = mano_param['shape']
            trans = mano_param['trans']

            hand_info[hand_type] = {
                'global_orient': mano_param['pose'][:3],
                'hand_pose': mano_param['pose'][3:],
                'shape': mano_param['shape'],
                'trans': mano_param['trans'],
            } 

        output = {
            'hand_info': hand_info
        }

        if self.compute_smplx:
            smplx_model = self.manotosmplx(hand_info)
            output['smplx_model'] = smplx_model

        return output
