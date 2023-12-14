import numpy as np
import os.path as osp
import cv2
import torch
import trimesh

from torch.utils.data import Dataset
from pycocotools.coco import COCO

from PIL import Image, ImageDraw
from collections import defaultdict
import os
import pickle
import json
import scipy.io as sio
import smplx

from .utils import interpolate_sequence
from settings import MANO_PATH, DATA_PATH, OUTPUT_WIDTH, OUTPUT_HEIGHT
# from manopth.manolayer import ManoLayer
from smplx.utils import Struct
import pickle


class CameraTransform:
    def init_mano_conversion(self):
        joint_regressor = np.load(f'{DATA_PATH}/J_regressor_mano_ih26m.npy')
        root_joint_idx = 20

        self.joint_regressor = joint_regressor
        self.root_joint_idx = root_joint_idx

    def __init__(self, root_path, mode) -> None:
        self.mode = mode # train, test, val
        
        assert mode in ['train', 'test', 'val']

        self.root_path = root_path
        self.annot_path = f'{self.root_path}/annotations/{mode}'

        with open(osp.join(self.annot_path, 'InterHand2.6M_' + self.mode + '_camera.json')) as f: 
            self.cameras = json.load(f)

        self.init_mano_conversion()
        self.rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])


    def get_camera_indices(self, capture_idx):
        cam_param = self.cameras[str(capture_idx)]
        return sorted(list(cam_param['focal'].keys()))

    def get_camera_param(self, capture_idx, cam_idx):
        cam_idx = str(cam_idx)
        cam_param = self.cameras[str(capture_idx)]
    
        focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
        princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)

        t, R = np.array(cam_param['campos'][cam_idx], dtype=np.float32).reshape(3), np.array(cam_param['camrot'][cam_idx], dtype=np.float32).reshape(3,3)
        t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
    
        return {'intrinsics': {'focal': focal, 'princpt': princpt}, 'extrinsics': {'R': R, 't': t}}

    def transform_pts(self, R, t, world_pts):
        cam_pts = np.dot(R, world_pts.transpose(1,0)).transpose(1,0) + t.reshape(1,3)

        return cam_pts
        
    def transform_mano_params(self, R, t, hand_type, mano_layer, mano_param, camera=None):
        joint_regressor = self.joint_regressor
        root_joint_idx = self.root_joint_idx

        # mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
        # root_pose = mano_pose[0].numpy()
        root_pose = mano_param['global_orient']

        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose)) # multiply camera rotation to MANO root pose
        root_pose = torch.FloatTensor(root_pose).view(1,3)
        
        hand_pose = torch.FloatTensor(mano_param['hand_pose']).view(1, -1)
        shape = torch.FloatTensor(mano_param['shape']).view(1, -1)

        device = mano_layer[hand_type].th_shapedirs.device
        vertices, joints = mano_layer[hand_type](torch.cat([root_pose, hand_pose], 1).to(device), shape.to(device)) # this is rotation-aligned, but not translation-aligned with the camera coordinates
        mesh = vertices[0].cpu().numpy() / 1000
        joint_from_mesh = np.dot(joint_regressor, mesh)
                
        # compenstate rotation (translation from origin to root joint was not cancled)
        root_joint = joint_from_mesh[root_joint_idx, None, :]
        trans = np.array(mano_param['trans'])
        trans = np.dot(R, trans.reshape(3,1)).reshape(1,3) - root_joint + np.dot(R, root_joint.transpose(1,0)).transpose(1,0) + t / 1000 # change translation vector
        trans = torch.from_numpy(trans).view(1, 3)
       
        transfomed_mano_param = {
                'hand_type': hand_type,
                
                'global_orient': root_pose[0].float().cpu().numpy(),
                'hand_pose': hand_pose[0].float().cpu().numpy(),
                'shape': shape[0].float().cpu().numpy(),
                'trans': trans[0].float().cpu().numpy(),
            } 
        
        return transfomed_mano_param





class AAtoPCA:        
    def load_mano_inverse_hand_components(self):
        model_path = MANO_PATH + '/mano'
        ext = 'pkl'
        
        hand_components = dict()
        for hand_type, is_rhand in [('right', True), ('left', False)]:
            if osp.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format(
                    'RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
        
            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))

            data_struct = Struct(**model_data)
            hand_components[hand_type] = np.linalg.inv(np.array(data_struct.hands_components, dtype=np.float32))
            
        return hand_components

    def __init__(self) -> None:
        self.inverse_hand_components = self.load_mano_inverse_hand_components()

    def __call__(self, hand_type, pose):
        is_np = isinstance(pose, np.ndarray)

        if not is_np:
            pose = np.array(pose)        
    
        hand_pose = pose[3:]
        pca_hand_pose = hand_pose @ self.inverse_hand_components[hand_type]
        pose[3:] = pca_hand_pose

        if is_np:
            return pose
        
        return pose.tolist()

    def compute_mano_sequence(self, mano_sequence):
        for i in list(mano_sequence.keys()):
            for j in range(len(mano_sequence[i])):
                mano_sequence[i][j]['pose'] = self(mano_sequence[i][j]['hand_type'], mano_sequence[i][j]['pose'])

        return mano_sequence



class InterHand(Dataset):
    def __init__(self, root_path, mode):
        
        # self.aa_to_pca = AAtoPCA()

        self.mode = mode # train, test, val
        self.root_path = root_path
        

        assert mode in ['train', 'test', 'val']

        self.img_path = f'{self.root_path}/images/{mode}'
        self.annot_path = f'{self.root_path}/annotations/{mode}'
        
        # load annotation
        print(f"Load annotation from  {self.annot_path}")

        db = COCO(osp.join(self.annot_path, 'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.annot_path, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f: joints = json.load(f)
        with open(osp.join(self.annot_path, 'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot.json')) as f: mano_params = json.load(f)
        
        self.joint_num = 21 # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}

        keys = set()
        self.image_paths = defaultdict(dict) 
        self.mano_data = defaultdict(dict)
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
 
            capture_id = str(img['capture'])
            seq_name = img['seq_name']
            camera_idx = img['camera']
            frame_idx = img['frame_idx']

            img_path = osp.join(self.img_path, img['file_name'])

            if camera_idx not in self.image_paths[capture_id]:
                self.image_paths[capture_id][camera_idx] = {frame_idx: img_path}
            else:
                self.image_paths[capture_id][camera_idx][frame_idx] = img_path
            
            try:
                left_params = mano_params[capture_id][str(frame_idx)]['left']
                right_params = mano_params[capture_id][str(frame_idx)]['right']
                
                self.mano_data[capture_id][frame_idx] = {
                    'left': left_params,
                    'right': right_params
                }

                keys.add(capture_id)
            except KeyError as e:
                print(f'{e}, capture_id: {capture_id}, frame_idx: {frame_idx}')
        self.keys = sorted(list(keys))

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        capture_id = self.keys[idx]        
        mano_data = self.mano_data[capture_id]
        image_paths = self.image_paths[capture_id]
        
        return {'capture_id': capture_id, 'mano_data': mano_data, 'image_paths': image_paths}
 