import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import numpy as np
import pickle
import torch
import trimesh
from colour import Color
from torch import nn
from pytorch3d.structures import Meshes
from typing import Union, Iterable
from human_body_prior.body_model.body_model import BodyModel
# from human_body_prior.models.ik_engine import IK_Engine
from smplx.vertex_ids import vertex_ids 
import math
from settings import DATA_PATH


class SourceKeyPoints(nn.Module):
    def compute_vertex_normal_batched(self, vertices, indices):
        return Meshes(verts=vertices, faces=indices.expand(len(vertices), -1, -1)).verts_normals_packed().view(-1, vertices.shape[1],3)

    def __init__(self,
                 bm: Union[str, BodyModel],
                 vids: Iterable[int],
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, persistant_buffer=False) if isinstance(bm, str) else bm
        self.vids = vids
        self.kpts_colors = np.array([Color('grey').rgb for _ in vids])
        self.bm_f = []

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        j3d = new_body.Jtr
        virtual_markers = new_body.v[:, self.vids]

        torso_vids = [5950, 5431, 5488, 5490]
        torso_kpts = new_body.v[:, torso_vids] 
        
        hand_vids = [4687, 4686, 7398, 7403]
        hand_kpts = new_body.v[:, hand_vids] 
        # 54, 25
        # print(j3d[:, 25:55].shape)
        # exit(0)

        return {'source_kpts': virtual_markers, 'body': new_body, 'torso_kpts': torso_kpts, 'hand_kpts': hand_kpts}



device = o3d.core.Device("cuda:0")


class MANOTOSMPLX:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'mano_correspondances'):
            cls.mano_correspondances = dict()
            with open(f'{DATA_PATH}/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
                idxs_data = pickle.load(f)
                cls.mano_correspondances['left'] = idxs_data['left_hand'] 
                cls.mano_correspondances['right'] = idxs_data['right_hand']

            sample_amass = np.load(f'{DATA_PATH}/VPOSER/amass_sample.npz', allow_pickle=True)
            cls.amass_vids = sample_amass['vids']
            cls.markers_orig = sample_amass['markers']
        
            cls.vposer_expr_dir = f'{DATA_PATH}/VPOSER/V02_05' #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
            cls.bm_fname =  f'{DATA_PATH}/models/smplx/SMPLX_NEUTRAL.npz'#'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads

        return super(MANOTOSMPLX, cls).__new__(cls)
        
        
    def __init__(self, device) -> None:
        self.device = device

        self.smplx_model = BodyModel(self.bm_fname, persistant_buffer=False).to(device)

        self.initial_body_params = dict()

    def optimize(self, source_pts, target_pts):
        data_loss = torch.nn.MSELoss(reduction='sum')
        optimizer_args = {'type':'LBFGS', 'max_iter':500, 'lr':1, 'tolerance_change': 1e-4, 'history_size':200}
        stepwise_weights = [{'data': 1., 'poZ_body': 0.01, 'betas': .5},]
    
        ik_engine = IK_Engine(vposer_expr_dir=self.vposer_expr_dir,
                            verbosity=1,
                            display_rc=(2, 2),
                            data_loss=data_loss,
                            stepwise_weights=stepwise_weights,
                            optimizer_args=optimizer_args).to(self.device)
        
        ik_res = ik_engine(source_pts, target_pts, self.initial_body_params)
        ik_res_detached = {k: v.detach() for k, v in ik_res.items()}
        self.initial_body_params = ik_res_detached 

    def __call__(self, hand_info):
        vids = list()
        target_vertices = list()
        for hand_type in hand_info.keys():
            verts = hand_info[hand_type]['verts']
        
            vids.append(self.mano_correspondances[hand_type])
            target_vertices.append(verts)

        mano_vids = np.concatenate(vids)
        target_vertices = np.concatenate(target_vertices, axis=0)

        vids, target_pts = self.add_extra_targets(hand_info, mano_vids, target_vertices)
        source_pts = SourceKeyPoints(bm=self.smplx_model, vids=vids).to(self.device)

        target_pts = torch.tensor(target_pts)[None, ...].to(self.device)
        self.optimize(source_pts, target_pts)

        output = self.smplx_model(**self.initial_body_params) 
        verts = output.v.cpu().numpy()[0]
        faces = output.f.cpu().numpy()
        
        print(verts.shape, faces.shape)

        index_mask = np.ones(verts.shape[0], np.bool)
        index_mask[mano_vids] = 0
        # smplx_wo_hands_indices = np.nonzero(index_mask)[0]
        

        # verts = np.take(verts, smplx_wo_hands_indices, 0)

        # smplx_wo_hands_indices
        # verts = np.take(verts, smplx_wo_hands_indices, 0)

        print(verts.shape)
        # verts = verts[mask]
        # print(smplx_wo_hands_indices, mano_vids.shape, verts.shape)
        # mano_vids
        # point_cloud = trimesh.PointCloud(verts)
        # point_cloud.show()
        # mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        #                faces=[[0, 1, 2]])

        mesh = trimesh.Trimesh(verts, faces)
        face_mask = index_mask[mesh.faces].all(axis=1)

        mesh.update_vertices(index_mask)
        mesh.update_faces(face_mask)

        verts = mesh.vertices
        faces = mesh.faces

        # mesh.show()

        return {
            'verts': verts,
            'faces': faces
        }


    def add_extra_targets(self, hand_info, mano_vids, target_vertices):
        markers_orig = self.markers_orig
        amass_vids = self.amass_vids
        
        rnd_frame_ids = np.random.choice(len(markers_orig), 1, replace=False)
        target_pts = markers_orig[rnd_frame_ids].squeeze()

        non_hand_kpts = list()
        non_hand_vids = list()
        for idx, vid in enumerate(amass_vids):
            if np.count_nonzero(mano_vids == vid):
                pass
            else:
                non_hand_vids.append(vid)
                non_hand_kpts.append(target_pts[idx])
        

        elbow = {
            'left': {
                'vid': non_hand_vids[24],
                'kpt': non_hand_kpts[24]
            },
            'right': {
                'vid': non_hand_vids[15],
                'kpt': non_hand_kpts[15]
            }
        }

        shoulder = {
            'left': {
                'vid': non_hand_vids[2],
                'kpt': non_hand_kpts[2]
            },
            'right': {
                'vid': non_hand_vids[17],
                'kpt': non_hand_kpts[17]
            }
        }

        extra_kpts = list()
        extra_vids = list()
        for hand_type, hand in hand_info.items():
            for mano_vertex_name, mano_vertex_id in vertex_ids['mano'].items():
                smplx_vertex_name = hand_type[0] + mano_vertex_name 
                smplx_vertex_id = vertex_ids['smplx'][smplx_vertex_name]

                mano_kpt = hand['verts'][mano_vertex_id]
                
                extra_kpts.append(mano_kpt)
                extra_vids.append(smplx_vertex_id)

                # print(mano_vertex_id, smplx_vertex_id, mano_kpt)
                # exit(0)


            j3d = hand['j3d']
    
            wrist_j3d = j3d[0]
            wrist_vector = (j3d[4] - j3d[0]) / np.linalg.norm((j3d[4] - j3d[0]))

            # kpt = wrist_j3d - 1 * wrist_vector
            # elbow[hand_type]['kpt'] = kpt
            # extra_kpts.append(elbow[hand_type]['kpt'])
            # extra_vids.append(elbow[hand_type]['vid'])

            kpt = wrist_j3d - 1 * wrist_vector
            shoulder[hand_type]['kpt'] = kpt
            
            extra_kpts.append(shoulder[hand_type]['kpt'])
            extra_vids.append(shoulder[hand_type]['vid'])

        hip = wrist_j3d - 3 * wrist_vector
        extra_kpts.append(hip)
        extra_vids.append(non_hand_vids[13])


        extra_vids = np.array(extra_vids)
        extra_kpts = np.array(extra_kpts)

        selected_indices = np.random.choice(mano_vids.shape[0], 512, replace=False)

        vids = np.concatenate([extra_vids, mano_vids[selected_indices]])
        target_pts = np.concatenate([extra_kpts, target_vertices[selected_indices]], axis=0)

        return vids, target_pts


from scipy.spatial.transform import Rotation as R
import trimesh


class Forearms:
    def __init__(self, radius, num_vecs_circle, mano_texture):
        self.colors = {
            'left': None,
            'right': None
        }

        self.radius = radius
        self.num_vecs_circle = num_vecs_circle
        self.mano_texture = mano_texture

    def __call__(self, hand_type, joints, texture_mesh=True):
        mesh = self.add_forearms_fixed(joints, self.radius, self.num_vecs_circle, hand_type)

        if self.colors[hand_type] is None:
            hand_vertex_color = self.mano_texture(hand_type)
        
            vdx = np.random.choice(hand_vertex_color.shape[0], mesh.visual.vertex_colors.shape[0])
            color = hand_vertex_color[vdx]
            
            self.colors[hand_type] = color.mean(0)
        
        if texture_mesh:
            mesh.visual.vertex_colors = self.colors[hand_type]

        return mesh

    # adds cylindrical forearms with fixed elbow locations for comparison with EventHands
    @classmethod
    def add_forearms_fixed(cls, joints, radius, num_vecs_circle, side):
        root = joints[0, :]
        mcp_middle = joints[9, :]
        root_opposite = np.array([0.0, 0.0, -0.5])

        if side == 'right':
            root_opposite = np.array([-0.4, 0.0, -0.5])
        elif side == 'left':
            root_opposite = np.array([0.4, 0.0, -0.5])

        dir_forearm = root + root_opposite
        dir_forearm = (dir_forearm / np.linalg.norm(dir_forearm)) * 2
        vec_supp1 = mcp_middle - root
        vec_supp1 = vec_supp1 / np.linalg.norm(vec_supp1)
        vec_supp2 = np.cross(vec_supp1, dir_forearm)
        vec_supp2 = vec_supp2 / np.linalg.norm(vec_supp2)
        vec_first_rel = np.cross(dir_forearm, vec_supp2) * radius / 2


        angle = 2 * math.pi / num_vecs_circle
        
        verts_additional = []

        verts_additional.append(root.reshape(1, 3))
        verts_additional.append(root_opposite.reshape(1, 3)) 

        # normals_additional = []
        for v in range(num_vecs_circle):
            angle_1 = angle * v
            angle_2 = angle * (v + 1)

            rotvec_1 = angle_1 * dir_forearm
            rotvec_2 = angle_2 * dir_forearm
            rot_1 = R.from_rotvec(rotvec_1)
            rot_2 = R.from_rotvec(rotvec_2)

            vec_rotated_1_rel = rot_1.apply(vec_first_rel)
            vec_rotated_2_rel = rot_2.apply(vec_first_rel)

            vec_rotated_1 = root + vec_rotated_1_rel
            vec_rotated_2 = root + vec_rotated_2_rel

            vec_rotated_opposite_1 = vec_rotated_1 + 0.9 * dir_forearm
            vec_rotated_opposite_2 = vec_rotated_2 + 0.9 * dir_forearm

            verts_additional.append(vec_rotated_2.reshape(1, 3)) 
            verts_additional.append(vec_rotated_1.reshape(1, 3))

            verts_additional.append(vec_rotated_opposite_2.reshape(1, 3))
            verts_additional.append(vec_rotated_opposite_1.reshape(1, 3))



            # normals_additional.append(dir_forearm.reshape(1, 3))
            # normals_additional.append(dir_forearm.reshape(1, 3))
            # normals_additional.append(dir_forearm.reshape(1, 3))

            # normals_additional.append(vec_rotated_2_rel.reshape(1, 3))
            # normals_additional.append(vec_rotated_1_rel.reshape(1, 3))
            # normals_additional.append(vec_rotated_2_rel.reshape(1, 3))
            

            # verts_additional.append(vec_rotated_opposite_2.reshape(1, 3))
            # verts_additional.append(vec_rotated_1.reshape(1, 3))
            # verts_additional.append(vec_rotated_2.reshape(1, 3))
            # verts_additional.append(vec_rotated_1.reshape(1, 3))
            
            # normals_additional.append(vec_rotated_2_rel.reshape(1, 3))
            # normals_additional.append(vec_rotated_1_rel.reshape(1, 3))
            # normals_additional.append(vec_rotated_2_rel.reshape(1, 3))

            # verts_additional.append(vec_rotated_opposite_2.reshape(1, 3))
            # verts_additional.append(vec_rotated_opposite_1.reshape(1, 3))

            # normals_additional.append(-dir_forearm.reshape(1, 3))
            # normals_additional.append(-dir_forearm.reshape(1, 3))
            # normals_additional.append(-dir_forearm.reshape(1, 3))

        verts_additional = np.concatenate(verts_additional, 0)
        # normals_additional = np.concatenate(normals_additional, 0)

        # uvs_additional = 0.5 * np.ones((num_vecs_circle * 12, 2))
    

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts_additional)
        # pcd.estimate_normals()
        # pcd.normals = o3d.utility.Vector3dVector(normals_additional)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 1)

        # trimesh.points.PointCloud(verts_additional).export('arms.ply')
        # o3d.io.write_triangle_mesh('arms_mesh.obj', mesh)
        # mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.25, create_uv_map=True)
        # mesh.translate(root, False)
        # mesh.rotate()
        mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles).convex_hull
        return mesh

        # if hand_vertex_color is not None:
        #     vdx = np.random.choice(hand_vertex_color.shape[0], mesh.visual.vertex_colors.shape[0])
        #     mesh.visual.vertex_colors = hand_vertex_color[vdx]

        #     # print(mesh.visual.vertex_colors.shape, hand_vertex_color.shape, vdx)

        # mesh.export('arms_hnd.obj')
        # # trimesh.util.concatenate([hand_mesh, mesh]).export(f'arms_{side}.obj')
        # # o3d.io.write_triangle_mesh('arms.obj', mesh)
        # trimesh.points.PointCloud(verts_additional).export('arms.ply')
        # vertices = np.concatenate((vertices, verts_additional), 0)
        # uvs = np.concatenate((uvs, uvs_additional), 0)
        # normals = np.concatenate((normals, normals_additional), 0)

        # return vertices, normals, uvs
        # return mesh