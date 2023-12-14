import sys; sys.path.append('..')
import numpy as np
import torch
import pickle
import hashlib
import os
import trimesh
import cv2
import smplx
import random
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from settings import DATA_PATH, MANO_PATH, MANO_CMPS


class HTML_numpy():
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.tex_mean = self.model['mean']  # the mean texture
        self.tex_basis = self.model['basis'] # 101 PCA comps
        self.index_map = self.model['index_map'] # the index map, from a compact vector to a 2D texture image

        self.num_total_comp = 101

    def check_alpha(self, alpha):
        # just for checking the alpha's length
        if alpha.size < self.num_total_comp :
            n_alpha = np.zeros(self.num_total_comp,1)
            n_alpha[0:alpha.size,:] = alpha
        elif alpha.size > self.num_total_comp:
            n_alpha = alpha.reshape(alpha.size,1)[0:self.num_total_comp,:]
        else:
            n_alpha = alpha
        return alpha

    def __call__(self, alpha):
        # first check the length of the input alpha vector
                # first check the length of the input alpha vector
        alpha = self.check_alpha(alpha)
        offsets = np.dot(self.tex_basis, alpha)
        tex_code = offsets + self.tex_mean
        new_tex_img = np.clip(self.vec2img(tex_code, self.index_map) / 255, 0, 1) * 255

        return new_tex_img


    def vec2img(self, tex_code, index_map):
        # inverse vectorize: from compact texture vector to 2D texture image
        img1d = np.zeros(1024*1024*3)
        img1d[index_map] = tex_code
        return img1d.reshape((3, 1024,1024)).transpose(2,1,0)

    def generate_texture(self, texture_root_path):        
        alpha = np.random.randn(self.num_total_comp) * 2
        result = self(alpha)
        
        texture_path = f'{texture_root_path}/{hashlib.md5(result.tobytes()).hexdigest()}.png'
        cv2.imwrite(texture_path, result[:, :, ::-1])


class ManoTexture:
    def generate_uv_map(self):
        texture_root_path = f'{DATA_PATH}/mano_textures'
        os.makedirs(texture_root_path, exist_ok=True)

        if not os.listdir(texture_root_path):
            print('Computing Textures for Mano..')
            os.system(f'python mano_texture.py {texture_root_path}')
            print('Done')

        file_name = random.choice(os.listdir(texture_root_path))
        file_path = f'{texture_root_path}/{file_name}'

        uv = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB).astype(dtype=np.float32) / 255
        uv = torch.tensor(uv[None, ...])
        return uv

    def generate_vertex_color(self):
        uv = self.generate_uv_map()
                      
        for hand_type in ['left', 'right']:
            uvs = self.hands[hand_type]
            faces_uvs = uvs['faces_uvs'] 
            verts_uvs = uvs['verts_uvs']

            textures_uv = TexturesUV(uv, faces_uvs, verts_uvs)
    
            verts = torch.rand(1, 778, 3)
            faces = torch.tensor(self.mano_layer[hand_type].faces[None, ...].astype(dtype=np.int64), dtype=torch.int64)

            meshes = Meshes(verts, faces, textures_uv)
            verts_colors_packed = torch.zeros_like(meshes.verts_packed())
            verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()  * 255 # (*)

            vertext_colors = verts_colors_packed.numpy().astype(dtype=np.uint8)
            self.hand_vertex_color[hand_type] = vertext_colors

    def reset(self):
        self.generate_vertex_color()
    
    def __init__(self, root_path, device):
        self.device = device
        self.uv_map = None

        self.hands = dict()
        for hand_type in ['left', 'right']:
            with open(f'{root_path}/TextureBasis/uvs_{hand_type}.pkl', 'rb') as pickle_file:
                uvs = pickle.load(pickle_file)
                
                faces_uvs = uvs['faces_uvs']
                verts_uvs = uvs['verts_uvs'].astype(dtype=np.float32)

                self.hands[hand_type] = {
                    'verts_uvs': torch.tensor(verts_uvs[None, ...]),
                    'faces_uvs': torch.tensor(faces_uvs[None, ...])
                }

        self.mano_layer = {'right': smplx.create(MANO_PATH, 'mano', use_pca=True, is_rhand=True, num_pca_comps=MANO_CMPS), 
                           'left':  smplx.create(MANO_PATH, 'mano', use_pca=True, is_rhand=False, num_pca_comps=MANO_CMPS)}

        self.hand_vertex_color = {
            'left': np.ones((778, 3), dtype=np.uint8) * 255,
            'right': np.ones((778, 3), dtype=np.uint8) * 255
        }
        
        self.reset()

    def __call__(self, hand_type):
        return self.hand_vertex_color[hand_type]


def main():
    if len(sys.argv) != 2:
        print('Texture Save Path not given as argument') 
        exit(0)

    tex_model_path = f"{DATA_PATH}/TextureBasis/model_wosr/model.pkl"
    texture_generator = HTML_numpy(tex_model_path)    

    for _ in range(0, 1000):
        texture_generator.generate_texture(sys.argv[1])

        
if __name__ == '__main__':
    main()
