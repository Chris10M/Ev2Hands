import random
import pyrender
import os
import lmdb
import numpy as np
import torch
import trimesh
import pickle5 as pickle
import time
import cv2
import imutils
import torch.nn as nn
import pyarrow as pa
from glob import glob
from functools import partial
from settings import ROOT_TRAIN_DATA_PATH, OUTPUT_WIDTH, OUTPUT_HEIGHT, GENERATION_MODE, AUGMENTATED_SEQUENCE, DATA_PATH
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def word_to_screen_transform(cameras, points, img_w, img_h, **kwargs):
    verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(points)
    # view to NDC transform
    to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
    projection_transform = cameras.get_projection_transform(**kwargs).compose(
        to_ndc_transform
    )
    verts_ndc = projection_transform.transform_points(verts_view)
    verts_ndc[..., 2] = verts_view[..., 2]

    verts_ndc = torch.clamp_(verts_ndc, -1, 1)

    screen_space = torch.zeros_like(verts_ndc)
    screen_space[:, :, 0] = (1.0 - verts_ndc[:, :, 0]) * 0.5 * img_w.reshape(-1, 1)
    screen_space[:, :, 1] = (1.0 - verts_ndc[:, :, 1]) * 0.5 * img_h.reshape(-1, 1)

    return screen_space[:, :, :2]


def tensor_dict_to_numpy_dict(v):
    nv = dict()
    for _k, _v in v.items():
        if isinstance(_v, torch.Tensor): 
            nv[_k] = _v.cpu().numpy()

        elif isinstance(_v, np.ndarray): 
            nv[_k] = _v.copy()

        elif isinstance(_v, dict):
            nv[_k] = tensor_dict_to_numpy_dict(_v)

        else:
            nv[_k] = _v

    return nv

def pa_serializify(v):
    if isinstance(v, dict):
        v = tensor_dict_to_numpy_dict(v)      

    elif isinstance(v, np.ndarray): 
        v = v.copy()

    pastring = pa.serialize(v).to_buffer().to_pybytes()
    pad_bytes = bytes(len(pastring) % 4096)
    value = pastring + pad_bytes

    return value


class LMDB:
    def __init__(self, path, lock=True):
        save_path = os.path.join(f'{path}/{GENERATION_MODE}')
        os.makedirs(save_path, exist_ok=True)
        self.env = lmdb.open(save_path, map_size=10099511627776, lock=lock, writemap=False)
        self.lmdb_index = 0
    
    def index(self):
        idx = self.lmdb_index 
        self.lmdb_index += 1
        return idx
    
    def create_lmdb_records_from_cache(self, loop_cache):
        outputs = list(loop_cache.values())

        outputs.append({
            'capture_id': None,
            'cam_index': None, 
            'handedness': None, 
            'camera_hand_info': None, 
            'event': None, 
            'image': None, 
            'depth': None, 
            'segmentation': None,  
        })

        for output in outputs:
            capture_id = output['capture_id']
            cam_index = output['cam_index']
            handedness = output['handedness']
            camera_hand_info = output['camera_hand_info']

            event = output['event']
            frame_color = output['image'] 
            depth_map = output['depth']
            segmentation = output['segmentation']

            cache_index = self.index()
            
            cache = dict()
            capture_idKey = 'capture_id-%09d'.encode() % cache_index
            cache[capture_idKey] = capture_id
        
            cam_indexKey = 'cam_index-%09d'.encode() % cache_index
            cache[cam_indexKey] = cam_index

            eventKey = 'event-%09d'.encode() % cache_index
            cache[eventKey] = event
                
            segmentationKey = 'segmentation-%09d'.encode() % cache_index
            cache[segmentationKey] = segmentation

            depthKey = 'depth-%09d'.encode() % cache_index
            cache[depthKey] = depth_map

            handednessKey = 'handedness-%09d'.encode() % cache_index
            cache[handednessKey] = handedness
            
            imageKey = 'image-%09d'.encode() % cache_index
            cache[imageKey] = frame_color

            camera_hand_infoKey = 'camera_hand_info-%09d'.encode() % cache_index
            cache[camera_hand_infoKey] = camera_hand_info

            # hand_infoKey = 'hand_info-%09d'.encode() % cache_index
            # cache[hand_infoKey] = hand_info

            cache['num-samples'.encode()] = str(cache_index)

            self.writeCache(cache)                

    def writeCacheThreaded(self, cache):
        with ThreadPoolExecutor(128) as workers:
            futures = dict()
            for k, v in cache.items():
                futures[k] = workers.submit(pa_serializify, v)
        
        with self.env.begin(write=True) as txn:
            for k, v in futures.items():
                value = futures[k].result()
                txn.put(k, value)

    def writeCache(self, cache):
        with self.env.begin(write=True) as txn:
            for k, v in cache.items():
                if isinstance(v, dict):
                    v = tensor_dict_to_numpy_dict(v)      
                
                t = time.time()
                pastring = pa.serialize(v).to_buffer().to_pybytes()
                pad_bytes = bytes(len(pastring) % 4096)
                value = pastring + pad_bytes
                print('s', time.time() -t )
                txn.put(k, value)
                print('f', time.time() -t)

                # print(v == value)
                # picklestring = pickle.dumps(v, 4)
                # picklestring = pickletools.optimize(picklestring)
                # pad_bytes = bytes(len(picklestring) % 4096)
                # obj = pickle.loads(v)
                # print(v == obj)
                # v = picklestring + pad_bytes
                # txn.put(k, v)
        # exit(0)

def transform_mesh(mesh, R, t):
    vertices = mesh.vertices
    vertices = np.dot(R, vertices.transpose(1,0)).transpose(1,0) + t.reshape(1,3) / 1000 # milimeter to meter. 
    mesh.vertices = vertices
    
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    
    return mesh


def load_camera_params(cam_params, cameras=None):
    if cameras: assert len(cam_params) == len(cameras)
    
    _cameras = list()
    transforms = list()
    for idx, cam_param in enumerate(cam_params):
        if cameras is None:
            focal = np.array(cam_param['intrinsics']['focal'], dtype=np.float32).reshape(2)
            princpt = np.array(cam_param['intrinsics']['princpt'], dtype=np.float32).reshape(2)
            
            camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        else:
            camera = cameras[idx]
        
        
        _cameras.append(pyrender.Node(camera=camera))
        transforms.append(partial(transform_mesh, R=cam_param['extrinsics']['R'], t=cam_param['extrinsics']['t']))

    return _cameras, transforms


def pad_image_to_target(img, pad_img):
    th, tw = pad_img.shape[:2]
    h, w = img.shape[:2]
    if h > w:
        img = imutils.resize(img, height=th)
    else:
        img = imutils.resize(img, width=tw)
    h, w = img.shape[:2]
    
    pad_img[:h, :w] = img
    return pad_img
    

def generate_hand_node(mesh, smooth=True):
    color_0, texcoord_0, primitive_material = pyrender.Mesh._get_trimesh_props(mesh, smooth=smooth, material=None)

    primitive_material = pyrender.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.5, 0.5, 0.5, 1.0],
        metallicFactor=0.5,
        roughnessFactor=0.7,
    )
    
    if smooth:
        positions = mesh.vertices.copy()
        normals = mesh.vertex_normals.copy()
        indices = mesh.faces.copy()
    else:
        positions = mesh.vertices[mesh.faces].reshape((3 * len(mesh.faces), 3))
        normals = np.repeat(mesh.face_normals, 3, axis=0)
        indices = None

    primitive = pyrender.Primitive(
                positions=positions,
                normals=normals,
                texcoord_0=texcoord_0,
                color_0=color_0,
                indices=indices,
                material=primitive_material,
                mode=pyrender.GLTF.TRIANGLES,
            )

    mesh_node = pyrender.Node(mesh=pyrender.Mesh(primitives=[primitive]))
    return mesh_node


class RGBDRenderer:
    def load_background_image(self):
        file_path = random.choice(glob(f'{DATA_PATH}/background_images/*.png'))
        
        image = cv2.imread(file_path)[:, :, ::-1]
        image = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        return image

    def augment_background(self, rgb):        
        img1 = np.copy(self.background_image)
        img2 = rgb

        rows,cols,channels = img2.shape
        roi = img1[0:rows, 0:cols]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg)
        img1[0:rows, 0:cols ] = dst
        
        return img1

    def generate_train_lights(self,):
        for light_node in self.lights: 
            self.scene.remove_node(light_node)

        light_pose = np.eye(4)

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=random.randrange(1, 5))            
        light_pose[:3, 3] = np.array([0, -1, 1]) + np.random.rand(3) / 10
        self.lights.append(pyrender.Node(light=light, matrix=light_pose))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=random.randrange(1, 5))            
        light_pose[:3, 3] = np.array([0, 1, 1]) + np.random.rand(3) / 10
        self.lights.append(pyrender.Node(light=light, matrix=light_pose))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=random.randrange(1, 5))            
        light_pose[:3, 3] = np.array([1, 1, 2]) + np.random.rand(3) / 10
        self.lights.append(pyrender.Node(light=light, matrix=light_pose))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=random.randrange(1, 5))
        light_pose[:3, 3] = (2 * np.random.rand(3) - 1) * 2  + np.random.rand(3) / 10
        self.lights.append(pyrender.Node(light=light, matrix=light_pose))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=random.randrange(1, 5))            
        light_pose[:3, 3] = (2 * np.random.rand(3) - 1) * 2  + np.random.rand(3) / 10
        self.lights.append(pyrender.Node(light=light, matrix=light_pose))

        for light_node in self.lights: 
            self.scene.add_node(light_node)


    def __init__(self, cam_params, renderer, **kwargs):
        self.background_image = self.load_background_image()

        self.cam_params = cam_params

        self.cameras, self.transforms = load_camera_params(cam_params, **kwargs)

        scene = pyrender.Scene(ambient_light=(0.1, 0.1, 0.1), bg_color=(0, 0, 0))
        self.scene = scene

        self.lights = list()

        if 'train' in GENERATION_MODE:
            self.generate_train_lights()
        else:
            light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10)

            light_pose = np.eye(4)
            light_pose[:3, 3] = np.array([0, -1, 1])
            scene.add(light, pose=light_pose)
            light_pose[:3, 3] = np.array([0, 1, 1])
            scene.add(light, pose=light_pose)
            light_pose[:3, 3] = np.array([1, 1, 2])
            scene.add(light, pose=light_pose)



        self.renderer = renderer
        
    def __call__(self, meshes):
        # if 'train' in GENERATION_MODE:
        self.generate_train_lights()

        images = list()
        depth_map = list()
        for idx, camera_node in enumerate(self.cameras):
            self.scene.add_node(camera_node)
            transform = self.transforms[idx]

            mesh_nodes = list()
            for mesh in meshes:
                mesh = transform(mesh=mesh)
                mesh_node = generate_hand_node(mesh)
                self.scene.add_node(mesh_node)
                mesh_nodes.append(mesh_node)
                    
            rgb, depth = self.renderer.render(self.scene)

            rgb = self.augment_background(rgb)        
                
            for mesh_node in mesh_nodes:
                self.scene.remove_node(mesh_node)

            images.append(rgb[None, ...])
            depth_map.append(depth[None, ...])

            self.scene.remove_node(camera_node)

        images = np.concatenate(images, axis=0)
        depth_map = np.concatenate(depth_map, axis=0)

        N = depth_map.shape[0]
        
        depth_max = np.max(depth_map.reshape(N, -1), 1)[..., None, None] 
        depth_min = np.min(depth_map.reshape(N, -1), 1)[..., None, None]
    
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return images, depth_map        


class SegmentationRenderer:
    def __init__(self, cam_params, renderer, **kwargs):
        self.cam_params = cam_params
        self.cameras, self.transforms = load_camera_params(cam_params, **kwargs)

        scene = pyrender.Scene(ambient_light=(2, 2, 2), bg_color=(0, 0, 0))

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)
        
        self.renderer = renderer
        self.scene = scene
        
    def __call__(self, meshes, mesh_transform=True):
        images = list()
        for idx, camera_node in enumerate(self.cameras):
            self.scene.add_node(camera_node)

            mesh_nodes = list()
            for mesh in meshes:
                if mesh_transform:
                    mesh = self.transforms[idx](mesh=mesh)
                
                mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh))
                self.scene.add_node(mesh_node)
                mesh_nodes.append(mesh_node)
            
            rgb, _ = self.renderer.render(self.scene)
                        
            for mesh_node in mesh_nodes:
                self.scene.remove_node(mesh_node)

            images.append(rgb[None, ...])
        
            self.scene.remove_node(camera_node)

        images = np.concatenate(images, axis=0) 
        images = np.argmax(images, axis=3).astype(dtype=np.uint8)
        
        return images


class GTMANORenderer():
    def __init__(self, renderer, camera):

        scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        camera_pose = np.eye(4)
    
        self.rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        scene.add(camera, pose=camera_pose)

        self.scene = scene
        self.renderer = renderer

    def __call__(self, camera_hand_info):

        hand_meshs = list()
        for hand_type in ['left', 'right']:
            verts = camera_hand_info[hand_type]['verts'] * 1000
            faces = camera_hand_info[hand_type]['faces']

            gt_mesh = trimesh.Trimesh(verts, faces)
            gt_mesh.visual.vertex_colors = [0, 255, 0]

            hand_meshs.append(gt_mesh)

        hand_meshs = trimesh.util.concatenate(hand_meshs)
        hand_meshs.apply_transform(self.rot)

        mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(hand_meshs))
        self.scene.add_node(mesh_node)
            
        rgb, depth = self.renderer.render(self.scene)          
        self.scene.remove_node(mesh_node)


        return rgb