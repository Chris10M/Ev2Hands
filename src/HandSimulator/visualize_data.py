import os
import sys; sys.path.append('..') # add settings.
import cv2
import pickle
import pyrender
import numpy as np
import h5py
import torch
import trimesh

from tqdm import trange
from settings import GENERATION_MODE, ROOT_TRAIN_DATA_PATH, OUTPUT_HEIGHT, OUTPUT_WIDTH, MAIN_CAMERA
from twohands import TwoHands


def main():
    assert GENERATION_MODE in ['train', 'test', 'val']
    path = f'{ROOT_TRAIN_DATA_PATH}/{GENERATION_MODE}.h5'

    print(f'Loading {path}...')

    annotation_path = f'{ROOT_TRAIN_DATA_PATH}/{GENERATION_MODE}_anno.pickle'
    if not os.path.exists(annotation_path): 
        print(f'{annotation_path} does not exist.')
        return
    
    with open(annotation_path, 'rb') as pickle_file:
        annotations = pickle.load(pickle_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    two_hands = TwoHands({}, device, False)

    renderer = pyrender.OffscreenRenderer(viewport_width=OUTPUT_WIDTH, viewport_height=OUTPUT_HEIGHT)

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    nc = pyrender.Node(camera=MAIN_CAMERA, matrix=np.eye(4))
    scene.add_node(nc)

    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

    N_EVENTS = 2048
    with h5py.File(path, 'r') as f:
        dataset = f['event']

        for idx in trange(0, len(dataset), N_EVENTS):
            xs, ys, t, ps, annotation_index, event_labels = dataset[idx:idx+N_EVENTS].T

            annotation = annotations[int(annotation_index[-1])]

            h, w = OUTPUT_HEIGHT, OUTPUT_WIDTH
            event_bgr = np.zeros((h, w, 3), dtype=np.uint8)
            segmentation_mask = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            for edx, (x, y, p) in enumerate(zip(xs, ys, ps)):
                event_bgr[y, x, 0 if p == -1 else 2] = 255

                cid = event_labels[edx]                
                if cid == 3:
                    segmentation_mask[y, x] = 255
                else:
                    segmentation_mask[y, x, cid] = 255

            meshes = []
            for hand_type, hand in annotation.items():
                global_orient = hand['global_orient']
                hand_pose = hand['hand_pose']
                shape = hand['shape']
                trans = hand['trans']

                verts, j3d, faces = two_hands.infer_mano(hand_type, global_orient, hand_pose, shape, trans)

                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                meshes.append(mesh)

            meshes = trimesh.util.concatenate(meshes)
            meshes.apply_transform(rot)

            mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(meshes))
            scene.add_node(mesh_node)            

            rgb, depth = renderer.render(scene)           

            image_stack = np.hstack([event_bgr, segmentation_mask, rgb])
            cv2.imshow('image_stack', image_stack)
            c = cv2.waitKey(1) 

            scene.remove_node(mesh_node)

            if c == ord('q'):
                break


if __name__ == '__main__':
    main()
