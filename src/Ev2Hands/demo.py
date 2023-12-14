import sys; sys.path.append('..') # add settings.
import os
import torch
import cv2
import time
import pyrender
import numpy as np
import trimesh

import arg_parser

from torch.utils.data import DataLoader
from dataset import Ev2HandRDataset
from model import TEHNetWrapper
from settings import OUTPUT_HEIGHT, OUTPUT_WIDTH, MAIN_CAMERA, REAL_TEST_DATA_PATH


def demo(net, device, batch):
    net.eval()

    events = batch['events']
    events = events.to(device=device, dtype=torch.float32)

    start_time = time.time()
    with torch.no_grad():
        outputs = net(events)

    # Waits for everything to finish running
    torch.cuda.synchronize()
    end_time = time.time()

    N = events.shape[0]
    print(end_time - start_time)

    outputs['class_logits'] = outputs['class_logits'].softmax(1).argmax(1).int().cpu()
    
    frames = list()
    for idx in range(N):
        hands = dict()

        hands['left'] = {
            'vertices': outputs['left']['vertices'][idx].cpu(),
            'j3d': outputs['left']['j3d'][idx].cpu(),
        }

        hands['right'] = {
            'vertices': outputs['right']['vertices'][idx].cpu(),
            'j3d': outputs['right']['j3d'][idx].cpu(),
        }

        coordinates = batch['coordinates'][idx]

        seg_mask = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
        for edx, (y, x) in enumerate(coordinates):
            y, x = y.int(), x.int()

            cid = outputs['class_logits'][idx][edx]            

            if cid == 3:
                seg_mask[y, x] = 255
            else:
                seg_mask[y, x, cid] = 255

        hands['seg_mask'] = seg_mask

        frames.append(hands)

    return frames



def main():    
    arg_parser.demo()
    os.makedirs('outputs', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = TEHNetWrapper(device=device)

    save_path = os.environ['CHECKPOINT_PATH']     
    batch_size = int(os.environ['BATCH_SIZE'])

    checkpoint = torch.load(save_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    
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
    
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    
    mano_hands = net.hands

    if os.path.exists(REAL_TEST_DATA_PATH) and len(os.listdir(REAL_TEST_DATA_PATH)) > 0:
        data = REAL_TEST_DATA_PATH
    else:
        data = '../data/demo/example.pickle'
    
    validation_dataset = Ev2HandRDataset(data, demo=True, augment=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    video_fps = 25
    video = cv2.VideoWriter('outputs/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (3 * OUTPUT_WIDTH, OUTPUT_HEIGHT))
    
    for bdx, batch in enumerate(validation_loader):
        frames = demo(net=net, device=device, batch=batch)

        for idx, frame in enumerate(frames):
            event_frame = batch['event_frame'][idx].cpu().numpy().astype(dtype=np.uint8)
            seg_mask = frame['seg_mask']

            pred_meshes = list()
            for hand_type in ['left', 'right']:
                faces = mano_hands[hand_type].faces

                pred_mesh = trimesh.Trimesh(frame[hand_type]['vertices'].cpu().numpy() * 1000, faces)
                pred_mesh.visual.vertex_colors = [255, 0, 0]
                pred_meshes.append(pred_mesh)
    
            pred_meshes = trimesh.util.concatenate(pred_meshes)
            pred_meshes.apply_transform(rot)

            camera = MAIN_CAMERA

            nc = pyrender.Node(camera=camera, matrix=np.eye(4))
            scene.add_node(nc)

            mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(pred_meshes))
            scene.add_node(mesh_node)            
            pred_rgb, depth = renderer.render(scene)           
            scene.remove_node(mesh_node)
            scene.remove_node(nc)

            pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)           
            pred_rgb[pred_rgb == 255] = 0

            img_stack = np.hstack([event_frame, seg_mask, pred_rgb])
            video.write(img_stack)

            cv2.imshow('image', img_stack)
            c = cv2.waitKey(1)

            if c == ord('q'):
                video.release()
                exit(0)
            
    video.release()


if __name__ == '__main__':
    main()
 