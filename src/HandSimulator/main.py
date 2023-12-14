import os; os.environ['PYOPENGL_PLATFORM']='egl'
import sys; sys.path.append('..') # add settings.
from settings import SIMULATOR_FPS, GENERATION_MODE, MAIN_CAMERA, INTERPOLATION_FPS, NUMBER_OF_AUGMENTATED_SEQUENCES, AUGMENTATED_SEQUENCE, RENDER_SMPLX, ROOT_TRAIN_DATA_PATH, OUTPUT_HEIGHT, OUTPUT_WIDTH, INTERHAND_ROOT_PATH
from renderer import Renderer
import torch
import cv2
import random
import time
import numpy as np
import pickle
from functools import partial
from tqdm import tqdm

import augmentations
from twohands import TwoHands
from dataset import InterHand, AAtoPCA
from dataset import CameraTransform
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, TimeoutError

N_WORKERS = int(os.getenv('N_WORKERS', 1))
WORKER_ID = int(os.getenv('WORKER_ID', 0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loop(job_id, save_index, mano_image_paths, mano_sequence, capture_id, cam_index, camera_transform):
    start_time = time.time()
    
    class File:
        save_path = os.path.join(f'{ROOT_TRAIN_DATA_PATH}/{GENERATION_MODE}/parts') 
        id = 0
    os.makedirs(File.save_path, exist_ok=True)
    
    def write_cache(cache):
        file_name = f'{job_id}_{File.id}.pickle'
        file_path = f'{File.save_path}/{file_name}'

        with open(file_path, 'wb') as pickle_file:
            pickle.dump(cache, pickle_file)            

        File.id += 1


    cache = dict()
    
    image_paths = mano_image_paths[cam_index]
    frame_keys = sorted(list(image_paths.keys()), key=lambda v: int(v))
        
    cam_param = camera_transform.get_camera_param(capture_id, cam_index)
    camera_params = [cam_param]

    two_hands = TwoHands(mano_sequence, device=device, compute_smplx=RENDER_SMPLX)
    renderer = Renderer(device, two_hands, camera_params, fps=SIMULATOR_FPS, cameras=[MAIN_CAMERA,])
    mano_transform = partial(CameraTransform.transform_mano_params, self=camera_transform, mano_layer=two_hands.mano_hands, R=cam_param['extrinsics']['R'], t=cam_param['extrinsics']['t'], camera=MAIN_CAMERA)

    for frame_index in (range(0, len(two_hands))):

        two_hands_output = two_hands[frame_index]

        camera_hand_info = dict()
        for hand_type, mano_param in two_hands_output['hand_info'].items():
            camera_hand_info[hand_type] = mano_transform(hand_type=hand_type, mano_param=mano_param)
            
        handedness = np.zeros(2)
        for hand in two_hands_output['hand_info'].keys():
            if hand == 'left': handedness[0] = 1
            else: handedness[1] = 1
        
        output = renderer(two_hands_output)
        if output is None: continue


        cache_index = save_index
        save_index += 1


        event = output['event']
        frame_color = output['image'] 
        depth_map = output['depth']
        segmentation = output['segmentation']
            
        t, x, y, p = event
        x, y = x.astype(dtype=np.int32), y.astype(dtype=np.int32)

        events = np.hstack([x[..., None], y[..., None], t[..., None], p[..., None]])
        event_labels = segmentation[y, x].astype(dtype=np.uint8)

        write_frame = False
        show_frame = False

        if write_frame or show_frame:
            ts, xs, ys, ps = event
            h, w = frame_color.shape[:2]
            event_bgr = np.zeros((h, w, 3), dtype=np.uint8)
            for x, y, p in zip(xs, ys, ps):
                event_bgr[y, x, 0 if p == -1 else 2] = 255
                
            image_path = image_paths[frame_keys[frame_index]]
            rgb_image = cv2.imread(image_path)

        if write_frame:
            print(ts.shape[0])
            os.makedirs(f'outputs/rgb_images/', exist_ok=True)
            os.makedirs(f'outputs/images/', exist_ok=True)
            os.makedirs(f'outputs/events/', exist_ok=True)
            os.makedirs(f'outputs/segmentation/', exist_ok=True)

            cv2.imwrite(f'outputs/images/{frame_index}_frame_color.png', frame_color[:, :, ::-1])
            cv2.imwrite(f'outputs/segmentation/{frame_index}_segmentation.png', segmentation * 100)
            cv2.imwrite(f'outputs/events/{frame_index}.png', event_bgr)
            cv2.imwrite(f'outputs/rgb_images/{frame_index}.png', rgb_image)

        if show_frame:
            print(ts.shape[0])
            cv2.imshow('frame_color.png', frame_color[:, :, ::-1])
            cv2.imshow('segmentation.png', segmentation * 100)
            cv2.imshow('rgb_image.png', rgb_image)
            cv2.imshow('events.png', event_bgr)

            cv2.waitKey(1)

        eventKey = 'events-%09d'.encode() % cache_index
        cache[eventKey] = events

        eventKey = 'event_labels-%09d'.encode() % cache_index
        cache[eventKey] = event_labels
        
        imageKey = 'image-%09d'.encode() % cache_index
        cache[imageKey] = frame_color

        camera_hand_infoKey = 'camera_hand_info-%09d'.encode() % cache_index
        cache[camera_hand_infoKey] = camera_hand_info
            
        if (frame_index + 1) % 1000 == 0:
            write_cache(cache)
            cache = {}            
            print(f'Written {frame_index + 1} of  {len(two_hands)} in {time.time() - start_time}')
            start_time = time.time()
    
    write_cache(cache)
    renderer.close()

    return save_index


def get_number_of_jobs(interhand, input_fps, camera_transform, n_aug_count):
    count = 0
    jobs = list()
    for augmentation_count in range(0, n_aug_count): 
        for idx in range(len(interhand)):
            mano_data = interhand[idx]
            capture_id = mano_data['capture_id']
            camera_indices = camera_transform.get_camera_indices(capture_id)
            mano_sequence = augmentations.interpolate_sequence(mano_data['mano_data'], input_fps, INTERPOLATION_FPS)
            mano_image_paths = mano_data['image_paths']
            
            for cam_index in camera_indices:
                if count % N_WORKERS == WORKER_ID:
                    jobs.append(True)
                else:
                    jobs.append(False)
                
                count += 1


    return jobs


def main(): 
    n_workers = 4
    workers = ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context('spawn'))

    n_aug_count = NUMBER_OF_AUGMENTATED_SEQUENCES if ('train' in GENERATION_MODE and AUGMENTATED_SEQUENCE) else 1
    mode = GENERATION_MODE
    
    input_fps = 5
    root_path = INTERHAND_ROOT_PATH
    camera_transform = CameraTransform(root_path, mode=mode)
    
    interhand = InterHand(root_path=root_path, mode=mode)
    interhand[0]

    jobs = get_number_of_jobs(interhand, input_fps, camera_transform, n_aug_count)
    n_jobs = sum(jobs)
    print(n_jobs, jobs)

    pbar = tqdm(total=n_jobs + 1)

    aa_to_pca = AAtoPCA()

    current_job = 0
    futures = list()
    count = 0
    for augmentation_count in range(0, n_aug_count): 
        for idx in range(len(interhand)):
            mano_data = interhand[idx]
            capture_id = mano_data['capture_id']
            
            camera_indices = camera_transform.get_camera_indices(capture_id)
            
            mano_sequence = augmentations.interpolate_sequence(mano_data['mano_data'], input_fps, INTERPOLATION_FPS)
            mano_sequence = aa_to_pca.compute_mano_sequence(mano_sequence)
            mano_image_paths = mano_data['image_paths']
            
            for cam_index in camera_indices:
                start_index = count

                if jobs[current_job]:
                    future = workers.submit(loop, current_job, start_index, mano_image_paths, mano_sequence, capture_id, cam_index, camera_transform) 
                    futures.append(future)

                current_job += 1
                count += len(mano_sequence)

                if len(futures) > n_workers:      
                    for future in futures: future.result()              
                    futures = list()
                    pbar.update(n_workers)

    pbar.update(len(futures))
    for future in futures: future.result()
    pbar.close()


if __name__ == '__main__':
    main()
