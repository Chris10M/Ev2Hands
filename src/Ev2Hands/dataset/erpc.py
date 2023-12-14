import sys; sys.path.append('../..') # add settings.
import numpy as np
np.random.seed(0)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import traceback; sys.path.append('..') # add settings.
import torch
import random
import os
import pickle 
import h5py
import sys

from torch.utils.data import Dataset

from .augmentations import augment_events
from settings import LNES_WINDOW_MS, OUTPUT_WIDTH, OUTPUT_HEIGHT
os.environ['ERPC'] = '1'


def pc_normalize(pc):
    pc[:, 0] /= OUTPUT_WIDTH
    pc[:, 1] /= OUTPUT_HEIGHT
    pc[:, :2] = 2 * pc[:, :2] - 1
    
    ts = pc[:, 2:]
    
    t_max = ts.max(0).values
    t_min = ts.min(0).values

    ts = (2 * ((ts - t_min) / (t_max - t_min))) - 1

    pc[:, 2:] = ts

    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = torch.zeros((npoint,))
    distance = torch.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)

    return centroids.long()


class Ev2HandSDataset(Dataset):
    @classmethod
    def to_device(cls, v, device):
        if isinstance(v, torch.Tensor): 
            nv = v.to(device)

        elif isinstance(v, dict):
            nv = dict()
            for _k, _v in v.items():
                nv[_k] = cls.to_device(_v, device)
                
        elif isinstance(v, (list, tuple)):
            nv = list()
            for i in range(0, len(v)):
                nv.append(cls.to_device(v[i], device))
            
        return nv

    def get_values(self, keys):
        values = list()
        try:
            
            with self.env.begin(write=False) as txn:

                cursor = txn.cursor()
                for key, value in cursor.getmulti([key.encode() for key in keys]):
                    # value = pickle.loads(value)
                    value = pa.deserialize(value)

                    values.append(value)

                # for key in keys:
                #     values.append(pa.deserialize(txn.get(key.encode())))
        except:
            traceback.print_exc()

        if len(values) != len(keys):
            raise IndexError

        return values

    def __init__(self, root, augment=True, demo=False, sampling=True, indices=None):
        self.root = root
        self.demo = demo

        file_path = f'{self.root}.h5' 

        annotation_path = f'{self.root}_anno.pickle'
        with open(annotation_path, 'rb') as f:
            self.annotations = pickle.load(f)

        self.file = h5py.File(file_path, 'r')
        self.dataset = self.file['event']

        self.nSamples = len(self.dataset)

        print(f'root: {root} nSamples: {self.nSamples}')

        self.augment = augment
        self.sampling = sampling

    def __len__(self):
        return self.nSamples

    def reset(self):
        idx = random.randint(0, len(self))
        return self[idx]

    def get_previous_camera_info(self, index, camera_hand_info, hand_data):
        try:
            camera_hand_infoKey = 'camera_hand_info-%09d' % (index - 1)
            camera_hand_info,  = self.get_values([camera_hand_infoKey])                
        except IndexError:
            ...

        for hand_type, hand in camera_hand_info.items():
            global_orient = torch.tensor(hand['global_orient'], dtype=torch.float32)
            hand_pose = torch.tensor(hand['hand_pose'], dtype=torch.float32)
            shape = torch.tensor(hand['shape'], dtype=torch.float32)
            trans = torch.tensor(hand['trans'], dtype=torch.float32)

            hand_data[hand_type]['previous'] = {
                'global_orient': global_orient,
                'hand_pose': hand_pose,
                'shape': shape,
                'trans': trans,
                'valid': True
            }

        return hand_data

    def get_item(self, index):
        eventsKey = 'events-%09d' % index
        event_labelsKey = 'event_labels-%09d' % index

        camera_hand_infoKey = 'camera_hand_info-%09d' % index

        frame_info = self.get_values([eventsKey, event_labelsKey, camera_hand_infoKey])                
        events, event_labels, camera_hand_info = frame_info

        events = events.astype(dtype=np.float32)
        event_labels = event_labels.astype(dtype=np.int32)
        
        return events, event_labels, camera_hand_info

    def __getitem__(self, index):        
        n_events = 2048

        start_index = index
        end_index = index + n_events
        data = self.dataset[start_index: end_index]

        x, y, t, p, annotation_index, event_labels = data.T

        event_grid = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.float32)
        count_grid = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.float32)
            
        x, y = x.astype(dtype=np.int32), y.astype(dtype=np.int32)

        np.add.at(event_grid, (y, x, 0), t)
        np.add.at(event_grid, (y, x, 1), p == 1)
        np.add.at(event_grid, (y, x, 2), p != 1)

        np.add.at(count_grid, (y, x), 1)

        yi, xi = np.nonzero(count_grid)
        t_avg = (event_grid[yi, xi, 0] / count_grid[yi, xi]) * 1e-6 # ns to ms 
        p_evn = event_grid[yi, xi, 1] 
        n_evn = event_grid[yi, xi, 2]

        events = np.hstack([xi[:, np.newaxis], yi[:, np.newaxis], t_avg[:, np.newaxis], 
                            p_evn[:, np.newaxis], n_evn[:, np.newaxis]])

        events = events.astype(dtype=np.float32)
        event_labels = event_labels.astype(dtype=np.int32)

        annotation_index = annotation_index[-1]
        camera_hand_info = self.annotations[annotation_index]
                
        if self.augment and random.random() > 0.5:
            events, event_labels = augment_events(events, event_labels)

        indices = np.argsort(events[:, 2])
        events = events[indices]
        event_labels = event_labels[indices]

        events[:, 2] -= events[0, 2] # normalize ts
        
        if self.sampling:
            sampled_indices = np.random.choice(events.shape[0], n_events)
            try:
                events = events[sampled_indices]
                event_labels = event_labels[sampled_indices]
            except TypeError:
                return self.reset()
        else:
            if events.shape[0] < n_events:
                sampled_indices = np.random.choice(events.shape[0], n_events - events.shape[0])
                try:
                    events = np.concatenate([events, events[sampled_indices]], 0)
                    event_labels = np.concatenate([event_labels, event_labels[sampled_indices]], 0)
                except TypeError:
                    return self.reset()

        events = torch.tensor(events, dtype=torch.float32)
        event_labels = torch.tensor(event_labels, dtype=torch.long)

        if self.demo:
            coordinates = np.zeros((events.shape[0], 2))
            segmentation_mask = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            event_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            for idx, (x, y, t_avg, p_evn, n_evn) in enumerate(events):
                y, x = y.int(), x.int()
                
                coordinates[idx] = (y, x)
                event_frame[y, x, 0] = (p_evn / (p_evn + n_evn)) * 255
                event_frame[y, x, 1] = (n_evn / (p_evn + n_evn)) * 255

                cid = event_labels[idx].int()                
                if cid == 3:
                    segmentation_mask[y, x] = 255
                else:
                    segmentation_mask[y, x, cid] = 255


        events[:, :3] = pc_normalize(events[:, :3])

        handedness = [0, 0]
        hand_data = {
            # 'index': current_index,
            'mano_gt': 1.0,
            'events': events.permute(1, 0),
            'class_logits': event_labels,
        }


        if self.demo:
            # hand_data['image'] = torch.tensor(image, dtype=torch.uint8)
            hand_data['event_frame'] = torch.tensor(event_frame, dtype=torch.uint8)
            hand_data['segmentation_mask'] = torch.tensor(segmentation_mask, dtype=torch.uint8)
            hand_data['coordinates'] = torch.tensor(coordinates, dtype=torch.float32)
        
        for hand_type, hand in camera_hand_info.items():
            handedness = [1, 1] # there is some hand in the hand_info.

            # j3d = torch.tensor(hand['j3d'], dtype=torch.float32)

            global_orient = torch.tensor(hand['global_orient'], dtype=torch.float32)
            hand_pose = torch.tensor(hand['hand_pose'], dtype=torch.float32)
            shape = torch.tensor(hand['shape'], dtype=torch.float32)
            trans = torch.tensor(hand['trans'], dtype=torch.float32)

            hand_data[hand_type] = {
                'global_orient': global_orient,
                'hand_pose': hand_pose,
                'shape': shape,
                'trans': trans,
                'valid': True
            }
        
        if 'left' not in hand_data:
            hand_data['left'] = hand_data['right']
            hand_data['left']['valid'] = False
            handedness[0] = 0

        elif 'right' not in hand_data:
            hand_data['right'] = hand_data['left']
            hand_data['right']['valid'] = False
            handedness[1] = 0

        hand_data['handedness'] = torch.tensor(handedness, dtype=torch.int)
        
        # hand_data = self.get_previous_camera_info(current_index, camera_hand_info, hand_data)

        return hand_data


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    
    worker_id = worker_info.id

    dataset.start_index = worker_id
