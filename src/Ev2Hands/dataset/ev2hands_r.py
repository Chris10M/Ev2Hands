import os
import random
import numpy as np
import torch
from settings import OUTPUT_WIDTH, OUTPUT_HEIGHT
from .evaluation_stream import EvalutaionStream

os.environ['ERPC'] = '1'


WINDOWS_SIZE = 2


def augment_events(events):
    augment = np.random.rand(events.shape[0]) < 0.5
    events[augment][:, -1] = np.abs(1 - events[augment][:, -1])
           
    return events


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


class Ev2HandRDataset:
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

    def __init__(self, root, augment=True, demo=False):
        self.root = root
        self.demo = demo

        self.paths = list()
        if not os.path.isdir(root):
            self.paths.append(root)
        else:
            self.paths += [os.path.join(root, filename) for filename in os.listdir(root)]

        self.streams = [EvalutaionStream(path) for path in self.paths]

        self.sample_indices = list()
        for stream_id, stream in enumerate(self.streams):
            n_events = len(stream.events)

            stream_ids = np.ones(n_events, dtype=np.int32) * stream_id
            indices = np.arange(0, n_events, 1, dtype=np.int32)
            
            ids = np.concatenate([stream_ids[..., None], indices[..., None]], 1)

            self.sample_indices.append(ids)

        self.sample_indices = np.concatenate(self.sample_indices, 0)
        
        self.nSamples = len(self.sample_indices)
    
        print(f'root: {root} nSamples: {self.nSamples}')


        self.augment = augment

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        stream_id, event_index = self.sample_indices[index]
        
        stream = self.streams[stream_id]
        try:
            stream.e_id = event_index
            stream.n_events = 0

            events, frame_indices = stream.get_events_by_time(np.random.randint(1, WINDOWS_SIZE+1))
        except StopIteration:
            index = random.randint(0, index - 2048)
            return self[index]

        if self.augment and random.random() > 0.5:
            events = augment_events(events)


        n_events = 2048

        events[:, 2] -= events[0, 2] # normalize ts

        event_grid = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.float32)
        count_grid = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.float32)

        x, y, t, p = events.T
        x, y = x.astype(dtype=np.int32), y.astype(dtype=np.int32)

        np.add.at(event_grid, (y, x, 0), t)
        np.add.at(event_grid, (y, x, 1), p == 1)
        np.add.at(event_grid, (y, x, 2), p != 1)

        np.add.at(count_grid, (y, x), 1)


        yi, xi = np.nonzero(count_grid)
        t_avg = event_grid[yi, xi, 0] / count_grid[yi, xi] 
        p_evn = event_grid[yi, xi, 1] 
        n_evn = event_grid[yi, xi, 2]
 
        events = np.hstack([xi[:, None], yi[:, None], t_avg[:, None], p_evn[:, None], n_evn[:, None]])

        sampled_indices = np.random.choice(events.shape[0], n_events)
        events = events[sampled_indices]
        frame_indices = frame_indices[sampled_indices]

        unique, counts = np.unique(frame_indices, return_counts=True)
        frame_index = sorted(zip(counts, unique), reverse=True, key=lambda x:x[0])[0][1]

        try:        
            joints_3d = stream.get_current_frame_3d_joint(frame_index)
            joints_2d = stream.get_current_frame_2d_joint(frame_index)
        except:
            index = random.randint(0, index - 2048)
            return self[index]

        events = torch.tensor(events, dtype=torch.float32)

        if self.demo:
            coordinates = np.zeros((events.shape[0], 2))
            event_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            for idx, (x, y, t_avg, p_evn, n_evn) in enumerate(events):
                y, x = y.int(), x.int()
                
                coordinates[idx] = (y, x)
                event_frame[y, x, 0] = (p_evn / (p_evn + n_evn)) * 255
                event_frame[y, x, -1] = (n_evn / (p_evn + n_evn)) * 255


        events[:, :3] = pc_normalize(events[:, :3])

        handedness = [1, 1]
        hand_data = {
            'mano_gt': 0.0,
            'events': events.permute(1, 0),
        }

        if self.demo:
            hand_data['event_frame'] = torch.tensor(event_frame, dtype=torch.uint8)
            hand_data['coordinates'] = torch.tensor(coordinates, dtype=torch.float32)


        for hdx, hand_type in enumerate(['left', 'right']):
            j3d = joints_3d[hdx]
            j2d = joints_2d[hdx]

            hand_data[hand_type] = {
                'j3d': torch.tensor(j3d, dtype=torch.float32),
                'j2d': torch.tensor(j2d, dtype=torch.float32),
                'valid': True
            }

        hand_data['handedness'] = torch.tensor(handedness, dtype=torch.int)
        
        return hand_data
