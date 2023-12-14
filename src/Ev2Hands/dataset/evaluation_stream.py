import pickle
import torch
import numpy as np
from dv import AedatFile

from settings import OUTPUT_HEIGHT, OUTPUT_WIDTH
from camera import undistort, opencv_camera_view_to_screen_space_transform


WINDOWS_SIZE = 2
OVERLAP_TIME = 1

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


class EvalutaionStream:
    def __init__(self, file_path):
        if file_path.endswith('pickle'):
            with open(file_path, 'rb') as pickle_file:
                data = pickle.load(pickle_file)

            self.events = data['events']
            self.joints = data['joints'] / 1000 # mm to meter
            self.camera = data['camera']

            xy = undistort(self.events[:, :2][:, None, ...].astype(np.float32), self.camera['camera_matrix'], self.camera['dist'], OUTPUT_WIDTH, OUTPUT_HEIGHT)
            self.events[:, :2] = xy
        else:
            with AedatFile(file_path) as f:
                events = np.hstack([packet for packet in f['f'].numpy()])

            self.events = np.vstack([events['x'], events['y'], events['timestamp'], events['polarity']]).T
            self.joints = np.zeros([1, 2, 21, 3])
            self.camera = {'projection_matrix': np.eye(4)[:3,:]}

        self.e_id = 0
        self.n_events = 0

    def next_event_count(self, n_events):
        self.e_id += n_events
        self.n_events = 0

    @property
    def total_events(self):
        return len(self.events)

    def next_event_time(self):
        self.n_events = 0

        xs, ys, ts, polarity, frame_index = self.get_event()
        
        start_time = ts 
        end_time = ts

        while True:
            xs, ys, ts, polarity, frame_index = self.get_event()

            end_time = ts

            elapsed_time_ms = abs(end_time - start_time)
        
            if elapsed_time_ms > OVERLAP_TIME:
                break
            
            self.n_events += 1
            
        self.e_id += self.n_events
        self.n_events = 0

    def get_event(self):
        events = self.events
        idx = self.e_id + self.n_events
    
        try:
            event = events[idx]
        except IndexError:
            raise StopIteration
            
        xs, ys, ts, polarity = event[:4]

        if event.shape[0] == 5:
            frame_index = event[-1]
        else:
            frame_index = -1 

        self.n_events += 1

        return xs, ys, ts * 1e-3, polarity, frame_index

    def get_events_by_counts(self, n_events):
        xs, ys, ts, polarity, frame_index = self.get_event()
         
        frame_indices = [frame_index]
        events = [[xs, ys, ts, polarity]]

        event_counter = 0
        while True:
            xs, ys, ts, polarity, frame_index = self.get_event()

            end_time = ts

            event_counter += 1
            events.append([xs, ys, ts, polarity])
            frame_indices.append(frame_index)

            if event_counter > n_events: break

        return np.array(events), np.array(frame_indices)

    def get_events_by_time(self, windows_size=None): 
        if windows_size is None: windows_size = WINDOWS_SIZE

        xs, ys, ts, polarity, frame_index = self.get_event()
        
        start_time = ts 
        end_time = ts

        frame_indices = [frame_index]
        events = [[xs, ys, ts, polarity]]
        while True:
            xs, ys, ts, polarity, frame_index = self.get_event()

            end_time = ts
            elapsed_time_ms = abs(end_time - start_time)

            if elapsed_time_ms > windows_size and len(events) >= 2048:
                break

            events.append([xs, ys, ts, polarity])
            frame_indices.append(frame_index)

        return np.array(events), np.array(frame_indices)

    def get_current_frame_3d_joint(self, frame_indices):
        if isinstance(frame_indices, (list, tuple, np.ndarray)):
            frame_indices, counts = np.unique(frame_indices, return_counts=True)
        
        try:
            joints = self.joints[frame_indices]
        except IndexError:
            raise StopIteration
        
        return joints

    def get_current_frame_2d_joint(self, frame_indices):        
        joints = self.get_current_frame_3d_joint(frame_indices)
        
        return opencv_camera_view_to_screen_space_transform(self.camera['camera_matrix'], joints  * 1000)


class ERPCParser(EvalutaionStream):
    def __init__(self, path) -> None:
        super().__init__(path)

        self.nSamples = len(self.events)

    def __len__(self):
        return self.nSamples

    def __iter__(self):
        return self

    def __getitem__(self, index):        
        n_events = 2048

        events, frame_indices = self.get_events_by_time()
        self.next_event_time()
            
        joints_3d = self.get_current_frame_3d_joint(frame_indices)
        joints_3d = joints_3d[:1]
        # joints_2d = self.get_current_frame_2d_joint(frame_indices)

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

        events = np.hstack([xi[..., None], yi[..., None], t_avg[..., None], p_evn[..., None], n_evn[[..., None]]])

        sampled_indices = np.random.choice(events.shape[0], n_events)
        events = events[sampled_indices]

        events = torch.tensor(events, dtype=torch.float32)

        events[:, :3] = pc_normalize(events[:, :3])

        events = events
        joints_3d = torch.tensor(joints_3d) 
        # joints_2d = torch.tensor(joints_2d) 


        values, counts = np.unique(frame_indices, return_counts=True)
        frame_index = values[np.argmax(counts)]

        data = dict()
        data['data'] = events.permute(1, 0)
        data['j3d'] = joints_3d
        # data['j2d'] = joints_2d

        data['frame_index'] = frame_index

        return data
