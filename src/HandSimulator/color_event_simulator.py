import sys
from tracemalloc import start; sys.path.append('..')
from settings import ESIM_POSITIVE_THRESHOLD, ESIM_NEGATIVE_THRESHOLD, OUTPUT_WIDTH, OUTPUT_HEIGHT, SIMULATOR_FPS, LNES_WINDOW_MS
from esim_torch.esim_torch import EventSimulator_torch


#!/usr/bin/env python3
import numpy as np
import sys
import time
import os
import cv2
from tqdm import tqdm
import imageio as io

# inp_folder = sys.argv[1]
# out_file = sys.argv[2]
# THR = float(sys.argv[3])
import torch
import numba
import numpy as np
from numba import cuda
import math


USE_CUDA = True


@cuda.jit
def event_generator_kernel(frame_id, image_log, mem_frame, threshold_pos, threshold_neg, eps, event_counter, event_block):
    i, j = cuda.grid(2)

    idx = 0
    event = cuda.local.array((25, 4), numba.int64) # max 25 events per pixel.
    if i < image_log.shape[0] and j < image_log.shape[1]:    
        while image_log[i, j] - mem_frame[i, j] > threshold_pos-eps:
            if frame_id != 0:
                event[idx, 0] = i
                event[idx, 1] = j
                event[idx, 2] = frame_id
                event[idx, 3] = 1

                idx += 1

            mem_frame[i, j] += threshold_pos

        while image_log[i, j] - mem_frame[i, j] < -threshold_neg+eps:
            if frame_id != 0:
                event[idx, 0] = i
                event[idx, 1] = j
                event[idx, 2] = frame_id
                event[idx, 3] = -1

                idx += 1
        
            mem_frame[i, j] -= threshold_neg

    event_counter[i, j] = idx
    cuda.syncthreads()

    for k in range(0, idx):
        event_block[i, j, k, 0] = event[k, 0]
        event_block[i, j, k, 1] = event[k, 1]
        event_block[i, j, k, 2] = event[k, 2]
        event_block[i, j, k, 3] = event[k, 3]
    

def cuda_event_generator_driver(frame_id, image_log, mem_frame, threshold_pos, threshold_neg, eps):
    threadsperblock = (16, 16)
    blockspergrid_y = math.ceil(image_log.shape[0] / threadsperblock[0])
    blockspergrid_x = math.ceil(image_log.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_y, blockspergrid_x)

    d_image_log = cuda.to_device(image_log)
   
    event_counter = cuda.to_device(np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.int32))
    event_block = cuda.to_device(np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 25, 4), dtype=np.int64))

    event_generator_kernel[blockspergrid, threadsperblock](frame_id, d_image_log, mem_frame, threshold_pos, threshold_neg, eps, event_counter, event_block)
    
    event_counter = event_counter.copy_to_host()
    event_block = event_block.copy_to_host()

    event_indices = np.nonzero(event_counter)

    total_n_events = np.sum(event_counter[event_indices])
    ts = np.zeros(total_n_events, dtype=np.int64)
    xs = np.zeros(total_n_events, dtype=np.int16)
    ys = np.zeros(total_n_events, dtype=np.int16)
    ps = np.zeros(total_n_events, dtype=np.int8)
    
    current_events = 0
    for y, x in zip(event_indices[0], event_indices[1]):
        n_events = event_counter[y, x] 
        
        events = event_block[y, x, :n_events, :]

        ys[current_events: current_events+n_events] = events[:, 0]
        xs[current_events: current_events+n_events] = events[:, 1]
        ts[current_events: current_events+n_events] = events[:, 2]
        ps[current_events: current_events+n_events] = events[:, 3]

        current_events += n_events

    if not total_n_events: return mem_frame, None
    
    event = np.array([ts, xs, ys, ps])

    return mem_frame, event


def cpu_event_generator_driver(frame_id, image_log, mem_frame, threshold_pos, threshold_neg, eps):
    xs, ys, ts, ps = [], [], [], []
    for i in range(image_log.shape[0]):
        for j in range(image_log.shape[1]):
                while image_log[i, j] - mem_frame[i, j] > threshold_pos-eps:
                    if frame_id != 0:
                        xs.append(j)
                        ys.append(i)
                        ts.append(frame_id)
                        ps.append(1)

                    mem_frame[i, j] += threshold_pos

                while image_log[i, j] - mem_frame[i, j] < -threshold_neg+eps:
                    if frame_id != 0:
                        xs.append(j)
                        ys.append(i)
                        ts.append(frame_id)
                        ps.append(-1)

                    mem_frame[i, j] -= threshold_neg

    if not len(ps):
        return mem_frame, None

    event = np.array([np.array(ts, dtype=np.int64), 
                        np.array(xs, dtype=np.int16), 
                        np.array(ys, dtype=np.int16), 
                        np.array(ps, dtype=np.int8)])

    return mem_frame, event




class ColorEventSimulator:
    def __init__(self, contrast_threshold_pos, contrast_threshold_neg, device):
        H, W = OUTPUT_HEIGHT, OUTPUT_WIDTH
        mem_frame = np.zeros((H, W))
        bg_val = 159./255.
        mem_frame += np.log(bg_val**2.2+0.01)

        color_mask = np.zeros((H, W, 3))
        color_mask[0::2, 0::2, 0] = 1  # r

        color_mask[0::2, 1::2, 1] = 1  # g
        color_mask[1::2, 0::2, 1] = 1  # g

        color_mask[1::2, 1::2, 2] = 1  # b

        self.color_mask = color_mask

        if USE_CUDA:
            self.mem_frame = cuda.to_device(mem_frame)
        else:
            self.mem_frame = mem_frame

        self.threshold_pos = contrast_threshold_pos
        self.threshold_neg = contrast_threshold_neg
        self.frame_id = 0
        
    def forward(self, rgb_image):
        start_time = time.time()

        rgb_image = rgb_image.astype(dtype=np.float32) / 255
        image_linear = rgb_image ** 2.2
        image_linear = (image_linear * self.color_mask).sum(-1)

        image_log = np.log(image_linear + 1e-4)

        eps = 1e-6

        if USE_CUDA:
            self.mem_frame, event = cuda_event_generator_driver(self.frame_id, image_log, self.mem_frame, self.threshold_pos, self.threshold_neg, eps)
        else:
            self.mem_frame, event = cpu_event_generator_driver(self.frame_id, image_log, self.mem_frame, self.threshold_pos, self.threshold_neg, eps)

        self.frame_id += 1

        return event

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class ColorESIM:
    def __init__(self, contrast_threshold_pos, contrast_threshold_neg, device):
        self.esim = EventSimulator_torch(contrast_threshold_neg=contrast_threshold_neg,
                                    contrast_threshold_pos=contrast_threshold_pos, refractory_period_ns=0)

        self.frame_index = 0

        H, W = OUTPUT_HEIGHT, OUTPUT_WIDTH
        color_mask = np.zeros((H, W, 3))
        color_mask[0::2, 0::2, 0] = 1  # r

        color_mask[0::2, 1::2, 1] = 1  # g
        color_mask[1::2, 0::2, 1] = 1  # g

        color_mask[1::2, 1::2, 2] = 1  # b

        self.color_mask = color_mask

        self.device = device

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


    def forward(self, rgb_image):
        timestamp_ns = np.array([(self.frame_index / SIMULATOR_FPS) * 1e9, ]).astype(dtype=np.int64)
        self.frame_index += 1


        rgb_image = rgb_image.astype(dtype=np.float32) / 255
        image_linear = rgb_image ** 2.2
        image_linear = (image_linear * self.color_mask).sum(-1)

        image_log = np.log(image_linear + 1e-4).astype(dtype=np.float32)


        frame_log = torch.from_numpy(image_log).to(self.device).unsqueeze(0)
        timestamps_ns = torch.from_numpy(timestamp_ns).to(self.device)

        events = self.esim.forward(frame_log, timestamps_ns)
            
        if events is not None: 
            t = events['t'].cpu().numpy().astype(dtype=np.int64)

            xs = events['x'].cpu().numpy().astype(dtype=np.int16)
            ys = events['y'].cpu().numpy().astype(dtype=np.int16)

            polarities = events['p'].cpu().numpy().astype(dtype=np.int8)

            event = np.array([t, xs, ys, polarities])
            return event
