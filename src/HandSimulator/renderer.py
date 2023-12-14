import os; os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys; sys.path.append('..')
import pyrender 
from functools import partial
import torch
import cv2
import numpy as np

from color_event_simulator import ColorEventSimulator, ColorESIM
from settings import *


from utils import RGBDRenderer, SegmentationRenderer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Renderer:
    def create_esim(self):
        self.frame_index = 0

        esim = ColorESIM(contrast_threshold_neg=ESIM_POSITIVE_THRESHOLD,
                            contrast_threshold_pos=ESIM_NEGATIVE_THRESHOLD,
                            device=self.device)

        return esim

    def generate_events(self, image):
        return self.esim.forward(image)


    def __init__(self, device, two_hands, camera_params, fps, **kwargs) -> None:
        self.fps = fps
        
        self.device = device

        self.renderer = pyrender.OffscreenRenderer(viewport_width=OUTPUT_WIDTH, viewport_height=OUTPUT_HEIGHT)

        self.segmentation_renderer = SegmentationRenderer(cam_params=camera_params, renderer=self.renderer, **kwargs) 
        self.RGBD_renderer = RGBDRenderer(cam_params=camera_params, renderer=self.renderer, **kwargs)

        self.two_hands = two_hands

        self.esim = self.create_esim()

    def __call__(self, hand_info, **kwargs):
        meshes = self.two_hands.generate_mesh(hand_info, texture_type='segmentation')
        segmentation_maps = self.segmentation_renderer(meshes, **kwargs)
        
        meshes = self.two_hands.generate_mesh(hand_info, texture_type='uv')        
        images, depth_maps = self.RGBD_renderer(meshes, **kwargs)
        
        image = images[0]
        segmentation_map = segmentation_maps[0]
        depth_map = depth_maps[0]
        
        event = self.generate_events(image)
        

        output = {
                    'event': event,
                    'image': image, 
                    'depth': depth_map,
                    'segmentation': segmentation_map
                }

        if event is None:
            return None

        return output

    def close(self):
        self.renderer.delete()
