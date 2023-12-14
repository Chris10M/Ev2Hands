import os
if os.name != 'nt': os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender
import numpy as np
import os
import platform


ESIM_REFRACTORY_PERIOD_NS = 0
ESIM_POSITIVE_THRESHOLD = 0.4
ESIM_NEGATIVE_THRESHOLD = 0.4

RENDER_SMPLX = False

AUGMENTATED_SEQUENCE = True
NUMBER_OF_AUGMENTATED_SEQUENCES = 10

SIMULATOR_FPS = 1000                         # fps for event generation using ESIM
INTERPOLATION_FPS = 30
OUTPUT_WIDTH = 346
OUTPUT_HEIGHT = 260
LNES_WINDOW_MS = 5


INTERHAND_ROOT_PATH = '/CT/datasets01/static00/InterHand2.6m/InterHand2.6M_5fps_batch1'
ROOT_TRAIN_DATA_PATH = '/CT/datasets07/nobackup/Ev2Hands/Ev2Hands-S'

REAL_TRAIN_DATA_PATH = '/CT/datasets07/nobackup/Ev2Hands/Ev2Hands-R/train_data'
REAL_TEST_DATA_PATH = '/CT/datasets07/nobackup/Ev2Hands/Ev2Hands-R/test_data'

DATA_PATH = '../data'
MANO_PATH = '../data/models'


GENERATION_MODE = str(os.getenv('GENERATION_MODE', 'train')) 

MANO_CMPS = 6

SEGMENTAION_COLOR = {'left': [0, 1, 0], 'right': [0, 0, 1]}

MAIN_CAMERA = pyrender.PerspectiveCamera(yfov=np.deg2rad(30), aspectRatio=OUTPUT_WIDTH / OUTPUT_HEIGHT)
PROJECTION_MATRIX = MAIN_CAMERA.get_projection_matrix(OUTPUT_WIDTH, OUTPUT_HEIGHT)

HAND_COLOR = [198/255, 134/255, 66/255]
