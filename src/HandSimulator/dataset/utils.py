import collections
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import smplx
import torch


def generate_interpolators(hands_dict, fps_input, fps_output):
    inter_X = dict()
    max_input_sequence_length = 0
    for hand_type, hand_params in hands_dict.items():
        if hand_params is None: continue

        length_in = hands_dict[hand_type]['frame_count']
        x_in = np.linspace(0, length_in, num=length_in, endpoint=False)    
        max_input_sequence_length = max(max_input_sequence_length, length_in)
        
        inter_X[hand_type] = {'x_in': x_in}

    output_length = np.round((max_input_sequence_length / fps_input) * fps_output).astype(dtype=np.int32)
    inter_X['output_length'] = output_length

    for hand_type, hand_params in hands_dict.items():
        if hand_params is None: continue

        x_in = inter_X[hand_type]['x_in']
        x_out = np.linspace(0, x_in[-1], num=output_length, endpoint=True)

        inter_X[hand_type]['x_out'] = x_out

    return inter_X



# interpolate to desired frame rate
def interpolate_sequence(seq_dict, fps_input, fps_output):
    hands_dict = {
        'left': None,
        'right': None
    }
    
    sorted_frame_ids = sorted(list(seq_dict.keys()), key=lambda v: int(v))
    
    for idx, frame_idx in enumerate(sorted_frame_ids):
        hands = seq_dict[frame_idx]

        for hand_type, hand in hands.items():
            if hands_dict[hand_type] is None:
                hands_dict[hand_type] = {'pose': [], 'shape': [], 'trans': [], 'frame_count': 0}

            if hand is None: continue

            pose = np.array(hand['pose'], dtype=np.float32)[None, ...]
            shape = np.array(hand['shape'], dtype=np.float32)[None, ...]
            trans = np.array(hand['trans'], dtype=np.float32)[None, ...]
            
            hands_dict[hand_type]['pose'].append(pose)
            hands_dict[hand_type]['shape'].append(shape)
            hands_dict[hand_type]['trans'].append(trans)

            hands_dict[hand_type]['frame_count'] += 1
 
    inter_X = generate_interpolators(hands_dict, fps_input, fps_output)
    output_length = inter_X['output_length']

    interpolated_frames = collections.defaultdict(list)
    for hand_type, hand_params in hands_dict.items():
        if hand_params is None: continue

        x_in = inter_X[hand_type]['x_in']
        x_out = inter_X[hand_type]['x_out']

        pose = np.concatenate(hand_params['pose'], 0)
        shape = np.concatenate(hand_params['shape'], 0)
        trans = np.concatenate(hand_params['trans'], 0)
        
        inter_pose = list()
        for i in range(0, pose.shape[1], 3):                
            y = R.from_rotvec(pose[:, i: i + 3])
            slerp = Slerp(x_in, y)
            inter_pose.append(slerp(x_out).as_rotvec())

        inter_pose = np.concatenate(inter_pose, axis=1)

        inter_shape = list()
        for i in range(0, shape.shape[1]):       
            y_right = shape[:, i]
            interp = interp1d(x_in, y_right, kind='cubic')            
            inter_shape.append(interp(x_out)[..., None])

        inter_shape = np.concatenate(inter_shape, axis=1)

        inter_trans = list()
        for i in range(0, trans.shape[1]):       
            y_right = trans[:, i]
            interp = interp1d(x_in, y_right, kind='cubic')            
            inter_trans.append(interp(x_out)[..., None])

        inter_trans = np.concatenate(inter_trans, axis=1)
                    
        for i in range(output_length): 
            interpolated_frames[i].append({
                'hand_type': hand_type,

                'pose': inter_pose[i], 
                'shape': inter_shape[i], 
                'trans': inter_trans[i]
            })

    return interpolated_frames


# interpolate to desired frame rate
def mano_data_to_mano_sequence(seq_dict):
    mano_sequence = collections.defaultdict(list)

    sorted_frame_ids = sorted(list(seq_dict.keys()), key=lambda v: int(v))
    
    for i, frame_idx in enumerate(sorted_frame_ids):
        hands = seq_dict[frame_idx]

        for hand_type, hand in hands.items():
            if hand is None: continue

            pose = np.array(hand['pose'], dtype=np.float32)
            shape = np.array(hand['shape'], dtype=np.float32)
            trans = np.array(hand['trans'], dtype=np.float32)

            mano_sequence[i].append({
                    'hand_type': hand_type,

                    'pose': pose, 
                    'shape': shape, 
                    'trans': trans
                })

    return mano_sequence
