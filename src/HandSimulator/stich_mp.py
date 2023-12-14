import os
import sys; sys.path.append('..') # add settings.
from settings import GENERATION_MODE, ROOT_TRAIN_DATA_PATH
import glob
import pickle
from tqdm import tqdm
from natsort import natsorted
import numpy as np
import h5py


def worker_fn(file_path, dataset_index):
    if not os.path.exists(file_path): return

    path = f'{ROOT_TRAIN_DATA_PATH}/{GENERATION_MODE}.h5'

    with open(file_path, 'rb') as pickle_file:
        cache = pickle.load(pickle_file)            

    last_key = list(cache.keys())[-1].decode()
    last_index = int(last_key[-9:])
    
    camera_hand_infos = []
    for _ in range(dataset_index, last_index + 1):        
        eventKey = 'events-%09d'.encode() % dataset_index
        event_label_key = 'event_labels-%09d'.encode() % dataset_index
        camera_hand_infoKey = 'camera_hand_info-%09d'.encode() % dataset_index

        if eventKey not in cache: continue
        if event_label_key not in cache: continue
        if camera_hand_infoKey not in cache: continue

        events = cache[eventKey]
        event_labels = cache[event_label_key]
        camera_hand_info = cache[camera_hand_infoKey]

        camera_hand_infos.append(camera_hand_info)

        x, y, t, p = events.T
        annotation_index = np.ones_like(x) * dataset_index

        events = np.hstack([x[..., None], y[..., None], t[..., None], p[..., None], annotation_index[..., None], event_labels[..., None]])

        if not os.path.exists(path):            
            with h5py.File(path, 'w') as f:
                f.create_dataset('event', data=events, chunks=True, maxshape=(None, events.shape[1]))
        else:
            with h5py.File(path, 'a') as f:
                dataset = f['event']


                # Determine the shape of the existing dataset
                existing_shape = dataset.shape

                # Determine the shape of the new data
                new_shape = events.shape

                # Adjust the shape of the existing dataset to accommodate the new data
                dataset.resize(existing_shape[0] + new_shape[0], axis=0)

                # Append the new data to the existing dataset
                dataset[existing_shape[0]:] = events

        dataset_index += 1

    return dataset_index, camera_hand_infos


def main():
    os.system(f'rm {ROOT_TRAIN_DATA_PATH}/{GENERATION_MODE}.h5')

    save_path = os.path.join(f'{ROOT_TRAIN_DATA_PATH}/{GENERATION_MODE}/parts') 
    assert GENERATION_MODE in ['train', 'test', 'val']

    total_n_files = len(glob.glob(f'{save_path}/*.pickle'))
    
    pbar = tqdm(total=total_n_files) 

    annotation_path = f'{ROOT_TRAIN_DATA_PATH}/{GENERATION_MODE}_anno.pickle'
        
    annotations = []
    dataset_index = 0

    n_files = 0
    job_id = 0
    while True:
        file_paths = glob.glob(f'{save_path}/{job_id}_*.pickle')
        job_id += 1
        
        n_files += len(file_paths)

        for file_path in natsorted(file_paths):   
            dataset_index, camera_hand_infos = worker_fn(file_path, dataset_index)   
            annotations.extend(camera_hand_infos)

            pbar.update(1)

        if not (n_files < total_n_files): break

    pbar.close()

    with open(annotation_path, 'wb') as pickle_file:
        pickle.dump(annotations, pickle_file)


if __name__ == '__main__':
    main()
