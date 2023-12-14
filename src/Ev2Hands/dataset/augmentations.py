import sys; sys.path.append('../..') # add settings.
import numpy as np
from settings import OUTPUT_WIDTH, OUTPUT_HEIGHT


def augment_events_old(events, event_labels):
    augment = np.random.rand(events.shape[0]) < 0.5
    events[augment][:, -1] = np.abs(1 - events[augment][:, -1])


    num_noisy_events = events.shape[0] // 32
    
    xs = np.random.randint(0, OUTPUT_WIDTH, size=num_noisy_events)
    ys = np.random.randint(0, OUTPUT_HEIGHT, size=num_noisy_events)
    # ts = events[-1, 2] + np.linspace(0.1, 1, num_noisy_events) * 1e3
    # ts = events[-1, 2] + np.random.rand(num_noisy_events) * 1e3
    ts = events[np.random.randint(events.shape[0] - 1, size=num_noisy_events), 2] + np.random.rand(num_noisy_events) * 1e3

    ps = np.random.rand(num_noisy_events) < 0.5
    
    noise_events = np.stack([xs, ys, ts, ps], 1)
    noise_event_labels = np.ones(num_noisy_events) * 3
    
    try:
        events = np.concatenate([events, noise_events])
        event_labels = np.concatenate([event_labels, noise_event_labels])
    except:
        pass
        
    return events, event_labels


def augment_events(events, event_labels):
    n_events = events.shape[0]
    augment = np.random.rand(n_events) < 0.5
    n_aug_events = np.count_nonzero(augment)

    num_noisy_events = n_events // 32
    
    xs = np.random.randint(0, OUTPUT_WIDTH, size=num_noisy_events)
    ys = np.random.randint(0, OUTPUT_HEIGHT, size=num_noisy_events)
    ts = events[np.random.randint(n_events - 1, size=num_noisy_events), 2] + np.random.rand(num_noisy_events) * 1e3
    
    if events.shape[1] == 5:
        ps = np.random.rand(1) < 0.5
        events[augment][:, -1] = np.random.randint(ps, 8, size=n_aug_events)
        events[augment][:, -2] = np.random.randint(np.invert(ps), 8, size=n_aug_events)

        ps = np.random.rand(num_noisy_events) < 0.5

        n_pe = np.random.randint(0, 8, size=num_noisy_events) + ps
        n_ne = np.random.randint(0, 8, size=num_noisy_events) + np.invert(ps)
        
        noise_events = np.stack([xs, ys, ts, n_pe, n_ne], 1)
    
    elif events.shape[1] == 4:
        events[augment][:, -1] = np.abs(1 - events[augment][:, -1])

        ps = np.random.rand(num_noisy_events) < 0.5
    
        noise_events = np.stack([xs, ys, ts, ps], 1)
    else:
        raise Exception("No augmentation Available")

    noise_event_labels = np.ones(num_noisy_events) * 3
    
    try:
        events = np.concatenate([events, noise_events])
        event_labels = np.concatenate([event_labels, noise_event_labels])
    except:
        pass
        
    return events, event_labels

