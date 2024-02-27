from time import time
import numpy as np

class Timer():
    '''
    A simple timing helper class that measures the duration between the last time "get_split" was
    called and the present time.
    Note that the first split is created when this object is initialized
    '''
    def __init__(self):
        self.last_split = time()

    def get_split(self):
        t = time()
        split_ = t - self.last_split
        self.last_split = t
        return split_


def get_squared_euclidean_distance_matrix_np(m1, m2):
    '''
    Using trick mentioned in: https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
    Note that this returns the squared distance matrix (euclidean distance obtained simply by sqrt of these values)
    params::
    m1: Matrix 1 (N x M)
    m2: Matrix 2 (K x M)
    returns::
    (N X K) Matrix of pairwise distances between m1 and m2
    '''
    return np.sum(m1 ** 2, axis=1)[:, np.newaxis] - (2. * np.matmul(m1, m2.T)) + np.sum(m2 ** 2, axis=1)[np.newaxis]


def uniform_resample(data, current_freq, target_freq):
    # Helper for uniform resampling (upsample/downsample)

    sampling_ratio = target_freq / current_freq
    data_ss_ixs = np.minimum(np.round(np.arange(0, data.shape[0], 1 / sampling_ratio)).astype(int), data.shape[0] - 1)
    resampled_data = data[data_ss_ixs]
    return resampled_data


def rescale_range(values, in_min, in_max, out_min, out_max, return_inverse_transform_parameters=False):
    '''
    Helper for rescaling values from (in_min,in_max) to (out_min,out_max) linearly
    '''
    inverse_transform_parameters = {"sub_1": out_min,
                                    "div_1": out_max - out_min,
                                    "add_1": in_min,
                                    "mult_1": in_max - in_min}

    values = np.clip(values, a_min=in_min, a_max=in_max)
    values = (values - in_min) / (in_max - in_min)
    values = (values * (out_max - out_min)) + out_min

    if return_inverse_transform_parameters:
        return values, inverse_transform_parameters
    else:
        return values


def split_clip_into_segments(clip, chunk_time=10, min_clip_time=0, fs=22050):
    dur = len(clip) / fs
    chunks = np.round(dur / chunk_time).astype(int)
    chunk_samples = chunk_time * fs
    clips = []
    for n in range(chunks):
        n_clip = clip[n * chunk_samples:(n + 1) * chunk_samples]
        if len(n_clip)/fs >= min_clip_time:
            clips.append(n_clip)

    return clips
