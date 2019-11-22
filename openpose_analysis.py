import os
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
from glob import glob
from sklearn import preprocessing
from scipy.optimize import curve_fit
from bayes_opt import BayesianOptimization
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def read_yaml_file(path):
    """
    Read yaml file and return dictionary of data
    """
    skip_lines = 2
    with open(path) as f:
        for i in range(skip_lines):
            _ = f.readline()
        data = yaml.load(f)
        return data


def yaml_to_array(path, n_parts=18):
    """
    Read yaml file and return array of data
    each column is for (x, y, confidence, part#, person#, frame#)
    """
    n_frame = int(os.path.basename(path).split('_')[1])
    data = np.array(read_yaml_file(path)['data'])
    if len(data) > 0:
        n_data = int(len(data) / 3)
        n_people = int(n_data / 18)
        parts = np.tile(np.linspace(0, n_parts - 1, n_parts), n_people).astype(int).reshape(-1, 1)
        person = np.repeat(np.linspace(0, n_people - 1, n_people), n_parts).astype(int).reshape(-1, 1)
        n_frames = np.repeat(np.array(n_frame), n_data).astype(int).reshape(-1, 1)
        data = np.hstack((data.reshape(n_data, 3), parts, person, n_frames))
    else:
        data = None
    return data


def load_yaml_to_dataframe(folder_path):
    """
    Load all yaml files within a folder parsed by OpenPose 
    and return to dataframe
    """
    files = sorted(glob(os.path.join(folder_path, '*.yml')))
    data_array = []
    for f in files:
        data = yaml_to_array(f)
        if data is not None:
            data_array.append(data)
    data_array = np.vstack(data_array)
    df = pd.DataFrame(data_array,
                      columns=['x', 'y', 'confidence', 'part', 'person', 'frame'])
    return df.dropna()


def load_all_data(folder_path, indices=None):
    """
    Load all data from subfolders from a given ``folder_path``
    if ``indices`` (list) is provided, only read folder with name
    """
    df = []
    paths = glob(os.path.join(folder_path, '*')) # all subfolders
    if indices is not None:
        paths = [path for path in paths if os.path.split(path)[-1] in indices]

    for path in tqdm(paths):
        if os.path.isdir(os.path.join(path)):
            vid_id = os.path.split(path)[-1]
            df_id = load_yaml_to_dataframe(path)
            df_id['id'] = vid_id
            df.append(df_id)
    df = pd.concat(df)
    return df


def optimize_param_sin_curve(df, pbounds={'a': (0.1, 1.0), 'T': (100, 200)}):
    """
    Find the best fitted sine curve for the given parsed OpenPose dataframe
    """
    def sin_curve(x, a, T):
        return a * np.sin(2 * np.pi * x / T) + 0.5

    def mean_square_error(a, T):
        x = np.arange(len(x_scaled))
        return - (((sin_curve(x) - x_scaled.ravel()) ** 2).sum() / len(x_scaled)

    x = df.x.values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    optimizer = BayesianOptimization(
        f=mean_square_error,
        pbounds=pbounds,
        random_state=1
    )
    optimizer.maximize(
        init_points=10,
        n_iter=10,
    ) # Bayesian optimization
    params = optimizer.max['params']
    p, pcov = curve_fit(sin_curve, 
                        np.arange(len(x_scaled)),
                        x_scaled.ravel(),
                        p0=[params['a'], params['T']]) # initial point
    return p


def butter_lowpass(cutoff, fs, order=5):
    """
    Create low pass filter coefficient

    Reference: https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs=30, order=5):
    """
    Filter data using Butterworth filter
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def calculate_std_orig_vs_filtered(df):
    """
    For each video_id, person, and body part, calculate standard deviation between orginal 
    data and low-pass filtered data

    Return dictionary of video_id, std, person, and body part
    """
    data_deviation = []
    for (idx, person, part), df_person in tqdm_notebook(df.groupby(['id', 'person', 'part'])):
        if len(df_person) > 100:
            try:
                data = df_person[(df_person.x > 0)].x.values.reshape(-1, 1)
                data_normalized = preprocessing.StandardScaler().fit_transform(data).ravel()
                data_filtered = butter_lowpass_filter(data_normalized,
                                                    cutoff=1, fs=30, order=6)
                std = np.std(data_normalized - data_filtered)
                data_deviation.append({
                    'id': idx,
                    'std_signal': std,
                    'person': person,
                    'part': part
                })
            except:
                pass
    return data_deviation