#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

from __future__ import division
import librosa
import numpy as np
from scipy.io import wavfile
from scipy import signal
import pandas as pd
import os
import pickle
import h5py
from sklearn.utils import shuffle
from sklearn.metrics import log_loss, accuracy_score
from keras.callbacks import Callback

SEED = 123
NUM_FOLDS = 5

train_audio_path = "../input/train/audio"
test_data_path = "../input/test/audio"

labels = ["up", "down", "go", "stop", "left", "right", "on", "off","yes", "no", "silence", "unknown"]
full_labels = [
    "up", "down", "go", "stop", "left", "right", "on", "off", "yes", "no", "silence",
    "bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "marvin",
    "nine", "one", "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"
    ]

label_num_mapping = {labels[i]: i for i in range(len(labels))}
num_label_mapping = {i: labels[i] for i in range(len(labels))}

full_label_num_mapping = {full_labels[i]: i for i in range(len(full_labels))}
full_num_label_mapping = {i: full_labels[i] for i in range(len(full_labels))}

def read_wav_sample(path):
    ''' read in wav file as sample data, pad / truncate if needed'''
    sample_rate, samples = wavfile.read(path)
    if len(samples) > sample_rate:
        samples = samples[:sample_rate]
    else:
        samples = np.pad(samples, (0, max(0, sample_rate - len(samples))), 'constant')
    return samples

def read_logspectrogram(filepath, nperseg = 400, noverlap = 100):
    sample_rate, samples =  wavfile.read(filepath)
    if len(samples) > sample_rate:
        samples = samples[:sample_rate]
    else:
        samples = np.pad(samples, (0, max(0, sample_rate - len(samples))), 'constant')
    frequencies, times, spectrogram = signal.stft(x = samples, fs = sample_rate, nperseg = nperseg, noverlap = noverlap)
    return np.log(np.abs(spectrogram).T + 1e-10)

def read_mfcc(filepath, num_components = 10, n_fft = 960, hop_length = 320):
    samples, sample_rate = librosa.load(filepath, sr = 16000)
    if len(samples) > sample_rate:
        samples = samples[:sample_rate]
    else:
        samples = np.pad(samples, (0, max(0, sample_rate - len(samples))), 'constant')
    mfcc = librosa.feature.mfcc(y = samples, sr = 16000, n_fft = n_fft, hop_length = hop_length, n_mfcc = num_components)
    return mfcc.T

def read_logmelspectrogram(filepath, num_components = 40, n_fft = 400, hop_length = 160):
    samples, sample_rate = librosa.load(filepath, sr = 16000)
    if len(samples) > sample_rate:
        samples = samples[:sample_rate]
    else:
        samples = np.pad(samples, (0, max(0, sample_rate - len(samples))), 'constant')
    melspectrogram = librosa.feature.melspectrogram(y = samples, sr = 16000, n_fft = n_fft, hop_length = hop_length, n_mels = num_components)
    return librosa.power_to_db(melspectrogram).T

def get_training_label_wavfilename(fullset = False):
    if not fullset:
        df = pd.read_csv('../input/train_label_file.csv')
    else:
        df = pd.read_csv('../input/full_train_label_file.csv')
    df.columns = ['idx', 'filename', 'label']
    return df

def get_test_wavefilename():
    df = pd.read_csv("../input/sample_submission.csv")
    return df

def get_folds():
    return pickle.load(open("../input/folds.pkl", "rb"))

def batch_generator(X, y = None, batch_size = 128, task = 'train',
    noise_level = None, shift_level = None, stretch_level = None
    ):
    assert task in ['train', 'test'], "invalid task mode"
    num_batches = np.ceil(X.shape[0] / batch_size).astype(int)
    current_batch = 0
    while True:
        lower_bound = current_batch * batch_size
        upper_bound = lower_bound + batch_size
        if current_batch == num_batches -1:
            upper_bound = X.shape[0]
        current_batch += 1
        current_batch_size = upper_bound - lower_bound

        X_batch = X[lower_bound:upper_bound, :].copy()
        if task == 'train' and y is not None:
            y_batch = y[lower_bound:upper_bound, :]
            if noise_level:
                X_batch += (noise_level * np.random.random() * X_batch.std(axis = 1) * np.random.randn(current_batch_size, 16000)).reshape(current_batch_size, 16000, 1).astype(int)
            if shift_level:
                X_batch = np.roll(X_batch, int(shift_level * (np.random.random() -0.5 ) / 5 * 16000), 1)
            if stretch_level:
                for i in range(X_batch.shape[0]):
                    modified_samples = librosa.effects.time_stretch(X_batch[i, :].flatten().astype('float64'), 2 ** (stretch_level / 5 * (np.random.random() -0.5 )))
                    if len(modified_samples) > 16000:
                        modified_samples = modified_samples[int((modified_samples.shape[0] - 16000) /2): - int((modified_samples.shape[0] - 16000) /2)]
                    else:
                        modified_samples = np.pad(modified_samples, (0, max(0, 16000 - len(modified_samples))), 'constant')
                    X_batch[i, :] = modified_samples.reshape(16000, 1)
            yield (X_batch, y_batch)
        elif task == 'test':
            yield X_batch
        if current_batch == num_batches:
            current_batch = 0

def prepare_data(dataset, dump = True):
    df = get_training_label_wavfilename(fullset = True)
    np.random.seed(SEED)
    df = shuffle(df)
    test_df = get_test_wavefilename()
    y = np.zeros((df.shape[0], 31))
    for idx in range(df.shape[0]):
        y[idx, full_label_num_mapping[df.label.values[idx]]] = 1

    X, X_test = np.zeros(df.shape[0]), np.zeros(test_df.shape[0])
    if os.path.exists("../input/{}.h5".format(dataset)):
        print("======= loading data =======")
        with h5py.File("../input/{}.h5".format(dataset), "r") as hf:
            X = hf["."]["X"].value
            X_test = hf["."]["X_test"].value
    else:
        print("======= processing data =======")

        if dataset.startswith('mfcc'):
            # like mfcc_40_25_10 => 101 x 40
            num_components, window_size, window_stride = [float(i) for i in dataset.split("_")[1:]]
            num_components = int(num_components)
            n_fft = int(16000 * window_size / 1000)
            hop_length = int(16000 * window_stride / 1000)
            X = np.array([
                read_mfcc(file_path, num_components = num_components, n_fft = n_fft, hop_length = hop_length)
                for file_path in df.apply(lambda row: train_audio_path + '/' + row['label'] + '/' + row['filename'], axis = 1)
                    ])
            X_test = np.array([
                read_mfcc(file_path, num_components = num_components, n_fft = n_fft, hop_length = hop_length)
                for file_path in test_df.apply(lambda row: test_data_path + '/' + row['fname'], axis = 1)
                    ])

        if dataset.startswith('logspectrogram'):
            # like logspectrogram_25_18.75 => 55 x 201
            window_size, window_stride = [float(i) for i in dataset.split("_")[1:]]
            nperseg = int(16000 * window_size / 1000)
            noverlap = int(16000 * (window_size - window_stride) / 1000)
            X = np.array([
                read_logspectrogram(file_path, nperseg = nperseg, noverlap = noverlap)
                for file_path in df.apply(lambda row: train_audio_path + '/' + row['label'] + '/' + row['filename'],axis = 1)
                    ])
            X_test = np.array([
                read_logspectrogram(file_path, nperseg = nperseg, noverlap = noverlap)
                for file_path in test_df.apply(lambda row: test_data_path + '/' + row['fname'], axis = 1)
                    ])

        if dataset.startswith("logmelsepctrogram"):
            # like logmelsepctrogram_40_25_10 => 101 x 40
            num_components, window_size, window_stride = [float(i) for i in dataset.split("_")[1:]]
            num_components = int(num_components)
            n_fft = int(16000 * window_size / 1000)
            hop_length = int(16000 * window_stride / 1000)
            X = np.array([
                read_logmelspectrogram(file_path, n_fft = 400, hop_length = 160)
                for file_path in df.apply(lambda row: train_audio_path + '/' + row['label'] + '/' + row['filename'],axis = 1)
                    ])
            X_test = np.array([
                read_logmelspectrogram(file_path, n_fft = 400, hop_length = 160)
                for file_path in test_df.apply(lambda row: test_data_path + '/' + row['fname'], axis = 1)
                    ])

        if dataset == 'rawwav':
            # shape 16000,1
            X = np.array([
                read_wav_sample(file_path)
                for file_path in df.apply(lambda row: train_audio_path + '/' + row['label'] + '/' + row['filename'], axis = 1)
                    ]).reshape(df.shape[0], 16000, 1)

            X_test = np.array([
                read_wav_sample(file_path)
                for file_path in test_df.apply(lambda row: test_data_path + '/' + row['fname'], axis = 1)
                    ]).reshape(test_df.shape[0], 16000, 1)

        if not os.path.exists("../input/{}.h5".format(dataset)):
            if dump:
                with h5py.File("../input/{}.h5".format(dataset), "w") as hf:
                    dset_train = hf.create_dataset("X",  data = X)
                    dset_test = hf.create_dataset("X_test",  data = X_test)

    return df, test_df, X, y, X_test

class ConfusionMatrixCallback(Callback):
    def __init__(self, X, y, fullset = False, cm_per_epochs = 5, batch_size = 128):
        self.X = X
        self.y = y
        self.fullset = fullset
        self.cm_per_epochs = cm_per_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.cm_per_epochs == 0:
            try:
                y_pred = self.model.predict_generator(
                    generator = batch_generator(self.X, self.y, batch_size = batch_size, task = 'test'),
                    steps = np.ceil(self.X.shape[0] / batch_size).astype(int),
                    verbose = 0
                    )
                print(" - val loss: {} - val acc : {}".format(
                    np.round(log_loss(self.y.argmax(axis = 1), y_pred), 4),
                    np.round(accuracy_score(self.y.argmax(axis =1), y_pred.argmax(axis = 1)), 4)
                    ))
                print("\n")
                print("Confusion Matrix:")
                if self.fullset:
                    print(classification_report(self.y.argmax(axis =1), y_pred.argmax(axis = 1), target_names  = full_labels))
                else:
                    print(classification_report(self.y.argmax(axis =1), y_pred.argmax(axis = 1), target_names  = labels))
            except:
                print("may have problem calculating confusion matrix")
