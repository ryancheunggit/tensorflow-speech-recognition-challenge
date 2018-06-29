#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

import os
import numpy as np
from scipy.io import wavfile

def main():
    np.random.seed(0)

    if not os.path.exists('../input/train/audio/silence'):
        os.mkdir('../input/train/audio/silence')
    os.system('rm -rf ../input/train/audio/silence/*')

    path = '../input/train/audio/_background_noise_/'
    out_path = '../input/train/audio/silence/'

    files = [fp for fp in os.listdir(path) if fp.endswith('.wav')]

    for filename in files:
        print(filename)
        sample_rate, samples = wavfile.read(path + filename)
        print(len(samples))
        for i in range(400):
            out_name = "segment_{}_".format(i) + filename
            data = (samples[i * 200: i * 200 + sample_rate] * max(0, 2 * (np.random.random() - 0.25))).astype('int16')
            if data.max() == 0:
                print(out_name)
            wavfile.write(out_path + out_name, sample_rate, data)

    for i in range(500):
        d = np.zeros(16000)
        loc = np.random.randint(0, 16000, 4600)
        d[loc[::2]] = -1
        d[loc[1::2]] = 1
        wavfile.write(out_path + 'new_synthesised_{}.wav'.format(i), 16000, d)

if __name__ == '__main__':
    main()
