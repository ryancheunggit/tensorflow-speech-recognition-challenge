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

if __name__ == '__main__':
    main()
