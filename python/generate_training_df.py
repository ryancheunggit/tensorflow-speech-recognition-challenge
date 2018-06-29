#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

from util import *
import pandas as pd
import os


def main():
    filepaths = []
    filelabels = []

    for label in labels:
        if label == 'unknown':
            continue
        files = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        filepaths.extend(files)
        filelabels.extend([label for _ in range(len(files))])

    train = pd.DataFrame({'file':filepaths,'label':filelabels})
    train[['file', 'label']].to_csv("../input/train_label_file.csv", index = True)

    filepaths = []
    filelabels = []

    for label in full_labels:
        files = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        filepaths.extend(files)
        filelabels.extend([label for _ in range(len(files))])

    train = pd.DataFrame({'file':filepaths,'label':filelabels})
    train[['file', 'label']].to_csv("../input/full_train_label_file.csv", index = True)

if __name__ == '__main__':
    main()
