#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

import models
import pandas as pd
import sys
import os
import numpy as np
import keras.backend as K
from util import *
from datetime import datetime
from keras.callbacks import *
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(dataset, model_arch, model_size, batch_size):
    MODEL_NAME = "d-{}_m-{}_s-{}".format(dataset, model_arch, model_size)
    OUT_DIR = "../models/{}".format(MODEL_NAME) + "/"

    try:
        os.mkdir("../models/{}".format(MODEL_NAME))
    except:
        print("model directory already exists, will override existing files")
        pass

    df, test_df, X, y, X_test = prepare_data(dataset = dataset)

    probs = []
    train_meta = np.zeros((df.shape[0], 31))
    input_shape = X[0].shape
    print("========== input shape is : {} ===========".format(input_shape))

    folds = get_folds()
    num_folds = len(folds)
    model_builder = getattr(models, "build_{}_model_size_{}".format(model_arch, model_size))

    for i, (train_indices, valid_indices) in enumerate(folds):
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]

        model = load_model("../models/{}".format(MODEL_NAME) + "/" + 'bag_{}_model.h5'.format(i))
        print("========= generating oof predictions {} =========".format(datetime.now().strftime("%H:%M:%S")))
        train_meta[valid_indices] = model.predict_generator(
            generator = batch_generator(X_valid, y_valid, batch_size = batch_size, task = 'test'),
            steps = np.ceil(X_valid.shape[0] / batch_size).astype(int),
            verbose = 1
        )

        print("========= generating test set predictions {} =========".format(datetime.now().strftime("%H:%M:%S")))
        preds = model.predict_generator(
            generator = batch_generator(X_test, batch_size = batch_size, task = 'test'),
            steps = np.ceil(test_df.shape[0] / batch_size).astype(int),
            verbose = 1
        ) / num_folds

        probs.append(preds)

        del model
        K.clear_session()

    print("dump oof predicted probs")
    pd.DataFrame(train_meta, columns = full_labels).to_csv(OUT_DIR + "train_meta_probs.csv", index = True)

    print("dump test set predicted probs")
    test_df = pd.concat([test_df, pd.DataFrame(probs[0], columns = full_labels)], axis = 1)
    for i in range(1, num_folds):
        test_df[full_labels] += pd.DataFrame(probs[i]).values
    test_df.to_csv(OUT_DIR + "test_meta_probs.csv", index = False)

    test_df['label'] = pd.Series(test_df[full_labels].values.argmax(axis = 1)).map(full_num_label_mapping)
    test_df[["fname", "label"]].to_csv(OUT_DIR + "full_label_predicted.csv", index = False)

    test_df.label = test_df.label.map(lambda x: "unknown" if x not in labels else x)
    test_df[["fname", "label"]].to_csv(OUT_DIR + "label_predicted.csv", index = False)


if __name__ == '__main__':
    """
    usage:
        python inference.py mfcc_10_40_20_img arm_cnn s 1024
    """
    assert len(sys.argv) == 5, "need specify dataset, model architecure, model size and batch size"
    dataset, model_arch, model_size, batch_size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    main(dataset, model_arch, model_size, batch_size)
