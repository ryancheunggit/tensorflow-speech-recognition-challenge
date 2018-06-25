#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import models
import sys
import os
from util import *
from datetime import datetime
from keras.callbacks import *
from keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(dataset, model_arch, model_size, batch_size):
    MODEL_NAME = "d-{}_m-{}_s-{}".format(dataset, model_arch, model_size)
    OUT_DIR = "../models/{}".format(MODEL_NAME) + "/"
    os.system("mkdir -p ../models/{}".format(MODEL_NAME))
    df, test_df, X, y, X_test = prepare_data(dataset=dataset)
    nb_epochs = []
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
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

        model = model_builder(input_shape)
        initial_lr = K.eval(model.optimizer.lr)

        if i == 0:
            print("------------- SUMMARY OF MODEL -------------")
            print(model.summary())
            plot_model(model, to_file=OUT_DIR + 'model.png')
            print("--------------------------------------------")
        print("========= fitting {} th model {} =========".format(i + 1, datetime.now().strftime("%H:%M:%S")))
        train_data_generator = batch_generator(X_train, y_train, batch_size=batch_size, task='train')
        validation_data_generator = batch_generator(X_valid, y_valid,  batch_size=batch_size, task='train')
        # if model_arch == '1dcnn':
        #     train_data_generator = batch_generator(X_train, y_train, batch_size=batch_size, task='train',
        #                                            noise_level=0.3, shift_level=0.3, stretch_level=0.3)

        history = model.fit_generator(
            generator = train_data_generator,
            steps_per_epoch = np.ceil(X_train.shape[0] / batch_size).astype(int),
            epochs = 65535,
            validation_data = validation_data_generator,
            validation_steps = np.ceil(X_valid.shape[0] / batch_size).astype(int),
            callbacks = [
                ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, min_lr = max(0.00001, initial_lr / 1000.0), verbose = 1),
                # TensorBoard(log_dir="logs/{}_model_{}".format(MODEL_NAME, i), histogram_freq=0, write_graph=True, write_images=True),
                # CMCallback(X_valid, y_valid, fullset = True, batch_szie = batch_size),
                EarlyStopping(monitor = 'val_loss', patience = 15, verbose = 1),
                ModelCheckpoint(filepath='{}model_checkpoint_fold_{}.hdf5'.format(OUT_DIR, i), monitor='val_loss',
                                save_best_only=True, save_weights_only=True, mode='min')
                ],
            verbose = 1
        )

        nb_epochs.append(len(history.epoch))
        train_acc.append(history.history['categorical_accuracy'][-1])
        train_loss.append(history.history['loss'][-1])
        val_acc.append(history.history['val_categorical_accuracy'][-1])
        val_loss.append(history.history['val_loss'][-1])

        print("========= generating oof predictions {} =========".format(datetime.now().strftime("%H:%M:%S")))
        train_meta[valid_indices] = model.predict_generator(
            generator = batch_generator(X_valid, y_valid, batch_size = batch_size, task = 'test'),
            steps = np.ceil(X_valid.shape[0] / batch_size).astype(int),
            verbose = 0
        )

        print("========= generating test set predictions {} =========".format(datetime.now().strftime("%H:%M:%S")))
        preds = model.predict_generator(
            generator = batch_generator(X_test, batch_size = batch_size, task = 'test'),
            steps = np.ceil(test_df.shape[0] / batch_size).astype(int),
            verbose = 0
        ) / num_folds

        pd.DataFrame(preds, columns = full_labels).to_csv(OUT_DIR + "bag_{}_test_pred_probs.csv".format(i), index = False)
        probs.append(preds)

        model.save(OUT_DIR + 'bag_{}_model.h5'.format(i))

        del model
        K.clear_session()

    print("train loss avg {} -- std {}, val loss avg {} -- std {}".format(
            np.mean(train_loss), np.std(train_loss),
            np.mean(val_loss), np.std(val_loss)
        ))

    print("train acc avg {} -- std {}, val acc avg {} -- std {}".format(
            np.mean(train_acc), np.std(train_acc),
            np.mean(val_acc), np.std(val_acc)
        ))

    print("mean nb epochs {}".format(np.mean(nb_epochs)))

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
        python train.py mfcc_10_40_20_img arm_cnn s 1024
    """
    assert len(sys.argv) == 5, "need specify dataset, model architecure, model size and batch size"
    dataset, model_arch, model_size, batch_size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    main(dataset, model_arch, model_size, batch_size)
