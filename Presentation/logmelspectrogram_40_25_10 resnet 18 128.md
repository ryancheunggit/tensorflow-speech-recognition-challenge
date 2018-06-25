ren (master *+) python $ python train.py logmelspectrogram_40_25_10 resnet 18 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= processing data =======
========== input shape is : (101, 40) ===========
------------- SUMMARY OF MODEL -------------
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 101, 40)      0
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 101, 40, 1)   0           input_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 46, 19, 64)   2624        reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 46, 19, 64)   256         conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 46, 19, 64)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 46, 19, 64)   0           activation_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 23, 9, 64)    0           dropout_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 23, 9, 64)    32832       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 23, 9, 64)    256         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 23, 9, 64)    0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 23, 9, 64)    0           activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 23, 9, 64)    32832       dropout_2[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 23, 9, 128)   0           max_pooling2d_1[0][0]
                                                                 conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 23, 9, 128)   512         concatenate_1[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 23, 9, 128)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 23, 9, 128)   0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 23, 9, 64)    65600       dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 23, 9, 64)    256         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 23, 9, 64)    0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 23, 9, 64)    0           activation_4[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 23, 9, 64)    8256        concatenate_1[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 23, 9, 64)    32832       dropout_4[0][0]
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 23, 9, 128)   0           conv2d_6[0][0]
                                                                 conv2d_5[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 23, 9, 128)   512         concatenate_2[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 23, 9, 128)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 23, 9, 128)   0           activation_5[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 12, 5, 128)   131200      dropout_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 12, 5, 128)   512         conv2d_7[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 12, 5, 128)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 12, 5, 128)   0           activation_6[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 6, 3, 128)    16512       concatenate_2[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 6, 3, 128)    131200      dropout_6[0][0]
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 6, 3, 256)    0           conv2d_9[0][0]
                                                                 conv2d_8[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 6, 3, 256)    1024        concatenate_3[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 6, 3, 256)    0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 6, 3, 256)    0           activation_7[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 6, 3, 128)    262272      dropout_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 6, 3, 128)    512         conv2d_10[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 6, 3, 128)    0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 6, 3, 128)    0           activation_8[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 6, 3, 128)    32896       concatenate_3[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 6, 3, 128)    131200      dropout_8[0][0]
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 6, 3, 256)    0           conv2d_12[0][0]
                                                                 conv2d_11[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 6, 3, 256)    1024        concatenate_4[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 6, 3, 256)    0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 6, 3, 256)    0           activation_9[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 3, 2, 256)    524544      dropout_9[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 3, 2, 256)    1024        conv2d_13[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 3, 2, 256)    0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, 3, 2, 256)    0           activation_10[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 2, 1, 256)    65792       concatenate_4[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 2, 1, 256)    524544      dropout_10[0][0]
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 2, 1, 512)    0           conv2d_15[0][0]
                                                                 conv2d_14[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 2, 1, 512)    2048        concatenate_5[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 2, 1, 512)    0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
dropout_11 (Dropout)            (None, 2, 1, 512)    0           activation_11[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 2, 1, 256)    1048832     dropout_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 2, 1, 256)    1024        conv2d_16[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 2, 1, 256)    0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
dropout_12 (Dropout)            (None, 2, 1, 256)    0           activation_12[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 2, 1, 256)    131328      concatenate_5[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 2, 1, 256)    524544      dropout_12[0][0]
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 2, 1, 512)    0           conv2d_18[0][0]
                                                                 conv2d_17[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 2, 1, 512)    2048        concatenate_6[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 2, 1, 512)    0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
dropout_13 (Dropout)            (None, 2, 1, 512)    0           activation_13[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 1, 1, 512)    2097664     dropout_13[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 1, 1, 512)    2048        conv2d_19[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 1, 1, 512)    0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
dropout_14 (Dropout)            (None, 1, 1, 512)    0           activation_14[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 1, 1, 512)    262656      concatenate_6[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 1, 1, 512)    2097664     dropout_14[0][0]
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 1, 1, 1024)   0           conv2d_21[0][0]
                                                                 conv2d_20[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 1, 1, 1024)   4096        concatenate_7[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 1, 1, 1024)   0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
dropout_15 (Dropout)            (None, 1, 1, 1024)   0           activation_15[0][0]
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 1, 1, 512)    4194816     dropout_15[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 1, 1, 512)    2048        conv2d_22[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 1, 1, 512)    0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
dropout_16 (Dropout)            (None, 1, 1, 512)    0           activation_16[0][0]
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 1, 1, 512)    524800      concatenate_7[0][0]
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 1, 1, 512)    2097664     dropout_16[0][0]
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 1, 1, 1024)   0           conv2d_24[0][0]
                                                                 conv2d_23[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 1, 1, 1024)   4096        concatenate_8[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 1, 1, 1024)   0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 1, 1, 1024)   0           activation_17[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 1024)         0           average_pooling2d_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 31)           31775       flatten_1[0][0]
==================================================================================================
Total params: 15,030,175
Trainable params: 15,018,527
Non-trainable params: 11,648
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 16:50:14 =========
Epoch 1/65535
420/420 [==============================] - 28s 68ms/step - loss: 0.9720 - categorical_accuracy: 0.2389 - val_loss: 0.9017 - val_categorical_accuracy: 0.4979
Epoch 2/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.8499 - categorical_accuracy: 0.6148 - val_loss: 0.8070 - val_categorical_accuracy: 0.6766
Epoch 3/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.7659 - categorical_accuracy: 0.7562 - val_loss: 0.7332 - val_categorical_accuracy: 0.7773
Epoch 4/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.6990 - categorical_accuracy: 0.8164 - val_loss: 0.6665 - val_categorical_accuracy: 0.8510
Epoch 5/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.6406 - categorical_accuracy: 0.8492 - val_loss: 0.6114 - val_categorical_accuracy: 0.8738
Epoch 6/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.5884 - categorical_accuracy: 0.8693 - val_loss: 0.5647 - val_categorical_accuracy: 0.8735
Epoch 7/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.5408 - categorical_accuracy: 0.8850 - val_loss: 0.5174 - val_categorical_accuracy: 0.8969
Epoch 8/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.4978 - categorical_accuracy: 0.8955 - val_loss: 0.4774 - val_categorical_accuracy: 0.8983
Epoch 9/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.4583 - categorical_accuracy: 0.9038 - val_loss: 0.4403 - val_categorical_accuracy: 0.9055
Epoch 10/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.4219 - categorical_accuracy: 0.9131 - val_loss: 0.4061 - val_categorical_accuracy: 0.9083
Epoch 11/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.3890 - categorical_accuracy: 0.9174 - val_loss: 0.3736 - val_categorical_accuracy: 0.9206
Epoch 12/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.3584 - categorical_accuracy: 0.9238 - val_loss: 0.3462 - val_categorical_accuracy: 0.9145
Epoch 13/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.3305 - categorical_accuracy: 0.9277 - val_loss: 0.3187 - val_categorical_accuracy: 0.9218
Epoch 14/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.3048 - categorical_accuracy: 0.9320 - val_loss: 0.2950 - val_categorical_accuracy: 0.9201
Epoch 15/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2811 - categorical_accuracy: 0.9368 - val_loss: 0.2714 - val_categorical_accuracy: 0.9309
Epoch 16/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.2597 - categorical_accuracy: 0.9397 - val_loss: 0.2548 - val_categorical_accuracy: 0.9105
Epoch 17/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2397 - categorical_accuracy: 0.9427 - val_loss: 0.2340 - val_categorical_accuracy: 0.9278
Epoch 18/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2215 - categorical_accuracy: 0.9447 - val_loss: 0.2175 - val_categorical_accuracy: 0.9201
Epoch 19/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2048 - categorical_accuracy: 0.9470 - val_loss: 0.2001 - val_categorical_accuracy: 0.9341
Epoch 20/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1896 - categorical_accuracy: 0.9482 - val_loss: 0.1846 - val_categorical_accuracy: 0.9364
Epoch 21/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1754 - categorical_accuracy: 0.9512 - val_loss: 0.1737 - val_categorical_accuracy: 0.9272
Epoch 22/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1624 - categorical_accuracy: 0.9520 - val_loss: 0.1589 - val_categorical_accuracy: 0.9380
Epoch 23/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.1505 - categorical_accuracy: 0.9531 - val_loss: 0.1473 - val_categorical_accuracy: 0.9414
Epoch 24/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1395 - categorical_accuracy: 0.9555 - val_loss: 0.1373 - val_categorical_accuracy: 0.9394
Epoch 25/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1293 - categorical_accuracy: 0.9573 - val_loss: 0.1276 - val_categorical_accuracy: 0.9420
Epoch 26/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1200 - categorical_accuracy: 0.9569 - val_loss: 0.1199 - val_categorical_accuracy: 0.9372
Epoch 27/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1118 - categorical_accuracy: 0.9578 - val_loss: 0.1133 - val_categorical_accuracy: 0.9309
Epoch 28/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1040 - categorical_accuracy: 0.9586 - val_loss: 0.1062 - val_categorical_accuracy: 0.9312
Epoch 29/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0968 - categorical_accuracy: 0.9593 - val_loss: 0.1004 - val_categorical_accuracy: 0.9283
Epoch 30/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0902 - categorical_accuracy: 0.9603 - val_loss: 0.0954 - val_categorical_accuracy: 0.9166
Epoch 31/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0843 - categorical_accuracy: 0.9610 - val_loss: 0.0862 - val_categorical_accuracy: 0.9376
Epoch 32/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0787 - categorical_accuracy: 0.9613 - val_loss: 0.0802 - val_categorical_accuracy: 0.9420
Epoch 33/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0735 - categorical_accuracy: 0.9637 - val_loss: 0.0752 - val_categorical_accuracy: 0.9432
Epoch 34/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0688 - categorical_accuracy: 0.9637 - val_loss: 0.0712 - val_categorical_accuracy: 0.9434
Epoch 35/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0646 - categorical_accuracy: 0.9626 - val_loss: 0.0692 - val_categorical_accuracy: 0.9318
Epoch 36/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0605 - categorical_accuracy: 0.9648 - val_loss: 0.0643 - val_categorical_accuracy: 0.9425
Epoch 37/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0569 - categorical_accuracy: 0.9646 - val_loss: 0.0636 - val_categorical_accuracy: 0.9177
Epoch 38/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0539 - categorical_accuracy: 0.9634 - val_loss: 0.0574 - val_categorical_accuracy: 0.9398
Epoch 39/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0508 - categorical_accuracy: 0.9643 - val_loss: 0.0530 - val_categorical_accuracy: 0.9489
Epoch 40/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0478 - categorical_accuracy: 0.9653 - val_loss: 0.0538 - val_categorical_accuracy: 0.9294
Epoch 41/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0453 - categorical_accuracy: 0.9655 - val_loss: 0.0475 - val_categorical_accuracy: 0.9486
Epoch 42/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0428 - categorical_accuracy: 0.9663 - val_loss: 0.0479 - val_categorical_accuracy: 0.9344
Epoch 43/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0407 - categorical_accuracy: 0.9660 - val_loss: 0.0485 - val_categorical_accuracy: 0.9265
Epoch 44/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0388 - categorical_accuracy: 0.9657 - val_loss: 0.0425 - val_categorical_accuracy: 0.9436
Epoch 45/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0366 - categorical_accuracy: 0.9680 - val_loss: 0.0398 - val_categorical_accuracy: 0.9479
Epoch 46/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0353 - categorical_accuracy: 0.9659 - val_loss: 0.0395 - val_categorical_accuracy: 0.9454
Epoch 47/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0338 - categorical_accuracy: 0.9662 - val_loss: 0.0384 - val_categorical_accuracy: 0.9416
Epoch 48/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0320 - categorical_accuracy: 0.9681 - val_loss: 0.0350 - val_categorical_accuracy: 0.9504
Epoch 49/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0309 - categorical_accuracy: 0.9677 - val_loss: 0.0334 - val_categorical_accuracy: 0.9514
Epoch 50/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0296 - categorical_accuracy: 0.9682 - val_loss: 0.0386 - val_categorical_accuracy: 0.9180
Epoch 51/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0284 - categorical_accuracy: 0.9683 - val_loss: 0.0350 - val_categorical_accuracy: 0.9395
Epoch 52/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0274 - categorical_accuracy: 0.9682 - val_loss: 0.0358 - val_categorical_accuracy: 0.9265
Epoch 53/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0266 - categorical_accuracy: 0.9682 - val_loss: 0.0323 - val_categorical_accuracy: 0.9375
Epoch 54/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0257 - categorical_accuracy: 0.9680 - val_loss: 0.0372 - val_categorical_accuracy: 0.9085
Epoch 55/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0250 - categorical_accuracy: 0.9676 - val_loss: 0.0291 - val_categorical_accuracy: 0.9472
Epoch 56/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0240 - categorical_accuracy: 0.9688 - val_loss: 0.0287 - val_categorical_accuracy: 0.9498
Epoch 57/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0232 - categorical_accuracy: 0.9692 - val_loss: 0.0301 - val_categorical_accuracy: 0.9390
Epoch 58/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0229 - categorical_accuracy: 0.9687 - val_loss: 0.0341 - val_categorical_accuracy: 0.9246
Epoch 59/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0225 - categorical_accuracy: 0.9676 - val_loss: 0.0465 - val_categorical_accuracy: 0.8515
Epoch 60/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0216 - categorical_accuracy: 0.9696 - val_loss: 0.0266 - val_categorical_accuracy: 0.9468
Epoch 61/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0208 - categorical_accuracy: 0.9707 - val_loss: 0.0271 - val_categorical_accuracy: 0.9432
Epoch 62/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0208 - categorical_accuracy: 0.9691 - val_loss: 0.0275 - val_categorical_accuracy: 0.9407
Epoch 63/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0202 - categorical_accuracy: 0.9697 - val_loss: 0.0262 - val_categorical_accuracy: 0.9463
Epoch 64/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0197 - categorical_accuracy: 0.9703 - val_loss: 0.0258 - val_categorical_accuracy: 0.9424
Epoch 65/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0195 - categorical_accuracy: 0.9690 - val_loss: 0.0232 - val_categorical_accuracy: 0.9520
Epoch 66/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0189 - categorical_accuracy: 0.9708 - val_loss: 0.0241 - val_categorical_accuracy: 0.9470
Epoch 67/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0188 - categorical_accuracy: 0.9702 - val_loss: 0.0239 - val_categorical_accuracy: 0.9459
Epoch 68/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0182 - categorical_accuracy: 0.9713 - val_loss: 0.0232 - val_categorical_accuracy: 0.9492
Epoch 69/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0178 - categorical_accuracy: 0.9725 - val_loss: 0.0291 - val_categorical_accuracy: 0.9223
Epoch 70/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0177 - categorical_accuracy: 0.9713 - val_loss: 0.0236 - val_categorical_accuracy: 0.9498
Epoch 71/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0176 - categorical_accuracy: 0.9705 - val_loss: 0.0242 - val_categorical_accuracy: 0.9478

Epoch 00071: ReduceLROnPlateau reducing learning rate to 0.1.
Epoch 72/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0154 - categorical_accuracy: 0.9821 - val_loss: 0.0189 - val_categorical_accuracy: 0.9653
Epoch 73/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0147 - categorical_accuracy: 0.9850 - val_loss: 0.0189 - val_categorical_accuracy: 0.9657
Epoch 74/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0143 - categorical_accuracy: 0.9865 - val_loss: 0.0188 - val_categorical_accuracy: 0.9658
Epoch 75/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0140 - categorical_accuracy: 0.9872 - val_loss: 0.0184 - val_categorical_accuracy: 0.9667
Epoch 76/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0137 - categorical_accuracy: 0.9882 - val_loss: 0.0182 - val_categorical_accuracy: 0.9664
Epoch 77/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0135 - categorical_accuracy: 0.9886 - val_loss: 0.0196 - val_categorical_accuracy: 0.9644
Epoch 78/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0132 - categorical_accuracy: 0.9891 - val_loss: 0.0182 - val_categorical_accuracy: 0.9668
Epoch 79/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0131 - categorical_accuracy: 0.9885 - val_loss: 0.0183 - val_categorical_accuracy: 0.9655
Epoch 80/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0127 - categorical_accuracy: 0.9895 - val_loss: 0.0185 - val_categorical_accuracy: 0.9649
Epoch 81/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0126 - categorical_accuracy: 0.9891 - val_loss: 0.0182 - val_categorical_accuracy: 0.9649
Epoch 82/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0124 - categorical_accuracy: 0.9900 - val_loss: 0.0179 - val_categorical_accuracy: 0.9655
Epoch 83/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0121 - categorical_accuracy: 0.9908 - val_loss: 0.0174 - val_categorical_accuracy: 0.9675
Epoch 84/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0119 - categorical_accuracy: 0.9905 - val_loss: 0.0177 - val_categorical_accuracy: 0.9661
Epoch 85/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0117 - categorical_accuracy: 0.9909 - val_loss: 0.0179 - val_categorical_accuracy: 0.9665
Epoch 86/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0116 - categorical_accuracy: 0.9909 - val_loss: 0.0185 - val_categorical_accuracy: 0.9657
Epoch 87/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0115 - categorical_accuracy: 0.9904 - val_loss: 0.0169 - val_categorical_accuracy: 0.9671
Epoch 88/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0112 - categorical_accuracy: 0.9914 - val_loss: 0.0172 - val_categorical_accuracy: 0.9665
Epoch 89/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0111 - categorical_accuracy: 0.9911 - val_loss: 0.0177 - val_categorical_accuracy: 0.9649
Epoch 90/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0109 - categorical_accuracy: 0.9915 - val_loss: 0.0170 - val_categorical_accuracy: 0.9668
Epoch 91/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0107 - categorical_accuracy: 0.9922 - val_loss: 0.0175 - val_categorical_accuracy: 0.9658
Epoch 92/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0106 - categorical_accuracy: 0.9918 - val_loss: 0.0170 - val_categorical_accuracy: 0.9666
Epoch 93/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0104 - categorical_accuracy: 0.9920 - val_loss: 0.0169 - val_categorical_accuracy: 0.9672

Epoch 00093: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 94/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0102 - categorical_accuracy: 0.9929 - val_loss: 0.0165 - val_categorical_accuracy: 0.9681
Epoch 95/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0101 - categorical_accuracy: 0.9938 - val_loss: 0.0163 - val_categorical_accuracy: 0.9684
Epoch 96/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0101 - categorical_accuracy: 0.9932 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 97/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0100 - categorical_accuracy: 0.9936 - val_loss: 0.0163 - val_categorical_accuracy: 0.9689
Epoch 98/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0100 - categorical_accuracy: 0.9938 - val_loss: 0.0162 - val_categorical_accuracy: 0.9681
Epoch 99/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0099 - categorical_accuracy: 0.9937 - val_loss: 0.0163 - val_categorical_accuracy: 0.9684
Epoch 100/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0098 - categorical_accuracy: 0.9942 - val_loss: 0.0163 - val_categorical_accuracy: 0.9681
Epoch 101/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0098 - categorical_accuracy: 0.9941 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688
Epoch 102/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0098 - categorical_accuracy: 0.9941 - val_loss: 0.0164 - val_categorical_accuracy: 0.9684

Epoch 00102: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 103/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9945 - val_loss: 0.0163 - val_categorical_accuracy: 0.9684
Epoch 104/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9942 - val_loss: 0.0163 - val_categorical_accuracy: 0.9685
Epoch 105/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9944 - val_loss: 0.0163 - val_categorical_accuracy: 0.9687
Epoch 106/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9945 - val_loss: 0.0163 - val_categorical_accuracy: 0.9687
Epoch 107/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9943 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686

Epoch 00107: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 108/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9943 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 109/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9943 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 110/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9947 - val_loss: 0.0162 - val_categorical_accuracy: 0.9689
Epoch 111/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688
Epoch 112/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9943 - val_loss: 0.0162 - val_categorical_accuracy: 0.9689

Epoch 00112: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 113/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688
Epoch 114/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688
Epoch 115/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9950 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688
Epoch 116/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688
Epoch 117/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9942 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00117: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 118/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9941 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 119/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688
Epoch 120/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9949 - val_loss: 0.0162 - val_categorical_accuracy: 0.9690
Epoch 121/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 122/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00122: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 123/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9941 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 124/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 125/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9950 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 126/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 127/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9948 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688

Epoch 00127: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 128/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 129/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 130/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 131/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 132/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9950 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00132: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 133/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 134/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 135/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9949 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 136/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 137/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00137: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 138/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9951 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 139/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 140/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9947 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 141/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 142/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00142: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 143/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9948 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 144/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 145/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 146/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 147/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00147: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 148/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 149/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9948 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 150/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9947 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 151/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9949 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 152/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9947 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686

Epoch 00152: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 153/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9949 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 154/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 155/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 156/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9942 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688
Epoch 157/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00157: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 158/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 159/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9948 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 160/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9941 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 161/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9948 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 162/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00162: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 163/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9943 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 164/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9947 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 165/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 166/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 167/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00167: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 168/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 169/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0097 - categorical_accuracy: 0.9943 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 170/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9945 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 171/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 172/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687

Epoch 00172: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 173/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9941 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687
Epoch 174/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9948 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687
Epoch 175/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9952 - val_loss: 0.0161 - val_categorical_accuracy: 0.9686
Epoch 176/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9950 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687
Epoch 177/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9947 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687

Epoch 00177: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 178/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 179/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9945 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687
Epoch 180/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9688
Epoch 181/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9947 - val_loss: 0.0161 - val_categorical_accuracy: 0.9688
Epoch 182/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9951 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687

Epoch 00182: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 183/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9947 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687
Epoch 184/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 185/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9947 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 186/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0162 - val_categorical_accuracy: 0.9685
Epoch 187/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9944 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687

Epoch 00187: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 188/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9949 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687
Epoch 189/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9950 - val_loss: 0.0161 - val_categorical_accuracy: 0.9687
Epoch 190/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9947 - val_loss: 0.0162 - val_categorical_accuracy: 0.9687
Epoch 191/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686
Epoch 192/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9946 - val_loss: 0.0162 - val_categorical_accuracy: 0.9686

Epoch 00192: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 00192: early stopping
========= generating oof predictions 18:14:48 =========
========= generating test set predictions 18:14:50 =========
========= fitting 2 th model 18:15:14 =========
Epoch 1/65535
420/420 [==============================] - 28s 67ms/step - loss: 0.9725 - categorical_accuracy: 0.2366 - val_loss: 0.9052 - val_categorical_accuracy: 0.4655
Epoch 2/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.8524 - categorical_accuracy: 0.5978 - val_loss: 0.8072 - val_categorical_accuracy: 0.6883
Epoch 3/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.7662 - categorical_accuracy: 0.7580 - val_loss: 0.7268 - val_categorical_accuracy: 0.8215
Epoch 4/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.6986 - categorical_accuracy: 0.8197 - val_loss: 0.6655 - val_categorical_accuracy: 0.8606
Epoch 5/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.6404 - categorical_accuracy: 0.8504 - val_loss: 0.6109 - val_categorical_accuracy: 0.8845
Epoch 6/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.5882 - categorical_accuracy: 0.8705 - val_loss: 0.5626 - val_categorical_accuracy: 0.8923
Epoch 7/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.5408 - categorical_accuracy: 0.8848 - val_loss: 0.5171 - val_categorical_accuracy: 0.9039
Epoch 8/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.4974 - categorical_accuracy: 0.8975 - val_loss: 0.4764 - val_categorical_accuracy: 0.9081
Epoch 9/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.4580 - categorical_accuracy: 0.9071 - val_loss: 0.4401 - val_categorical_accuracy: 0.9065
Epoch 10/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.4218 - categorical_accuracy: 0.9131 - val_loss: 0.4048 - val_categorical_accuracy: 0.9151
Epoch 11/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.3886 - categorical_accuracy: 0.9204 - val_loss: 0.3736 - val_categorical_accuracy: 0.9204
Epoch 12/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.3582 - categorical_accuracy: 0.9257 - val_loss: 0.3450 - val_categorical_accuracy: 0.9221
Epoch 13/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.3302 - categorical_accuracy: 0.9299 - val_loss: 0.3198 - val_categorical_accuracy: 0.9162
Epoch 14/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.3049 - categorical_accuracy: 0.9315 - val_loss: 0.2966 - val_categorical_accuracy: 0.9122
Epoch 15/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2812 - categorical_accuracy: 0.9367 - val_loss: 0.2721 - val_categorical_accuracy: 0.9260
Epoch 16/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2595 - categorical_accuracy: 0.9399 - val_loss: 0.2517 - val_categorical_accuracy: 0.9289
Epoch 17/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2399 - categorical_accuracy: 0.9414 - val_loss: 0.2316 - val_categorical_accuracy: 0.9386
Epoch 18/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2215 - categorical_accuracy: 0.9450 - val_loss: 0.2171 - val_categorical_accuracy: 0.9238
Epoch 19/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2047 - categorical_accuracy: 0.9475 - val_loss: 0.2001 - val_categorical_accuracy: 0.9298
Epoch 20/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1893 - categorical_accuracy: 0.9497 - val_loss: 0.1845 - val_categorical_accuracy: 0.9387
Epoch 21/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1752 - categorical_accuracy: 0.9509 - val_loss: 0.1709 - val_categorical_accuracy: 0.9383
Epoch 22/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1624 - categorical_accuracy: 0.9515 - val_loss: 0.1576 - val_categorical_accuracy: 0.9438
Epoch 23/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1504 - categorical_accuracy: 0.9538 - val_loss: 0.1487 - val_categorical_accuracy: 0.9328
Epoch 24/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1394 - categorical_accuracy: 0.9554 - val_loss: 0.1378 - val_categorical_accuracy: 0.9369
Epoch 25/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1294 - categorical_accuracy: 0.9549 - val_loss: 0.1289 - val_categorical_accuracy: 0.9377
Epoch 26/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1202 - categorical_accuracy: 0.9565 - val_loss: 0.1191 - val_categorical_accuracy: 0.9450
Epoch 27/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1115 - categorical_accuracy: 0.9590 - val_loss: 0.1100 - val_categorical_accuracy: 0.9455
Epoch 28/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1038 - categorical_accuracy: 0.9594 - val_loss: 0.1040 - val_categorical_accuracy: 0.9394
Epoch 29/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0967 - categorical_accuracy: 0.9601 - val_loss: 0.0976 - val_categorical_accuracy: 0.9374
Epoch 30/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0901 - categorical_accuracy: 0.9599 - val_loss: 0.0905 - val_categorical_accuracy: 0.9434
Epoch 31/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0841 - categorical_accuracy: 0.9615 - val_loss: 0.0849 - val_categorical_accuracy: 0.9435
Epoch 32/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0787 - categorical_accuracy: 0.9614 - val_loss: 0.0804 - val_categorical_accuracy: 0.9384
Epoch 33/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0737 - categorical_accuracy: 0.9622 - val_loss: 0.0744 - val_categorical_accuracy: 0.9454
Epoch 34/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0692 - categorical_accuracy: 0.9612 - val_loss: 0.0754 - val_categorical_accuracy: 0.9161
Epoch 35/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0647 - categorical_accuracy: 0.9628 - val_loss: 0.0692 - val_categorical_accuracy: 0.9314
Epoch 36/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0606 - categorical_accuracy: 0.9644 - val_loss: 0.0641 - val_categorical_accuracy: 0.9393
Epoch 37/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0571 - categorical_accuracy: 0.9639 - val_loss: 0.0615 - val_categorical_accuracy: 0.9340
Epoch 38/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0537 - categorical_accuracy: 0.9648 - val_loss: 0.0563 - val_categorical_accuracy: 0.9440
Epoch 39/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0509 - categorical_accuracy: 0.9640 - val_loss: 0.0529 - val_categorical_accuracy: 0.9449
Epoch 40/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0478 - categorical_accuracy: 0.9662 - val_loss: 0.0506 - val_categorical_accuracy: 0.9433
Epoch 41/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0454 - categorical_accuracy: 0.9654 - val_loss: 0.0494 - val_categorical_accuracy: 0.9422
Epoch 42/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0430 - categorical_accuracy: 0.9656 - val_loss: 0.0467 - val_categorical_accuracy: 0.9413
Epoch 43/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0406 - categorical_accuracy: 0.9681 - val_loss: 0.0431 - val_categorical_accuracy: 0.9494
Epoch 44/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0386 - categorical_accuracy: 0.9661 - val_loss: 0.0447 - val_categorical_accuracy: 0.9334
Epoch 45/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0369 - categorical_accuracy: 0.9656 - val_loss: 0.0429 - val_categorical_accuracy: 0.9352
Epoch 46/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0354 - categorical_accuracy: 0.9657 - val_loss: 0.0405 - val_categorical_accuracy: 0.9337
Epoch 47/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0339 - categorical_accuracy: 0.9667 - val_loss: 0.0379 - val_categorical_accuracy: 0.9439
Epoch 48/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0321 - categorical_accuracy: 0.9680 - val_loss: 0.0366 - val_categorical_accuracy: 0.9424
Epoch 49/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0306 - categorical_accuracy: 0.9679 - val_loss: 0.0349 - val_categorical_accuracy: 0.9448
Epoch 50/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0297 - categorical_accuracy: 0.9672 - val_loss: 0.0352 - val_categorical_accuracy: 0.9345
Epoch 51/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0287 - categorical_accuracy: 0.9661 - val_loss: 0.0341 - val_categorical_accuracy: 0.9383
Epoch 52/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0275 - categorical_accuracy: 0.9679 - val_loss: 0.0342 - val_categorical_accuracy: 0.9357
Epoch 53/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0266 - categorical_accuracy: 0.9679 - val_loss: 0.0377 - val_categorical_accuracy: 0.9087
Epoch 54/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0257 - categorical_accuracy: 0.9687 - val_loss: 0.0332 - val_categorical_accuracy: 0.9299
Epoch 55/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0247 - categorical_accuracy: 0.9696 - val_loss: 0.0291 - val_categorical_accuracy: 0.9453
Epoch 56/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0242 - categorical_accuracy: 0.9680 - val_loss: 0.0272 - val_categorical_accuracy: 0.9498
Epoch 57/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0234 - categorical_accuracy: 0.9694 - val_loss: 0.0288 - val_categorical_accuracy: 0.9425
Epoch 58/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0228 - categorical_accuracy: 0.9691 - val_loss: 0.0323 - val_categorical_accuracy: 0.9238
Epoch 59/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0222 - categorical_accuracy: 0.9689 - val_loss: 0.0266 - val_categorical_accuracy: 0.9486
Epoch 60/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0214 - categorical_accuracy: 0.9705 - val_loss: 0.0296 - val_categorical_accuracy: 0.9289
Epoch 61/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0210 - categorical_accuracy: 0.9698 - val_loss: 0.0265 - val_categorical_accuracy: 0.9442
Epoch 62/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0206 - categorical_accuracy: 0.9697 - val_loss: 0.0245 - val_categorical_accuracy: 0.9487
Epoch 63/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0201 - categorical_accuracy: 0.9703 - val_loss: 0.0241 - val_categorical_accuracy: 0.9475
Epoch 64/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0199 - categorical_accuracy: 0.9694 - val_loss: 0.0279 - val_categorical_accuracy: 0.9401
Epoch 65/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0193 - categorical_accuracy: 0.9706 - val_loss: 0.0261 - val_categorical_accuracy: 0.9411
Epoch 66/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0193 - categorical_accuracy: 0.9681 - val_loss: 0.0291 - val_categorical_accuracy: 0.9295
Epoch 67/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0186 - categorical_accuracy: 0.9715 - val_loss: 0.0268 - val_categorical_accuracy: 0.9348
Epoch 68/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0186 - categorical_accuracy: 0.9703 - val_loss: 0.0236 - val_categorical_accuracy: 0.9468
Epoch 69/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0181 - categorical_accuracy: 0.9708 - val_loss: 0.0258 - val_categorical_accuracy: 0.9397
Epoch 70/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0179 - categorical_accuracy: 0.9722 - val_loss: 0.0255 - val_categorical_accuracy: 0.9346
Epoch 71/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0175 - categorical_accuracy: 0.9717 - val_loss: 0.0219 - val_categorical_accuracy: 0.9509
Epoch 72/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0171 - categorical_accuracy: 0.9724 - val_loss: 0.0239 - val_categorical_accuracy: 0.9413
Epoch 73/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0172 - categorical_accuracy: 0.9703 - val_loss: 0.0339 - val_categorical_accuracy: 0.9044
Epoch 74/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0170 - categorical_accuracy: 0.9711 - val_loss: 0.0211 - val_categorical_accuracy: 0.9527
Epoch 75/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0169 - categorical_accuracy: 0.9713 - val_loss: 0.0230 - val_categorical_accuracy: 0.9387
Epoch 76/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0169 - categorical_accuracy: 0.9708 - val_loss: 0.0298 - val_categorical_accuracy: 0.9079
Epoch 77/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0166 - categorical_accuracy: 0.9714 - val_loss: 0.0248 - val_categorical_accuracy: 0.9282
Epoch 78/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0165 - categorical_accuracy: 0.9720 - val_loss: 0.0220 - val_categorical_accuracy: 0.9471
Epoch 79/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0164 - categorical_accuracy: 0.9713 - val_loss: 0.0219 - val_categorical_accuracy: 0.9459
Epoch 80/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0161 - categorical_accuracy: 0.9722 - val_loss: 0.0223 - val_categorical_accuracy: 0.9467

Epoch 00080: ReduceLROnPlateau reducing learning rate to 0.1.
Epoch 81/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0142 - categorical_accuracy: 0.9834 - val_loss: 0.0177 - val_categorical_accuracy: 0.9661
Epoch 82/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0134 - categorical_accuracy: 0.9868 - val_loss: 0.0173 - val_categorical_accuracy: 0.9670
Epoch 83/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0131 - categorical_accuracy: 0.9875 - val_loss: 0.0173 - val_categorical_accuracy: 0.9664
Epoch 84/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0128 - categorical_accuracy: 0.9880 - val_loss: 0.0171 - val_categorical_accuracy: 0.9683
Epoch 85/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0125 - categorical_accuracy: 0.9894 - val_loss: 0.0178 - val_categorical_accuracy: 0.9669
Epoch 86/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0123 - categorical_accuracy: 0.9898 - val_loss: 0.0171 - val_categorical_accuracy: 0.9666
Epoch 87/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0120 - categorical_accuracy: 0.9898 - val_loss: 0.0174 - val_categorical_accuracy: 0.9684
Epoch 88/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0117 - categorical_accuracy: 0.9910 - val_loss: 0.0172 - val_categorical_accuracy: 0.9665
Epoch 89/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0116 - categorical_accuracy: 0.9903 - val_loss: 0.0169 - val_categorical_accuracy: 0.9677
Epoch 90/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0114 - categorical_accuracy: 0.9909 - val_loss: 0.0172 - val_categorical_accuracy: 0.9663
Epoch 91/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0112 - categorical_accuracy: 0.9916 - val_loss: 0.0167 - val_categorical_accuracy: 0.9678
Epoch 92/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0110 - categorical_accuracy: 0.9914 - val_loss: 0.0168 - val_categorical_accuracy: 0.9672
Epoch 93/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0108 - categorical_accuracy: 0.9916 - val_loss: 0.0168 - val_categorical_accuracy: 0.9674
Epoch 94/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0106 - categorical_accuracy: 0.9922 - val_loss: 0.0166 - val_categorical_accuracy: 0.9670
Epoch 95/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0105 - categorical_accuracy: 0.9923 - val_loss: 0.0164 - val_categorical_accuracy: 0.9674
Epoch 96/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0103 - categorical_accuracy: 0.9923 - val_loss: 0.0165 - val_categorical_accuracy: 0.9679
Epoch 97/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0102 - categorical_accuracy: 0.9923 - val_loss: 0.0163 - val_categorical_accuracy: 0.9672
Epoch 98/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0100 - categorical_accuracy: 0.9927 - val_loss: 0.0164 - val_categorical_accuracy: 0.9675
Epoch 99/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0098 - categorical_accuracy: 0.9931 - val_loss: 0.0163 - val_categorical_accuracy: 0.9668
Epoch 100/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9934 - val_loss: 0.0158 - val_categorical_accuracy: 0.9670
Epoch 101/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0095 - categorical_accuracy: 0.9933 - val_loss: 0.0158 - val_categorical_accuracy: 0.9678
Epoch 102/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0094 - categorical_accuracy: 0.9935 - val_loss: 0.0162 - val_categorical_accuracy: 0.9671
Epoch 103/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0092 - categorical_accuracy: 0.9939 - val_loss: 0.0160 - val_categorical_accuracy: 0.9675
Epoch 104/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0091 - categorical_accuracy: 0.9939 - val_loss: 0.0158 - val_categorical_accuracy: 0.9664
Epoch 105/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0090 - categorical_accuracy: 0.9941 - val_loss: 0.0158 - val_categorical_accuracy: 0.9669
Epoch 106/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0088 - categorical_accuracy: 0.9944 - val_loss: 0.0157 - val_categorical_accuracy: 0.9672
Epoch 107/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0088 - categorical_accuracy: 0.9941 - val_loss: 0.0156 - val_categorical_accuracy: 0.9673
Epoch 108/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0086 - categorical_accuracy: 0.9944 - val_loss: 0.0154 - val_categorical_accuracy: 0.9674
Epoch 109/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0084 - categorical_accuracy: 0.9949 - val_loss: 0.0161 - val_categorical_accuracy: 0.9660
Epoch 110/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0084 - categorical_accuracy: 0.9944 - val_loss: 0.0158 - val_categorical_accuracy: 0.9656
Epoch 111/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0082 - categorical_accuracy: 0.9950 - val_loss: 0.0153 - val_categorical_accuracy: 0.9678
Epoch 112/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0082 - categorical_accuracy: 0.9944 - val_loss: 0.0153 - val_categorical_accuracy: 0.9666
Epoch 113/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9950 - val_loss: 0.0151 - val_categorical_accuracy: 0.9677
Epoch 114/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9946 - val_loss: 0.0154 - val_categorical_accuracy: 0.9651
Epoch 115/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0078 - categorical_accuracy: 0.9954 - val_loss: 0.0156 - val_categorical_accuracy: 0.9666
Epoch 116/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0077 - categorical_accuracy: 0.9953 - val_loss: 0.0156 - val_categorical_accuracy: 0.9653
Epoch 117/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0077 - categorical_accuracy: 0.9951 - val_loss: 0.0154 - val_categorical_accuracy: 0.9640
Epoch 118/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0076 - categorical_accuracy: 0.9948 - val_loss: 0.0158 - val_categorical_accuracy: 0.9623
Epoch 119/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0075 - categorical_accuracy: 0.9951 - val_loss: 0.0154 - val_categorical_accuracy: 0.9652

Epoch 00119: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 120/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0073 - categorical_accuracy: 0.9962 - val_loss: 0.0149 - val_categorical_accuracy: 0.9683
Epoch 121/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0071 - categorical_accuracy: 0.9966 - val_loss: 0.0148 - val_categorical_accuracy: 0.9685
Epoch 122/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9974 - val_loss: 0.0147 - val_categorical_accuracy: 0.9684
Epoch 123/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9973 - val_loss: 0.0148 - val_categorical_accuracy: 0.9685
Epoch 124/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0070 - categorical_accuracy: 0.9971 - val_loss: 0.0147 - val_categorical_accuracy: 0.9693
Epoch 125/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9971 - val_loss: 0.0147 - val_categorical_accuracy: 0.9691
Epoch 126/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9973 - val_loss: 0.0147 - val_categorical_accuracy: 0.9684
Epoch 127/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0069 - categorical_accuracy: 0.9976 - val_loss: 0.0147 - val_categorical_accuracy: 0.9693
Epoch 128/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0069 - categorical_accuracy: 0.9974 - val_loss: 0.0147 - val_categorical_accuracy: 0.9690

Epoch 00128: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 129/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0069 - categorical_accuracy: 0.9977 - val_loss: 0.0147 - val_categorical_accuracy: 0.9689
Epoch 130/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9978 - val_loss: 0.0147 - val_categorical_accuracy: 0.9690
Epoch 131/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9979 - val_loss: 0.0147 - val_categorical_accuracy: 0.9692
Epoch 132/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9979 - val_loss: 0.0147 - val_categorical_accuracy: 0.9688
Epoch 133/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9977 - val_loss: 0.0147 - val_categorical_accuracy: 0.9691

Epoch 00133: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 134/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9977 - val_loss: 0.0147 - val_categorical_accuracy: 0.9693
Epoch 135/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9977 - val_loss: 0.0147 - val_categorical_accuracy: 0.9692
Epoch 136/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0068 - categorical_accuracy: 0.9978 - val_loss: 0.0147 - val_categorical_accuracy: 0.9692
Epoch 137/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9979 - val_loss: 0.0147 - val_categorical_accuracy: 0.9693
Epoch 138/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9979 - val_loss: 0.0147 - val_categorical_accuracy: 0.9693

Epoch 00138: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 139/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9976 - val_loss: 0.0147 - val_categorical_accuracy: 0.9692
Epoch 140/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9978 - val_loss: 0.0147 - val_categorical_accuracy: 0.9691
Epoch 141/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0067 - categorical_accuracy: 0.9980 - val_loss: 0.0147 - val_categorical_accuracy: 0.9691
Epoch 142/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0067 - categorical_accuracy: 0.9981 - val_loss: 0.0147 - val_categorical_accuracy: 0.9690
Epoch 143/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0068 - categorical_accuracy: 0.9978 - val_loss: 0.0147 - val_categorical_accuracy: 0.9690

Epoch 00143: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 144/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9978 - val_loss: 0.0147 - val_categorical_accuracy: 0.9690
Epoch 00144: early stopping
========= generating oof predictions 19:18:50 =========
========= generating test set predictions 19:18:52 =========
========= fitting 3 th model 19:19:16 =========
Epoch 1/65535
420/420 [==============================] - 28s 66ms/step - loss: 0.9738 - categorical_accuracy: 0.2331 - val_loss: 0.9054 - val_categorical_accuracy: 0.4624
Epoch 2/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.8529 - categorical_accuracy: 0.6007 - val_loss: 0.8030 - val_categorical_accuracy: 0.7320
Epoch 3/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.7674 - categorical_accuracy: 0.7513 - val_loss: 0.7283 - val_categorical_accuracy: 0.8174
Epoch 4/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.6998 - categorical_accuracy: 0.8146 - val_loss: 0.6676 - val_categorical_accuracy: 0.8442
Epoch 5/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.6414 - categorical_accuracy: 0.8471 - val_loss: 0.6117 - val_categorical_accuracy: 0.8745
Epoch 6/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.5890 - categorical_accuracy: 0.8688 - val_loss: 0.5626 - val_categorical_accuracy: 0.8891
Epoch 7/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.5415 - categorical_accuracy: 0.8835 - val_loss: 0.5220 - val_categorical_accuracy: 0.8719
Epoch 8/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.4982 - categorical_accuracy: 0.8954 - val_loss: 0.4780 - val_categorical_accuracy: 0.9081
Epoch 9/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.4588 - categorical_accuracy: 0.9033 - val_loss: 0.4406 - val_categorical_accuracy: 0.9098
Epoch 10/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.4222 - categorical_accuracy: 0.9125 - val_loss: 0.4052 - val_categorical_accuracy: 0.9179
Epoch 11/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.3892 - categorical_accuracy: 0.9183 - val_loss: 0.3744 - val_categorical_accuracy: 0.9195
Epoch 12/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.3586 - categorical_accuracy: 0.9247 - val_loss: 0.3442 - val_categorical_accuracy: 0.9300
Epoch 13/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.3307 - categorical_accuracy: 0.9284 - val_loss: 0.3184 - val_categorical_accuracy: 0.9262
Epoch 14/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.3049 - categorical_accuracy: 0.9336 - val_loss: 0.2934 - val_categorical_accuracy: 0.9319
Epoch 15/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2815 - categorical_accuracy: 0.9357 - val_loss: 0.2717 - val_categorical_accuracy: 0.9308
Epoch 16/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2599 - categorical_accuracy: 0.9386 - val_loss: 0.2515 - val_categorical_accuracy: 0.9312
Epoch 17/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2399 - categorical_accuracy: 0.9413 - val_loss: 0.2322 - val_categorical_accuracy: 0.9341
Epoch 18/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2216 - categorical_accuracy: 0.9451 - val_loss: 0.2154 - val_categorical_accuracy: 0.9355
Epoch 19/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2049 - categorical_accuracy: 0.9460 - val_loss: 0.1983 - val_categorical_accuracy: 0.9415
Epoch 20/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1895 - categorical_accuracy: 0.9475 - val_loss: 0.1854 - val_categorical_accuracy: 0.9349
Epoch 21/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1752 - categorical_accuracy: 0.9504 - val_loss: 0.1714 - val_categorical_accuracy: 0.9391
Epoch 22/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1622 - categorical_accuracy: 0.9528 - val_loss: 0.1582 - val_categorical_accuracy: 0.9446
Epoch 23/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1504 - categorical_accuracy: 0.9531 - val_loss: 0.1493 - val_categorical_accuracy: 0.9296
Epoch 24/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1395 - categorical_accuracy: 0.9550 - val_loss: 0.1378 - val_categorical_accuracy: 0.9384
Epoch 25/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1294 - categorical_accuracy: 0.9558 - val_loss: 0.1269 - val_categorical_accuracy: 0.9465
Epoch 26/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1200 - categorical_accuracy: 0.9584 - val_loss: 0.1181 - val_categorical_accuracy: 0.9450
Epoch 27/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1116 - categorical_accuracy: 0.9582 - val_loss: 0.1128 - val_categorical_accuracy: 0.9352
Epoch 28/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1040 - categorical_accuracy: 0.9585 - val_loss: 0.1038 - val_categorical_accuracy: 0.9405
Epoch 29/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0967 - categorical_accuracy: 0.9600 - val_loss: 0.0972 - val_categorical_accuracy: 0.9429
Epoch 30/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0901 - categorical_accuracy: 0.9604 - val_loss: 0.0919 - val_categorical_accuracy: 0.9403
Epoch 31/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0840 - categorical_accuracy: 0.9616 - val_loss: 0.0860 - val_categorical_accuracy: 0.9435
Epoch 32/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0784 - categorical_accuracy: 0.9630 - val_loss: 0.0786 - val_categorical_accuracy: 0.9499
Epoch 33/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0733 - categorical_accuracy: 0.9638 - val_loss: 0.0748 - val_categorical_accuracy: 0.9427
Epoch 34/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0687 - categorical_accuracy: 0.9629 - val_loss: 0.0713 - val_categorical_accuracy: 0.9409
Epoch 35/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0645 - categorical_accuracy: 0.9642 - val_loss: 0.0664 - val_categorical_accuracy: 0.9445
Epoch 36/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0605 - categorical_accuracy: 0.9647 - val_loss: 0.0654 - val_categorical_accuracy: 0.9340
Epoch 37/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0567 - categorical_accuracy: 0.9655 - val_loss: 0.0609 - val_categorical_accuracy: 0.9402
Epoch 38/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0538 - categorical_accuracy: 0.9638 - val_loss: 0.0585 - val_categorical_accuracy: 0.9344
Epoch 39/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0505 - categorical_accuracy: 0.9646 - val_loss: 0.0533 - val_categorical_accuracy: 0.9470
Epoch 40/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0478 - categorical_accuracy: 0.9654 - val_loss: 0.0549 - val_categorical_accuracy: 0.9229
Epoch 41/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0452 - categorical_accuracy: 0.9657 - val_loss: 0.0530 - val_categorical_accuracy: 0.9180
Epoch 42/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0428 - categorical_accuracy: 0.9659 - val_loss: 0.0481 - val_categorical_accuracy: 0.9429
Epoch 43/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0405 - categorical_accuracy: 0.9665 - val_loss: 0.0471 - val_categorical_accuracy: 0.9353
Epoch 44/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0389 - categorical_accuracy: 0.9654 - val_loss: 0.0461 - val_categorical_accuracy: 0.9215
Epoch 45/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0368 - categorical_accuracy: 0.9666 - val_loss: 0.0420 - val_categorical_accuracy: 0.9411
Epoch 46/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0350 - categorical_accuracy: 0.9677 - val_loss: 0.0408 - val_categorical_accuracy: 0.9367
Epoch 47/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0336 - categorical_accuracy: 0.9672 - val_loss: 0.0382 - val_categorical_accuracy: 0.9439
Epoch 48/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0322 - categorical_accuracy: 0.9667 - val_loss: 0.0384 - val_categorical_accuracy: 0.9316
Epoch 49/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0309 - categorical_accuracy: 0.9675 - val_loss: 0.0371 - val_categorical_accuracy: 0.9318
Epoch 50/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0296 - categorical_accuracy: 0.9680 - val_loss: 0.0367 - val_categorical_accuracy: 0.9285
Epoch 51/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0285 - categorical_accuracy: 0.9685 - val_loss: 0.0346 - val_categorical_accuracy: 0.9340
Epoch 52/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0273 - categorical_accuracy: 0.9693 - val_loss: 0.0313 - val_categorical_accuracy: 0.9471
Epoch 53/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0267 - categorical_accuracy: 0.9666 - val_loss: 0.0341 - val_categorical_accuracy: 0.9344
Epoch 54/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0257 - categorical_accuracy: 0.9683 - val_loss: 0.0315 - val_categorical_accuracy: 0.9401
Epoch 55/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0248 - categorical_accuracy: 0.9693 - val_loss: 0.0299 - val_categorical_accuracy: 0.9446
Epoch 56/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0241 - categorical_accuracy: 0.9683 - val_loss: 0.0318 - val_categorical_accuracy: 0.9294
Epoch 57/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0234 - categorical_accuracy: 0.9690 - val_loss: 0.0299 - val_categorical_accuracy: 0.9402
Epoch 58/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0226 - categorical_accuracy: 0.9694 - val_loss: 0.0267 - val_categorical_accuracy: 0.9491
Epoch 59/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0220 - categorical_accuracy: 0.9701 - val_loss: 0.0328 - val_categorical_accuracy: 0.9141
Epoch 60/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0214 - categorical_accuracy: 0.9707 - val_loss: 0.0295 - val_categorical_accuracy: 0.9327
Epoch 61/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0209 - categorical_accuracy: 0.9703 - val_loss: 0.0357 - val_categorical_accuracy: 0.8937
Epoch 62/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0205 - categorical_accuracy: 0.9699 - val_loss: 0.0262 - val_categorical_accuracy: 0.9414
Epoch 63/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0199 - categorical_accuracy: 0.9717 - val_loss: 0.0259 - val_categorical_accuracy: 0.9474
Epoch 64/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0198 - categorical_accuracy: 0.9698 - val_loss: 0.0250 - val_categorical_accuracy: 0.9471
Epoch 65/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0194 - categorical_accuracy: 0.9701 - val_loss: 0.0248 - val_categorical_accuracy: 0.9456
Epoch 66/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0191 - categorical_accuracy: 0.9710 - val_loss: 0.0265 - val_categorical_accuracy: 0.9369
Epoch 67/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0186 - categorical_accuracy: 0.9716 - val_loss: 0.0379 - val_categorical_accuracy: 0.8907
Epoch 68/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0183 - categorical_accuracy: 0.9718 - val_loss: 0.0234 - val_categorical_accuracy: 0.9464
Epoch 69/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0184 - categorical_accuracy: 0.9704 - val_loss: 0.0291 - val_categorical_accuracy: 0.9185
Epoch 70/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0179 - categorical_accuracy: 0.9707 - val_loss: 0.0256 - val_categorical_accuracy: 0.9384
Epoch 71/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0176 - categorical_accuracy: 0.9721 - val_loss: 0.0221 - val_categorical_accuracy: 0.9518
Epoch 72/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0174 - categorical_accuracy: 0.9713 - val_loss: 0.0216 - val_categorical_accuracy: 0.9512
Epoch 73/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0171 - categorical_accuracy: 0.9717 - val_loss: 0.0223 - val_categorical_accuracy: 0.9495
Epoch 74/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0169 - categorical_accuracy: 0.9720 - val_loss: 0.0227 - val_categorical_accuracy: 0.9469
Epoch 75/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0170 - categorical_accuracy: 0.9712 - val_loss: 0.0231 - val_categorical_accuracy: 0.9437
Epoch 76/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0168 - categorical_accuracy: 0.9718 - val_loss: 0.0207 - val_categorical_accuracy: 0.9544
Epoch 77/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0163 - categorical_accuracy: 0.9738 - val_loss: 0.0245 - val_categorical_accuracy: 0.9298
Epoch 78/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0163 - categorical_accuracy: 0.9727 - val_loss: 0.0219 - val_categorical_accuracy: 0.9508
Epoch 79/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0162 - categorical_accuracy: 0.9727 - val_loss: 0.0212 - val_categorical_accuracy: 0.9510
Epoch 80/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0158 - categorical_accuracy: 0.9734 - val_loss: 0.0251 - val_categorical_accuracy: 0.9261
Epoch 81/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0158 - categorical_accuracy: 0.9727 - val_loss: 0.0226 - val_categorical_accuracy: 0.9452
Epoch 82/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0158 - categorical_accuracy: 0.9726 - val_loss: 0.0263 - val_categorical_accuracy: 0.9191

Epoch 00082: ReduceLROnPlateau reducing learning rate to 0.1.
Epoch 83/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0138 - categorical_accuracy: 0.9828 - val_loss: 0.0174 - val_categorical_accuracy: 0.9668
Epoch 84/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0131 - categorical_accuracy: 0.9868 - val_loss: 0.0175 - val_categorical_accuracy: 0.9674
Epoch 85/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0127 - categorical_accuracy: 0.9882 - val_loss: 0.0172 - val_categorical_accuracy: 0.9677
Epoch 86/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0124 - categorical_accuracy: 0.9888 - val_loss: 0.0171 - val_categorical_accuracy: 0.9672
Epoch 87/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0122 - categorical_accuracy: 0.9892 - val_loss: 0.0171 - val_categorical_accuracy: 0.9666
Epoch 88/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0119 - categorical_accuracy: 0.9899 - val_loss: 0.0169 - val_categorical_accuracy: 0.9680
Epoch 89/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0117 - categorical_accuracy: 0.9899 - val_loss: 0.0170 - val_categorical_accuracy: 0.9676
Epoch 90/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0115 - categorical_accuracy: 0.9905 - val_loss: 0.0167 - val_categorical_accuracy: 0.9681
Epoch 91/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0113 - categorical_accuracy: 0.9909 - val_loss: 0.0168 - val_categorical_accuracy: 0.9673
Epoch 92/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0111 - categorical_accuracy: 0.9912 - val_loss: 0.0165 - val_categorical_accuracy: 0.9693
Epoch 93/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0109 - categorical_accuracy: 0.9912 - val_loss: 0.0165 - val_categorical_accuracy: 0.9662
Epoch 94/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0107 - categorical_accuracy: 0.9921 - val_loss: 0.0165 - val_categorical_accuracy: 0.9675
Epoch 95/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0105 - categorical_accuracy: 0.9921 - val_loss: 0.0164 - val_categorical_accuracy: 0.9678
Epoch 96/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0103 - categorical_accuracy: 0.9925 - val_loss: 0.0163 - val_categorical_accuracy: 0.9684
Epoch 97/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0102 - categorical_accuracy: 0.9921 - val_loss: 0.0159 - val_categorical_accuracy: 0.9677
Epoch 98/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0101 - categorical_accuracy: 0.9923 - val_loss: 0.0162 - val_categorical_accuracy: 0.9672
Epoch 99/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0099 - categorical_accuracy: 0.9931 - val_loss: 0.0161 - val_categorical_accuracy: 0.9660
Epoch 100/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9933 - val_loss: 0.0158 - val_categorical_accuracy: 0.9681
Epoch 101/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0096 - categorical_accuracy: 0.9928 - val_loss: 0.0157 - val_categorical_accuracy: 0.9687
Epoch 102/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0094 - categorical_accuracy: 0.9936 - val_loss: 0.0158 - val_categorical_accuracy: 0.9682
Epoch 103/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0093 - categorical_accuracy: 0.9933 - val_loss: 0.0158 - val_categorical_accuracy: 0.9693
Epoch 104/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0092 - categorical_accuracy: 0.9936 - val_loss: 0.0157 - val_categorical_accuracy: 0.9674
Epoch 105/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0090 - categorical_accuracy: 0.9937 - val_loss: 0.0157 - val_categorical_accuracy: 0.9680
Epoch 106/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0089 - categorical_accuracy: 0.9942 - val_loss: 0.0157 - val_categorical_accuracy: 0.9680
Epoch 107/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0088 - categorical_accuracy: 0.9940 - val_loss: 0.0165 - val_categorical_accuracy: 0.9650

Epoch 00107: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 108/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0085 - categorical_accuracy: 0.9950 - val_loss: 0.0153 - val_categorical_accuracy: 0.9688
Epoch 109/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0085 - categorical_accuracy: 0.9954 - val_loss: 0.0151 - val_categorical_accuracy: 0.9695
Epoch 110/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0085 - categorical_accuracy: 0.9954 - val_loss: 0.0152 - val_categorical_accuracy: 0.9692
Epoch 111/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0084 - categorical_accuracy: 0.9957 - val_loss: 0.0152 - val_categorical_accuracy: 0.9693
Epoch 112/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0084 - categorical_accuracy: 0.9956 - val_loss: 0.0151 - val_categorical_accuracy: 0.9695
Epoch 113/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0083 - categorical_accuracy: 0.9959 - val_loss: 0.0152 - val_categorical_accuracy: 0.9687
Epoch 114/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0083 - categorical_accuracy: 0.9959 - val_loss: 0.0151 - val_categorical_accuracy: 0.9691
Epoch 115/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0082 - categorical_accuracy: 0.9959 - val_loss: 0.0151 - val_categorical_accuracy: 0.9692

Epoch 00115: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 116/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0082 - categorical_accuracy: 0.9960 - val_loss: 0.0151 - val_categorical_accuracy: 0.9692
Epoch 117/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0151 - val_categorical_accuracy: 0.9692
Epoch 118/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0082 - categorical_accuracy: 0.9960 - val_loss: 0.0151 - val_categorical_accuracy: 0.9697
Epoch 119/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0151 - val_categorical_accuracy: 0.9696
Epoch 120/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0082 - categorical_accuracy: 0.9959 - val_loss: 0.0151 - val_categorical_accuracy: 0.9695

Epoch 00120: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 121/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0151 - val_categorical_accuracy: 0.9695
Epoch 122/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 123/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0151 - val_categorical_accuracy: 0.9695
Epoch 124/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9961 - val_loss: 0.0151 - val_categorical_accuracy: 0.9695
Epoch 125/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0151 - val_categorical_accuracy: 0.9696

Epoch 00125: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 126/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0151 - val_categorical_accuracy: 0.9698
Epoch 127/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 128/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 129/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 130/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696

Epoch 00130: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 131/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9961 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 132/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 133/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 134/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 135/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9968 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696

Epoch 00135: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 136/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 137/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9961 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 138/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 139/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 140/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9960 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698

Epoch 00140: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 141/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 142/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 143/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 144/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 145/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695

Epoch 00145: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 146/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 147/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 148/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 149/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9967 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 150/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695

Epoch 00150: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 151/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 152/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698
Epoch 153/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9960 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 154/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 155/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696

Epoch 00155: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 156/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 157/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 158/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 159/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 160/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698

Epoch 00160: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 161/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 162/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698
Epoch 163/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 164/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 165/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698

Epoch 00165: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 166/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 167/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9968 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 168/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 169/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 170/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696

Epoch 00170: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 171/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 172/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9699
Epoch 173/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9961 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698
Epoch 174/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698
Epoch 175/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698

Epoch 00175: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 176/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 177/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 178/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9958 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 179/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 180/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 181/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 182/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696

Epoch 00182: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 183/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 184/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 185/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 186/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 187/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695

Epoch 00187: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 188/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 189/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9961 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 190/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9969 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 191/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 192/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698

Epoch 00192: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 193/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698
Epoch 194/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 195/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 196/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 197/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697

Epoch 00197: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 198/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9967 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 199/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 200/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 201/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9694
Epoch 202/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696

Epoch 00202: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 203/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 204/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9961 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698
Epoch 205/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 206/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698
Epoch 207/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9698

Epoch 00207: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 208/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 209/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 210/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 211/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 212/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695

Epoch 00212: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 213/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 214/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9968 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 215/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 216/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 217/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696

Epoch 00217: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 218/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 219/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 220/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 221/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 222/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9965 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695

Epoch 00222: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 223/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9969 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 224/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9969 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697
Epoch 225/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9962 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 226/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 227/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9967 - val_loss: 0.0150 - val_categorical_accuracy: 0.9697

Epoch 00227: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 228/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0081 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 229/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9964 - val_loss: 0.0150 - val_categorical_accuracy: 0.9695
Epoch 230/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9963 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 231/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9967 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696
Epoch 232/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0080 - categorical_accuracy: 0.9966 - val_loss: 0.0150 - val_categorical_accuracy: 0.9696

Epoch 00232: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 00232: early stopping
========= generating oof predictions 21:00:41 =========
========= generating test set predictions 21:00:43 =========
========= fitting 4 th model 21:01:07 =========
Epoch 1/65535
420/420 [==============================] - 28s 67ms/step - loss: 0.9746 - categorical_accuracy: 0.2228 - val_loss: 0.9206 - val_categorical_accuracy: 0.3440
Epoch 2/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.8543 - categorical_accuracy: 0.5849 - val_loss: 0.8041 - val_categorical_accuracy: 0.7140
Epoch 3/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.7678 - categorical_accuracy: 0.7446 - val_loss: 0.7267 - val_categorical_accuracy: 0.8221
Epoch 4/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.7002 - categorical_accuracy: 0.8081 - val_loss: 0.6667 - val_categorical_accuracy: 0.8433
Epoch 5/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.6412 - categorical_accuracy: 0.8446 - val_loss: 0.6112 - val_categorical_accuracy: 0.8743
Epoch 6/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.5889 - categorical_accuracy: 0.8655 - val_loss: 0.5622 - val_categorical_accuracy: 0.8887
Epoch 7/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.5413 - categorical_accuracy: 0.8814 - val_loss: 0.5184 - val_categorical_accuracy: 0.8903
Epoch 8/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.4979 - categorical_accuracy: 0.8926 - val_loss: 0.4766 - val_categorical_accuracy: 0.9065
Epoch 9/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.4584 - categorical_accuracy: 0.9028 - val_loss: 0.4392 - val_categorical_accuracy: 0.9093
Epoch 10/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.4220 - categorical_accuracy: 0.9118 - val_loss: 0.4049 - val_categorical_accuracy: 0.9173
Epoch 11/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.3890 - categorical_accuracy: 0.9180 - val_loss: 0.3739 - val_categorical_accuracy: 0.9200
Epoch 12/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.3585 - categorical_accuracy: 0.9233 - val_loss: 0.3440 - val_categorical_accuracy: 0.9261
Epoch 13/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.3306 - categorical_accuracy: 0.9275 - val_loss: 0.3200 - val_categorical_accuracy: 0.9146
Epoch 14/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.3050 - categorical_accuracy: 0.9320 - val_loss: 0.2936 - val_categorical_accuracy: 0.9306
Epoch 15/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2814 - categorical_accuracy: 0.9354 - val_loss: 0.2721 - val_categorical_accuracy: 0.9296
Epoch 16/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.2597 - categorical_accuracy: 0.9389 - val_loss: 0.2509 - val_categorical_accuracy: 0.9323
Epoch 17/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2399 - categorical_accuracy: 0.9405 - val_loss: 0.2321 - val_categorical_accuracy: 0.9341
Epoch 18/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2216 - categorical_accuracy: 0.9437 - val_loss: 0.2156 - val_categorical_accuracy: 0.9302
Epoch 19/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2048 - categorical_accuracy: 0.9458 - val_loss: 0.1988 - val_categorical_accuracy: 0.9369
Epoch 20/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.1895 - categorical_accuracy: 0.9479 - val_loss: 0.1833 - val_categorical_accuracy: 0.9449
Epoch 21/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1753 - categorical_accuracy: 0.9499 - val_loss: 0.1716 - val_categorical_accuracy: 0.9358
Epoch 22/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1623 - categorical_accuracy: 0.9519 - val_loss: 0.1614 - val_categorical_accuracy: 0.9257
Epoch 23/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1504 - categorical_accuracy: 0.9531 - val_loss: 0.1463 - val_categorical_accuracy: 0.9458
Epoch 24/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1394 - categorical_accuracy: 0.9550 - val_loss: 0.1370 - val_categorical_accuracy: 0.9425
Epoch 25/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1293 - categorical_accuracy: 0.9551 - val_loss: 0.1278 - val_categorical_accuracy: 0.9422
Epoch 26/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.1201 - categorical_accuracy: 0.9564 - val_loss: 0.1198 - val_categorical_accuracy: 0.9397
Epoch 27/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1116 - categorical_accuracy: 0.9584 - val_loss: 0.1138 - val_categorical_accuracy: 0.9267
Epoch 28/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1039 - categorical_accuracy: 0.9578 - val_loss: 0.1036 - val_categorical_accuracy: 0.9430
Epoch 29/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0967 - categorical_accuracy: 0.9597 - val_loss: 0.0968 - val_categorical_accuracy: 0.9411
Epoch 30/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0902 - categorical_accuracy: 0.9597 - val_loss: 0.0897 - val_categorical_accuracy: 0.9466
Epoch 31/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0841 - categorical_accuracy: 0.9605 - val_loss: 0.0843 - val_categorical_accuracy: 0.9469
Epoch 32/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0787 - categorical_accuracy: 0.9603 - val_loss: 0.0790 - val_categorical_accuracy: 0.9461
Epoch 33/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0736 - categorical_accuracy: 0.9612 - val_loss: 0.0754 - val_categorical_accuracy: 0.9421
Epoch 34/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0689 - categorical_accuracy: 0.9623 - val_loss: 0.0705 - val_categorical_accuracy: 0.9448
Epoch 35/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0645 - categorical_accuracy: 0.9634 - val_loss: 0.0664 - val_categorical_accuracy: 0.9448
Epoch 36/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0606 - categorical_accuracy: 0.9636 - val_loss: 0.0621 - val_categorical_accuracy: 0.9492
Epoch 37/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0569 - categorical_accuracy: 0.9653 - val_loss: 0.0608 - val_categorical_accuracy: 0.9365
Epoch 38/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0537 - categorical_accuracy: 0.9645 - val_loss: 0.0584 - val_categorical_accuracy: 0.9283
Epoch 39/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0508 - categorical_accuracy: 0.9632 - val_loss: 0.0521 - val_categorical_accuracy: 0.9513
Epoch 40/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0478 - categorical_accuracy: 0.9648 - val_loss: 0.0495 - val_categorical_accuracy: 0.9507
Epoch 41/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0452 - categorical_accuracy: 0.9654 - val_loss: 0.0505 - val_categorical_accuracy: 0.9340
Epoch 42/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0428 - categorical_accuracy: 0.9664 - val_loss: 0.0469 - val_categorical_accuracy: 0.9416
Epoch 43/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0407 - categorical_accuracy: 0.9656 - val_loss: 0.0431 - val_categorical_accuracy: 0.9489
Epoch 44/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0386 - categorical_accuracy: 0.9664 - val_loss: 0.0548 - val_categorical_accuracy: 0.8924
Epoch 45/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0370 - categorical_accuracy: 0.9656 - val_loss: 0.0412 - val_categorical_accuracy: 0.9422
Epoch 46/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0353 - categorical_accuracy: 0.9656 - val_loss: 0.0386 - val_categorical_accuracy: 0.9477
Epoch 47/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0336 - categorical_accuracy: 0.9671 - val_loss: 0.0437 - val_categorical_accuracy: 0.9208
Epoch 48/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0321 - categorical_accuracy: 0.9677 - val_loss: 0.0365 - val_categorical_accuracy: 0.9436
Epoch 49/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0308 - categorical_accuracy: 0.9672 - val_loss: 0.0338 - val_categorical_accuracy: 0.9498
Epoch 50/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0295 - categorical_accuracy: 0.9680 - val_loss: 0.0336 - val_categorical_accuracy: 0.9506
Epoch 51/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0285 - categorical_accuracy: 0.9677 - val_loss: 0.0318 - val_categorical_accuracy: 0.9498
Epoch 52/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0275 - categorical_accuracy: 0.9670 - val_loss: 0.0323 - val_categorical_accuracy: 0.9418
Epoch 53/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0266 - categorical_accuracy: 0.9673 - val_loss: 0.0296 - val_categorical_accuracy: 0.9493
Epoch 54/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0257 - categorical_accuracy: 0.9676 - val_loss: 0.0314 - val_categorical_accuracy: 0.9419
Epoch 55/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0249 - categorical_accuracy: 0.9673 - val_loss: 0.0298 - val_categorical_accuracy: 0.9393
Epoch 56/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0240 - categorical_accuracy: 0.9690 - val_loss: 0.0266 - val_categorical_accuracy: 0.9544
Epoch 57/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0233 - categorical_accuracy: 0.9690 - val_loss: 0.0307 - val_categorical_accuracy: 0.9310
Epoch 58/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0228 - categorical_accuracy: 0.9681 - val_loss: 0.0283 - val_categorical_accuracy: 0.9426
Epoch 59/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0220 - categorical_accuracy: 0.9691 - val_loss: 0.0279 - val_categorical_accuracy: 0.9429
Epoch 60/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0216 - categorical_accuracy: 0.9685 - val_loss: 0.0257 - val_categorical_accuracy: 0.9498
Epoch 61/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0212 - categorical_accuracy: 0.9689 - val_loss: 0.0265 - val_categorical_accuracy: 0.9406
Epoch 62/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0210 - categorical_accuracy: 0.9688 - val_loss: 0.0243 - val_categorical_accuracy: 0.9518
Epoch 63/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0201 - categorical_accuracy: 0.9703 - val_loss: 0.0245 - val_categorical_accuracy: 0.9479
Epoch 64/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0195 - categorical_accuracy: 0.9710 - val_loss: 0.0260 - val_categorical_accuracy: 0.9340
Epoch 65/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0194 - categorical_accuracy: 0.9707 - val_loss: 0.0252 - val_categorical_accuracy: 0.9419
Epoch 66/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0190 - categorical_accuracy: 0.9709 - val_loss: 0.0230 - val_categorical_accuracy: 0.9521
Epoch 67/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0188 - categorical_accuracy: 0.9705 - val_loss: 0.0227 - val_categorical_accuracy: 0.9527
Epoch 68/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0183 - categorical_accuracy: 0.9708 - val_loss: 0.0231 - val_categorical_accuracy: 0.9449
Epoch 69/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0183 - categorical_accuracy: 0.9693 - val_loss: 0.0246 - val_categorical_accuracy: 0.9407
Epoch 70/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0178 - categorical_accuracy: 0.9711 - val_loss: 0.0243 - val_categorical_accuracy: 0.9403
Epoch 71/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0177 - categorical_accuracy: 0.9700 - val_loss: 0.0215 - val_categorical_accuracy: 0.9520
Epoch 72/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0173 - categorical_accuracy: 0.9712 - val_loss: 0.0222 - val_categorical_accuracy: 0.9476
Epoch 73/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0171 - categorical_accuracy: 0.9716 - val_loss: 0.0238 - val_categorical_accuracy: 0.9413
Epoch 74/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0168 - categorical_accuracy: 0.9718 - val_loss: 0.0241 - val_categorical_accuracy: 0.9321
Epoch 75/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0167 - categorical_accuracy: 0.9723 - val_loss: 0.0218 - val_categorical_accuracy: 0.9475
Epoch 76/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0167 - categorical_accuracy: 0.9709 - val_loss: 0.0202 - val_categorical_accuracy: 0.9546
Epoch 77/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0164 - categorical_accuracy: 0.9720 - val_loss: 0.0226 - val_categorical_accuracy: 0.9407
Epoch 78/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0165 - categorical_accuracy: 0.9704 - val_loss: 0.0202 - val_categorical_accuracy: 0.9539
Epoch 79/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0162 - categorical_accuracy: 0.9723 - val_loss: 0.0271 - val_categorical_accuracy: 0.9297
Epoch 80/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0161 - categorical_accuracy: 0.9715 - val_loss: 0.0209 - val_categorical_accuracy: 0.9489
Epoch 81/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0159 - categorical_accuracy: 0.9725 - val_loss: 0.0249 - val_categorical_accuracy: 0.9340
Epoch 82/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0157 - categorical_accuracy: 0.9729 - val_loss: 0.0194 - val_categorical_accuracy: 0.9553
Epoch 83/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0158 - categorical_accuracy: 0.9725 - val_loss: 0.0215 - val_categorical_accuracy: 0.9445
Epoch 84/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0156 - categorical_accuracy: 0.9730 - val_loss: 0.0242 - val_categorical_accuracy: 0.9367
Epoch 85/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0155 - categorical_accuracy: 0.9732 - val_loss: 0.0205 - val_categorical_accuracy: 0.9483
Epoch 86/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0153 - categorical_accuracy: 0.9739 - val_loss: 0.0198 - val_categorical_accuracy: 0.9511
Epoch 87/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0151 - categorical_accuracy: 0.9733 - val_loss: 0.0226 - val_categorical_accuracy: 0.9391
Epoch 88/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0151 - categorical_accuracy: 0.9736 - val_loss: 0.0210 - val_categorical_accuracy: 0.9475

Epoch 00088: ReduceLROnPlateau reducing learning rate to 0.1.
Epoch 89/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0132 - categorical_accuracy: 0.9842 - val_loss: 0.0165 - val_categorical_accuracy: 0.9676
Epoch 90/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0125 - categorical_accuracy: 0.9870 - val_loss: 0.0162 - val_categorical_accuracy: 0.9685
Epoch 91/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0121 - categorical_accuracy: 0.9879 - val_loss: 0.0161 - val_categorical_accuracy: 0.9689
Epoch 92/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0118 - categorical_accuracy: 0.9893 - val_loss: 0.0159 - val_categorical_accuracy: 0.9700
Epoch 93/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0116 - categorical_accuracy: 0.9889 - val_loss: 0.0159 - val_categorical_accuracy: 0.9688
Epoch 94/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0114 - categorical_accuracy: 0.9900 - val_loss: 0.0156 - val_categorical_accuracy: 0.9700
Epoch 95/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0111 - categorical_accuracy: 0.9906 - val_loss: 0.0156 - val_categorical_accuracy: 0.9703
Epoch 96/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0110 - categorical_accuracy: 0.9904 - val_loss: 0.0155 - val_categorical_accuracy: 0.9706
Epoch 97/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0108 - categorical_accuracy: 0.9903 - val_loss: 0.0155 - val_categorical_accuracy: 0.9703
Epoch 98/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0106 - categorical_accuracy: 0.9909 - val_loss: 0.0156 - val_categorical_accuracy: 0.9685
Epoch 99/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0104 - categorical_accuracy: 0.9913 - val_loss: 0.0155 - val_categorical_accuracy: 0.9697
Epoch 100/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0102 - categorical_accuracy: 0.9917 - val_loss: 0.0152 - val_categorical_accuracy: 0.9715
Epoch 101/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0101 - categorical_accuracy: 0.9918 - val_loss: 0.0153 - val_categorical_accuracy: 0.9696
Epoch 102/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0099 - categorical_accuracy: 0.9921 - val_loss: 0.0149 - val_categorical_accuracy: 0.9698
Epoch 103/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9925 - val_loss: 0.0151 - val_categorical_accuracy: 0.9697
Epoch 104/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0096 - categorical_accuracy: 0.9928 - val_loss: 0.0149 - val_categorical_accuracy: 0.9712
Epoch 105/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0095 - categorical_accuracy: 0.9926 - val_loss: 0.0148 - val_categorical_accuracy: 0.9707
Epoch 106/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0093 - categorical_accuracy: 0.9930 - val_loss: 0.0147 - val_categorical_accuracy: 0.9708
Epoch 107/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0092 - categorical_accuracy: 0.9926 - val_loss: 0.0144 - val_categorical_accuracy: 0.9707
Epoch 108/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0091 - categorical_accuracy: 0.9932 - val_loss: 0.0143 - val_categorical_accuracy: 0.9718
Epoch 109/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0089 - categorical_accuracy: 0.9933 - val_loss: 0.0147 - val_categorical_accuracy: 0.9700
Epoch 110/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0088 - categorical_accuracy: 0.9932 - val_loss: 0.0143 - val_categorical_accuracy: 0.9714
Epoch 111/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0086 - categorical_accuracy: 0.9936 - val_loss: 0.0144 - val_categorical_accuracy: 0.9706
Epoch 112/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0085 - categorical_accuracy: 0.9937 - val_loss: 0.0143 - val_categorical_accuracy: 0.9700
Epoch 113/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0084 - categorical_accuracy: 0.9939 - val_loss: 0.0145 - val_categorical_accuracy: 0.9702
Epoch 114/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0082 - categorical_accuracy: 0.9943 - val_loss: 0.0143 - val_categorical_accuracy: 0.9702
Epoch 115/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0081 - categorical_accuracy: 0.9945 - val_loss: 0.0141 - val_categorical_accuracy: 0.9704
Epoch 116/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0080 - categorical_accuracy: 0.9944 - val_loss: 0.0139 - val_categorical_accuracy: 0.9715
Epoch 117/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0079 - categorical_accuracy: 0.9943 - val_loss: 0.0146 - val_categorical_accuracy: 0.9681
Epoch 118/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0078 - categorical_accuracy: 0.9944 - val_loss: 0.0139 - val_categorical_accuracy: 0.9704
Epoch 119/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0077 - categorical_accuracy: 0.9946 - val_loss: 0.0141 - val_categorical_accuracy: 0.9703
Epoch 120/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0076 - categorical_accuracy: 0.9948 - val_loss: 0.0139 - val_categorical_accuracy: 0.9695
Epoch 121/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0075 - categorical_accuracy: 0.9946 - val_loss: 0.0140 - val_categorical_accuracy: 0.9693
Epoch 122/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0074 - categorical_accuracy: 0.9948 - val_loss: 0.0139 - val_categorical_accuracy: 0.9690

Epoch 00122: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 123/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0072 - categorical_accuracy: 0.9956 - val_loss: 0.0137 - val_categorical_accuracy: 0.9705
Epoch 124/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0071 - categorical_accuracy: 0.9964 - val_loss: 0.0136 - val_categorical_accuracy: 0.9710
Epoch 125/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0071 - categorical_accuracy: 0.9963 - val_loss: 0.0135 - val_categorical_accuracy: 0.9714
Epoch 126/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0070 - categorical_accuracy: 0.9964 - val_loss: 0.0135 - val_categorical_accuracy: 0.9716
Epoch 127/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9968 - val_loss: 0.0134 - val_categorical_accuracy: 0.9717
Epoch 128/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0070 - categorical_accuracy: 0.9967 - val_loss: 0.0135 - val_categorical_accuracy: 0.9713
Epoch 129/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0069 - categorical_accuracy: 0.9969 - val_loss: 0.0134 - val_categorical_accuracy: 0.9712
Epoch 130/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0068 - categorical_accuracy: 0.9972 - val_loss: 0.0134 - val_categorical_accuracy: 0.9717
Epoch 131/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0069 - categorical_accuracy: 0.9969 - val_loss: 0.0134 - val_categorical_accuracy: 0.9718
Epoch 132/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0068 - categorical_accuracy: 0.9970 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 133/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0068 - categorical_accuracy: 0.9973 - val_loss: 0.0134 - val_categorical_accuracy: 0.9718

Epoch 00133: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 134/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0068 - categorical_accuracy: 0.9971 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 135/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9973 - val_loss: 0.0134 - val_categorical_accuracy: 0.9724
Epoch 136/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0068 - categorical_accuracy: 0.9973 - val_loss: 0.0134 - val_categorical_accuracy: 0.9722
Epoch 137/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9974 - val_loss: 0.0134 - val_categorical_accuracy: 0.9721
Epoch 138/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9973 - val_loss: 0.0134 - val_categorical_accuracy: 0.9721

Epoch 00138: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 139/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0067 - categorical_accuracy: 0.9978 - val_loss: 0.0134 - val_categorical_accuracy: 0.9722
Epoch 140/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9978 - val_loss: 0.0134 - val_categorical_accuracy: 0.9722
Epoch 141/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9973 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 142/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0067 - categorical_accuracy: 0.9973 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 143/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9978 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723

Epoch 00143: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 144/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0067 - categorical_accuracy: 0.9973 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 145/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9977 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 146/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9972 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 147/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0067 - categorical_accuracy: 0.9978 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 148/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0067 - categorical_accuracy: 0.9976 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723

Epoch 00148: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 149/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9978 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 150/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9974 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 151/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9976 - val_loss: 0.0134 - val_categorical_accuracy: 0.9724
Epoch 152/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0067 - categorical_accuracy: 0.9973 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723
Epoch 153/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0067 - categorical_accuracy: 0.9974 - val_loss: 0.0134 - val_categorical_accuracy: 0.9723

Epoch 00153: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 00153: early stopping
========= generating oof predictions 22:08:51 =========
========= generating test set predictions 22:08:53 =========
========= fitting 5 th model 22:09:17 =========
Epoch 1/65535
420/420 [==============================] - 28s 67ms/step - loss: 0.9723 - categorical_accuracy: 0.2384 - val_loss: 0.9072 - val_categorical_accuracy: 0.4545
Epoch 2/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.8505 - categorical_accuracy: 0.6125 - val_loss: 0.7990 - val_categorical_accuracy: 0.7435
Epoch 3/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.7655 - categorical_accuracy: 0.7586 - val_loss: 0.7276 - val_categorical_accuracy: 0.8178
Epoch 4/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.6988 - categorical_accuracy: 0.8162 - val_loss: 0.6658 - val_categorical_accuracy: 0.8550
Epoch 5/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.6408 - categorical_accuracy: 0.8468 - val_loss: 0.6101 - val_categorical_accuracy: 0.8849
Epoch 6/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.5887 - categorical_accuracy: 0.8674 - val_loss: 0.5653 - val_categorical_accuracy: 0.8687
Epoch 7/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.5412 - categorical_accuracy: 0.8832 - val_loss: 0.5173 - val_categorical_accuracy: 0.9007
Epoch 8/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.4980 - categorical_accuracy: 0.8947 - val_loss: 0.4764 - val_categorical_accuracy: 0.9087
Epoch 9/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.4585 - categorical_accuracy: 0.9031 - val_loss: 0.4391 - val_categorical_accuracy: 0.9111
Epoch 10/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.4224 - categorical_accuracy: 0.9103 - val_loss: 0.4048 - val_categorical_accuracy: 0.9204
Epoch 11/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.3892 - categorical_accuracy: 0.9171 - val_loss: 0.3737 - val_categorical_accuracy: 0.9211
Epoch 12/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.3586 - categorical_accuracy: 0.9218 - val_loss: 0.3455 - val_categorical_accuracy: 0.9206
Epoch 13/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.3306 - categorical_accuracy: 0.9278 - val_loss: 0.3193 - val_categorical_accuracy: 0.9181
Epoch 14/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.3050 - categorical_accuracy: 0.9318 - val_loss: 0.2935 - val_categorical_accuracy: 0.9321
Epoch 15/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2814 - categorical_accuracy: 0.9358 - val_loss: 0.2718 - val_categorical_accuracy: 0.9325
Epoch 16/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.2598 - categorical_accuracy: 0.9385 - val_loss: 0.2524 - val_categorical_accuracy: 0.9247
Epoch 17/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2398 - categorical_accuracy: 0.9417 - val_loss: 0.2333 - val_categorical_accuracy: 0.9269
Epoch 18/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2217 - categorical_accuracy: 0.9434 - val_loss: 0.2168 - val_categorical_accuracy: 0.9254
Epoch 19/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.2048 - categorical_accuracy: 0.9455 - val_loss: 0.2005 - val_categorical_accuracy: 0.9300
Epoch 20/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1896 - categorical_accuracy: 0.9484 - val_loss: 0.1855 - val_categorical_accuracy: 0.9339
Epoch 21/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1755 - categorical_accuracy: 0.9492 - val_loss: 0.1707 - val_categorical_accuracy: 0.9407
Epoch 22/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1623 - categorical_accuracy: 0.9522 - val_loss: 0.1587 - val_categorical_accuracy: 0.9400
Epoch 23/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1504 - categorical_accuracy: 0.9532 - val_loss: 0.1479 - val_categorical_accuracy: 0.9380
Epoch 24/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1396 - categorical_accuracy: 0.9536 - val_loss: 0.1390 - val_categorical_accuracy: 0.9314
Epoch 25/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1294 - categorical_accuracy: 0.9561 - val_loss: 0.1276 - val_categorical_accuracy: 0.9415
Epoch 26/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1201 - categorical_accuracy: 0.9560 - val_loss: 0.1192 - val_categorical_accuracy: 0.9419
Epoch 27/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.1117 - categorical_accuracy: 0.9577 - val_loss: 0.1121 - val_categorical_accuracy: 0.9361
Epoch 28/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.1039 - categorical_accuracy: 0.9586 - val_loss: 0.1046 - val_categorical_accuracy: 0.9378
Epoch 29/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0968 - categorical_accuracy: 0.9592 - val_loss: 0.0979 - val_categorical_accuracy: 0.9401
Epoch 30/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0900 - categorical_accuracy: 0.9610 - val_loss: 0.0949 - val_categorical_accuracy: 0.9222
Epoch 31/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0839 - categorical_accuracy: 0.9629 - val_loss: 0.0858 - val_categorical_accuracy: 0.9418
Epoch 32/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0786 - categorical_accuracy: 0.9612 - val_loss: 0.0800 - val_categorical_accuracy: 0.9439
Epoch 33/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0733 - categorical_accuracy: 0.9634 - val_loss: 0.0774 - val_categorical_accuracy: 0.9313
Epoch 34/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0687 - categorical_accuracy: 0.9626 - val_loss: 0.0711 - val_categorical_accuracy: 0.9418
Epoch 35/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0644 - categorical_accuracy: 0.9637 - val_loss: 0.0667 - val_categorical_accuracy: 0.9456
Epoch 36/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0605 - categorical_accuracy: 0.9646 - val_loss: 0.0631 - val_categorical_accuracy: 0.9431
Epoch 37/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0570 - categorical_accuracy: 0.9630 - val_loss: 0.0625 - val_categorical_accuracy: 0.9317
Epoch 38/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0536 - categorical_accuracy: 0.9645 - val_loss: 0.0567 - val_categorical_accuracy: 0.9433
Epoch 39/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0505 - categorical_accuracy: 0.9653 - val_loss: 0.0543 - val_categorical_accuracy: 0.9409
Epoch 40/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0480 - categorical_accuracy: 0.9639 - val_loss: 0.0509 - val_categorical_accuracy: 0.9445
Epoch 41/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0453 - categorical_accuracy: 0.9647 - val_loss: 0.0527 - val_categorical_accuracy: 0.9239
Epoch 42/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0426 - categorical_accuracy: 0.9672 - val_loss: 0.0500 - val_categorical_accuracy: 0.9262
Epoch 43/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0406 - categorical_accuracy: 0.9663 - val_loss: 0.0446 - val_categorical_accuracy: 0.9406
Epoch 44/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0386 - categorical_accuracy: 0.9667 - val_loss: 0.0429 - val_categorical_accuracy: 0.9421
Epoch 45/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0368 - categorical_accuracy: 0.9670 - val_loss: 0.0408 - val_categorical_accuracy: 0.9435
Epoch 46/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0351 - categorical_accuracy: 0.9670 - val_loss: 0.0401 - val_categorical_accuracy: 0.9365
Epoch 47/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0334 - categorical_accuracy: 0.9678 - val_loss: 0.0383 - val_categorical_accuracy: 0.9408
Epoch 48/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0320 - categorical_accuracy: 0.9674 - val_loss: 0.0381 - val_categorical_accuracy: 0.9337
Epoch 49/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0309 - categorical_accuracy: 0.9663 - val_loss: 0.0373 - val_categorical_accuracy: 0.9378
Epoch 50/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0296 - categorical_accuracy: 0.9674 - val_loss: 0.0342 - val_categorical_accuracy: 0.9424
Epoch 51/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0284 - categorical_accuracy: 0.9692 - val_loss: 0.0363 - val_categorical_accuracy: 0.9304
Epoch 52/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0274 - categorical_accuracy: 0.9685 - val_loss: 0.0376 - val_categorical_accuracy: 0.9207
Epoch 53/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0262 - categorical_accuracy: 0.9691 - val_loss: 0.0317 - val_categorical_accuracy: 0.9402
Epoch 54/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0258 - categorical_accuracy: 0.9672 - val_loss: 0.0350 - val_categorical_accuracy: 0.9222
Epoch 55/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0247 - categorical_accuracy: 0.9690 - val_loss: 0.0313 - val_categorical_accuracy: 0.9331
Epoch 56/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0237 - categorical_accuracy: 0.9701 - val_loss: 0.0301 - val_categorical_accuracy: 0.9407
Epoch 57/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0236 - categorical_accuracy: 0.9675 - val_loss: 0.0289 - val_categorical_accuracy: 0.9438
Epoch 58/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0226 - categorical_accuracy: 0.9690 - val_loss: 0.0276 - val_categorical_accuracy: 0.9448
Epoch 59/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0219 - categorical_accuracy: 0.9701 - val_loss: 0.0263 - val_categorical_accuracy: 0.9486
Epoch 60/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0215 - categorical_accuracy: 0.9692 - val_loss: 0.0286 - val_categorical_accuracy: 0.9389
Epoch 61/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0209 - categorical_accuracy: 0.9703 - val_loss: 0.0273 - val_categorical_accuracy: 0.9439
Epoch 62/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0207 - categorical_accuracy: 0.9699 - val_loss: 0.0272 - val_categorical_accuracy: 0.9390
Epoch 63/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0200 - categorical_accuracy: 0.9708 - val_loss: 0.0271 - val_categorical_accuracy: 0.9375
Epoch 64/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0198 - categorical_accuracy: 0.9697 - val_loss: 0.0256 - val_categorical_accuracy: 0.9435
Epoch 65/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0193 - categorical_accuracy: 0.9715 - val_loss: 0.0255 - val_categorical_accuracy: 0.9408
Epoch 66/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0190 - categorical_accuracy: 0.9711 - val_loss: 0.0263 - val_categorical_accuracy: 0.9348
Epoch 67/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0187 - categorical_accuracy: 0.9715 - val_loss: 0.0242 - val_categorical_accuracy: 0.9439
Epoch 68/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0183 - categorical_accuracy: 0.9717 - val_loss: 0.0260 - val_categorical_accuracy: 0.9328
Epoch 69/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0184 - categorical_accuracy: 0.9693 - val_loss: 0.0233 - val_categorical_accuracy: 0.9481
Epoch 70/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0179 - categorical_accuracy: 0.9709 - val_loss: 0.0261 - val_categorical_accuracy: 0.9310
Epoch 71/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0178 - categorical_accuracy: 0.9715 - val_loss: 0.0254 - val_categorical_accuracy: 0.9339
Epoch 72/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0172 - categorical_accuracy: 0.9730 - val_loss: 0.0223 - val_categorical_accuracy: 0.9473
Epoch 73/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0172 - categorical_accuracy: 0.9719 - val_loss: 0.0268 - val_categorical_accuracy: 0.9271
Epoch 74/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0173 - categorical_accuracy: 0.9713 - val_loss: 0.0278 - val_categorical_accuracy: 0.9215
Epoch 75/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0167 - categorical_accuracy: 0.9726 - val_loss: 0.0265 - val_categorical_accuracy: 0.9288
Epoch 76/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0167 - categorical_accuracy: 0.9721 - val_loss: 0.0228 - val_categorical_accuracy: 0.9477
Epoch 77/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0166 - categorical_accuracy: 0.9723 - val_loss: 0.0239 - val_categorical_accuracy: 0.9390
Epoch 78/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0161 - categorical_accuracy: 0.9742 - val_loss: 0.0206 - val_categorical_accuracy: 0.9533
Epoch 79/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0164 - categorical_accuracy: 0.9713 - val_loss: 0.0208 - val_categorical_accuracy: 0.9518
Epoch 80/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0162 - categorical_accuracy: 0.9717 - val_loss: 0.0203 - val_categorical_accuracy: 0.9540
Epoch 81/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0157 - categorical_accuracy: 0.9740 - val_loss: 0.0237 - val_categorical_accuracy: 0.9401
Epoch 82/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0158 - categorical_accuracy: 0.9727 - val_loss: 0.0219 - val_categorical_accuracy: 0.9492
Epoch 83/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0159 - categorical_accuracy: 0.9724 - val_loss: 0.0212 - val_categorical_accuracy: 0.9471
Epoch 84/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0156 - categorical_accuracy: 0.9737 - val_loss: 0.0250 - val_categorical_accuracy: 0.9355
Epoch 85/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0155 - categorical_accuracy: 0.9745 - val_loss: 0.0237 - val_categorical_accuracy: 0.9405
Epoch 86/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0156 - categorical_accuracy: 0.9727 - val_loss: 0.0213 - val_categorical_accuracy: 0.9477

Epoch 00086: ReduceLROnPlateau reducing learning rate to 0.1.
Epoch 87/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0136 - categorical_accuracy: 0.9841 - val_loss: 0.0176 - val_categorical_accuracy: 0.9661
Epoch 88/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0129 - categorical_accuracy: 0.9870 - val_loss: 0.0176 - val_categorical_accuracy: 0.9645
Epoch 89/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0124 - categorical_accuracy: 0.9888 - val_loss: 0.0174 - val_categorical_accuracy: 0.9664
Epoch 90/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0122 - categorical_accuracy: 0.9891 - val_loss: 0.0170 - val_categorical_accuracy: 0.9674
Epoch 91/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0118 - categorical_accuracy: 0.9905 - val_loss: 0.0170 - val_categorical_accuracy: 0.9675
Epoch 92/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0116 - categorical_accuracy: 0.9909 - val_loss: 0.0170 - val_categorical_accuracy: 0.9679
Epoch 93/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0114 - categorical_accuracy: 0.9911 - val_loss: 0.0172 - val_categorical_accuracy: 0.9664
Epoch 94/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0113 - categorical_accuracy: 0.9913 - val_loss: 0.0169 - val_categorical_accuracy: 0.9679
Epoch 95/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0110 - categorical_accuracy: 0.9919 - val_loss: 0.0167 - val_categorical_accuracy: 0.9673
Epoch 96/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0108 - categorical_accuracy: 0.9921 - val_loss: 0.0167 - val_categorical_accuracy: 0.9668
Epoch 97/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0107 - categorical_accuracy: 0.9920 - val_loss: 0.0171 - val_categorical_accuracy: 0.9670
Epoch 98/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0104 - categorical_accuracy: 0.9927 - val_loss: 0.0168 - val_categorical_accuracy: 0.9673
Epoch 99/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0103 - categorical_accuracy: 0.9928 - val_loss: 0.0166 - val_categorical_accuracy: 0.9670
Epoch 100/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0101 - categorical_accuracy: 0.9928 - val_loss: 0.0164 - val_categorical_accuracy: 0.9685
Epoch 101/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0099 - categorical_accuracy: 0.9933 - val_loss: 0.0166 - val_categorical_accuracy: 0.9671
Epoch 102/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0098 - categorical_accuracy: 0.9935 - val_loss: 0.0167 - val_categorical_accuracy: 0.9669
Epoch 103/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0097 - categorical_accuracy: 0.9932 - val_loss: 0.0163 - val_categorical_accuracy: 0.9675
Epoch 104/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0094 - categorical_accuracy: 0.9940 - val_loss: 0.0161 - val_categorical_accuracy: 0.9679
Epoch 105/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0094 - categorical_accuracy: 0.9938 - val_loss: 0.0162 - val_categorical_accuracy: 0.9673
Epoch 106/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0092 - categorical_accuracy: 0.9937 - val_loss: 0.0163 - val_categorical_accuracy: 0.9665
Epoch 107/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0091 - categorical_accuracy: 0.9942 - val_loss: 0.0160 - val_categorical_accuracy: 0.9668
Epoch 108/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0089 - categorical_accuracy: 0.9945 - val_loss: 0.0159 - val_categorical_accuracy: 0.9674
Epoch 109/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0088 - categorical_accuracy: 0.9945 - val_loss: 0.0158 - val_categorical_accuracy: 0.9662
Epoch 110/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0087 - categorical_accuracy: 0.9949 - val_loss: 0.0160 - val_categorical_accuracy: 0.9679
Epoch 111/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0085 - categorical_accuracy: 0.9952 - val_loss: 0.0158 - val_categorical_accuracy: 0.9676
Epoch 112/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0084 - categorical_accuracy: 0.9947 - val_loss: 0.0156 - val_categorical_accuracy: 0.9673
Epoch 113/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0083 - categorical_accuracy: 0.9948 - val_loss: 0.0154 - val_categorical_accuracy: 0.9658
Epoch 114/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0082 - categorical_accuracy: 0.9950 - val_loss: 0.0157 - val_categorical_accuracy: 0.9671
Epoch 115/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0080 - categorical_accuracy: 0.9955 - val_loss: 0.0157 - val_categorical_accuracy: 0.9671
Epoch 116/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0080 - categorical_accuracy: 0.9948 - val_loss: 0.0154 - val_categorical_accuracy: 0.9663
Epoch 117/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0078 - categorical_accuracy: 0.9955 - val_loss: 0.0156 - val_categorical_accuracy: 0.9656
Epoch 118/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0077 - categorical_accuracy: 0.9955 - val_loss: 0.0158 - val_categorical_accuracy: 0.9659
Epoch 119/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0076 - categorical_accuracy: 0.9956 - val_loss: 0.0164 - val_categorical_accuracy: 0.9641

Epoch 00119: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 120/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0075 - categorical_accuracy: 0.9964 - val_loss: 0.0151 - val_categorical_accuracy: 0.9679
Epoch 121/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0073 - categorical_accuracy: 0.9969 - val_loss: 0.0151 - val_categorical_accuracy: 0.9682
Epoch 122/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0073 - categorical_accuracy: 0.9972 - val_loss: 0.0152 - val_categorical_accuracy: 0.9685
Epoch 123/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0072 - categorical_accuracy: 0.9974 - val_loss: 0.0152 - val_categorical_accuracy: 0.9684
Epoch 124/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0072 - categorical_accuracy: 0.9976 - val_loss: 0.0152 - val_categorical_accuracy: 0.9678
Epoch 125/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0071 - categorical_accuracy: 0.9976 - val_loss: 0.0151 - val_categorical_accuracy: 0.9680
Epoch 126/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0072 - categorical_accuracy: 0.9973 - val_loss: 0.0150 - val_categorical_accuracy: 0.9689

Epoch 00126: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 127/65535
420/420 [==============================] - 27s 63ms/step - loss: 0.0071 - categorical_accuracy: 0.9974 - val_loss: 0.0150 - val_categorical_accuracy: 0.9683
Epoch 128/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0071 - categorical_accuracy: 0.9973 - val_loss: 0.0150 - val_categorical_accuracy: 0.9687
Epoch 129/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0071 - categorical_accuracy: 0.9975 - val_loss: 0.0150 - val_categorical_accuracy: 0.9682
Epoch 130/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0071 - categorical_accuracy: 0.9977 - val_loss: 0.0150 - val_categorical_accuracy: 0.9682
Epoch 131/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0071 - categorical_accuracy: 0.9978 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 132/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0070 - categorical_accuracy: 0.9978 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 133/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0071 - categorical_accuracy: 0.9976 - val_loss: 0.0150 - val_categorical_accuracy: 0.9680

Epoch 00133: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 134/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0070 - categorical_accuracy: 0.9980 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 135/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0071 - categorical_accuracy: 0.9976 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 136/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9980 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 137/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0071 - categorical_accuracy: 0.9976 - val_loss: 0.0150 - val_categorical_accuracy: 0.9678
Epoch 138/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0070 - categorical_accuracy: 0.9978 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679

Epoch 00138: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 139/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0071 - categorical_accuracy: 0.9974 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 140/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9977 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 141/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0070 - categorical_accuracy: 0.9977 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 142/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9978 - val_loss: 0.0150 - val_categorical_accuracy: 0.9676
Epoch 143/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9978 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679

Epoch 00143: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 144/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0071 - categorical_accuracy: 0.9976 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 145/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9977 - val_loss: 0.0150 - val_categorical_accuracy: 0.9679
Epoch 00145: early stopping
========= generating oof predictions 23:13:28 =========
========= generating test set predictions 23:13:30 =========
train loss avg 0.007629463091956926 -- std 0.0010884234706117488, val loss avg 0.014836618360502147 -- std 0.0008864517737468359
train acc avg 0.9968264968255971 -- std 0.0012132119625399525, val acc avg 0.9694584073439995 -- std 0.001515340419481311
mean nb epochs 173.2
dump oof predicted probs
dump test set predicted probs
