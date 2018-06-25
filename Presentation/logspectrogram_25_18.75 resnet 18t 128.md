ren (master *+) python $ python train.py logspectrogram_25_18.75 resnet 18t 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (55, 201) ===========
------------- SUMMARY OF MODEL -------------
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 55, 201)      0
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 55, 201, 1)   0           input_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 27, 95, 32)   1184        reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 27, 95, 32)   128         conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 27, 95, 32)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 27, 95, 32)   0           activation_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 27, 95, 32)   0           dropout_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 27, 95, 32)   10272       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 27, 95, 32)   128         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 27, 95, 32)   0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 27, 95, 32)   0           activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 27, 95, 32)   10272       dropout_2[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 27, 95, 64)   0           max_pooling2d_1[0][0]
                                                                 conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 27, 95, 64)   256         concatenate_1[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 27, 95, 64)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 27, 95, 64)   0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 27, 95, 32)   20512       dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 27, 95, 32)   128         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 27, 95, 32)   0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 27, 95, 32)   0           activation_4[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 27, 95, 32)   2080        concatenate_1[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 27, 95, 32)   10272       dropout_4[0][0]
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 27, 95, 64)   0           conv2d_6[0][0]
                                                                 conv2d_5[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 27, 95, 64)   256         concatenate_2[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 27, 95, 64)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 27, 95, 64)   0           activation_5[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 14, 48, 64)   41024       dropout_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 14, 48, 64)   256         conv2d_7[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 14, 48, 64)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 14, 48, 64)   0           activation_6[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 7, 24, 64)    4160        concatenate_2[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 7, 24, 64)    41024       dropout_6[0][0]
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 7, 24, 128)   0           conv2d_9[0][0]
                                                                 conv2d_8[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 7, 24, 128)   512         concatenate_3[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 7, 24, 128)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 7, 24, 128)   0           activation_7[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 7, 24, 64)    81984       dropout_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 7, 24, 64)    256         conv2d_10[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 7, 24, 64)    0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 7, 24, 64)    0           activation_8[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 7, 24, 64)    8256        concatenate_3[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 7, 24, 64)    41024       dropout_8[0][0]
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 7, 24, 128)   0           conv2d_12[0][0]
                                                                 conv2d_11[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 7, 24, 128)   512         concatenate_4[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 7, 24, 128)   0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 7, 24, 128)   0           activation_9[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 4, 12, 128)   163968      dropout_9[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 4, 12, 128)   512         conv2d_13[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 4, 12, 128)   0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, 4, 12, 128)   0           activation_10[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 2, 6, 128)    16512       concatenate_4[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 2, 6, 128)    163968      dropout_10[0][0]
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 2, 6, 256)    0           conv2d_15[0][0]
                                                                 conv2d_14[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 2, 6, 256)    1024        concatenate_5[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 2, 6, 256)    0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
dropout_11 (Dropout)            (None, 2, 6, 256)    0           activation_11[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 2, 6, 128)    327808      dropout_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 2, 6, 128)    512         conv2d_16[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 2, 6, 128)    0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
dropout_12 (Dropout)            (None, 2, 6, 128)    0           activation_12[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 2, 6, 128)    32896       concatenate_5[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 2, 6, 128)    163968      dropout_12[0][0]
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 2, 6, 256)    0           conv2d_18[0][0]
                                                                 conv2d_17[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 2, 6, 256)    1024        concatenate_6[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 2, 6, 256)    0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
dropout_13 (Dropout)            (None, 2, 6, 256)    0           activation_13[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 1, 3, 256)    655616      dropout_13[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 1, 3, 256)    1024        conv2d_19[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 1, 3, 256)    0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
dropout_14 (Dropout)            (None, 1, 3, 256)    0           activation_14[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 1, 2, 256)    65792       concatenate_6[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 1, 2, 256)    655616      dropout_14[0][0]
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 1, 2, 512)    0           conv2d_21[0][0]
                                                                 conv2d_20[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 1, 2, 512)    2048        concatenate_7[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 1, 2, 512)    0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
dropout_15 (Dropout)            (None, 1, 2, 512)    0           activation_15[0][0]
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 1, 2, 256)    1310976     dropout_15[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 1, 2, 256)    1024        conv2d_22[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 1, 2, 256)    0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
dropout_16 (Dropout)            (None, 1, 2, 256)    0           activation_16[0][0]
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 1, 2, 256)    131328      concatenate_7[0][0]
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 1, 2, 256)    655616      dropout_16[0][0]
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 1, 2, 512)    0           conv2d_24[0][0]
                                                                 conv2d_23[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 1, 2, 512)    2048        concatenate_8[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 1, 2, 512)    0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 1, 1, 512)    0           activation_17[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 512)          0           average_pooling2d_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 31)           15903       flatten_1[0][0]
==================================================================================================
Total params: 4,643,679
Trainable params: 4,637,855
Non-trainable params: 5,824
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 07:03:41 =========
Epoch 1/65535
420/420 [==============================] - 57s 135ms/step - loss: 0.5363 - categorical_accuracy: 0.1751 - val_loss: 0.4811 - val_categorical_accuracy: 0.3295
Epoch 2/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.4263 - categorical_accuracy: 0.5007 - val_loss: 0.3859 - val_categorical_accuracy: 0.5760
Epoch 3/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.3469 - categorical_accuracy: 0.6816 - val_loss: 0.3182 - val_categorical_accuracy: 0.7129
Epoch 4/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.2883 - categorical_accuracy: 0.7769 - val_loss: 0.2670 - val_categorical_accuracy: 0.7864
Epoch 5/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.2429 - categorical_accuracy: 0.8260 - val_loss: 0.2197 - val_categorical_accuracy: 0.8663
Epoch 6/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.2064 - categorical_accuracy: 0.8581 - val_loss: 0.1897 - val_categorical_accuracy: 0.8671
Epoch 7/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.1762 - categorical_accuracy: 0.8801 - val_loss: 0.1600 - val_categorical_accuracy: 0.9030
Epoch 8/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.1515 - categorical_accuracy: 0.8933 - val_loss: 0.1406 - val_categorical_accuracy: 0.8941
Epoch 9/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.1308 - categorical_accuracy: 0.9034 - val_loss: 0.1192 - val_categorical_accuracy: 0.9214
Epoch 10/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.1132 - categorical_accuracy: 0.9139 - val_loss: 0.1059 - val_categorical_accuracy: 0.9140
Epoch 11/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0986 - categorical_accuracy: 0.9196 - val_loss: 0.0939 - val_categorical_accuracy: 0.9100
Epoch 12/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0865 - categorical_accuracy: 0.9234 - val_loss: 0.0810 - val_categorical_accuracy: 0.9260
Epoch 13/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0761 - categorical_accuracy: 0.9287 - val_loss: 0.0739 - val_categorical_accuracy: 0.9146
Epoch 14/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0674 - categorical_accuracy: 0.9314 - val_loss: 0.0672 - val_categorical_accuracy: 0.9148
Epoch 15/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0602 - categorical_accuracy: 0.9338 - val_loss: 0.0580 - val_categorical_accuracy: 0.9315
Epoch 16/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0540 - categorical_accuracy: 0.9373 - val_loss: 0.0527 - val_categorical_accuracy: 0.9321
Epoch 17/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0492 - categorical_accuracy: 0.9372 - val_loss: 0.0512 - val_categorical_accuracy: 0.9165
Epoch 18/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0448 - categorical_accuracy: 0.9399 - val_loss: 0.0439 - val_categorical_accuracy: 0.9378
Epoch 19/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0412 - categorical_accuracy: 0.9422 - val_loss: 0.0410 - val_categorical_accuracy: 0.9348
Epoch 20/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0380 - categorical_accuracy: 0.9434 - val_loss: 0.0391 - val_categorical_accuracy: 0.9321
Epoch 21/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0354 - categorical_accuracy: 0.9445 - val_loss: 0.0451 - val_categorical_accuracy: 0.8950
Epoch 22/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0337 - categorical_accuracy: 0.9425 - val_loss: 0.0486 - val_categorical_accuracy: 0.8706
Epoch 23/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0314 - categorical_accuracy: 0.9469 - val_loss: 0.0343 - val_categorical_accuracy: 0.9330
Epoch 24/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0300 - categorical_accuracy: 0.9464 - val_loss: 0.0319 - val_categorical_accuracy: 0.9372
Epoch 25/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0286 - categorical_accuracy: 0.9471 - val_loss: 0.0309 - val_categorical_accuracy: 0.9361
Epoch 26/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0275 - categorical_accuracy: 0.9486 - val_loss: 0.0287 - val_categorical_accuracy: 0.9406
Epoch 27/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0266 - categorical_accuracy: 0.9492 - val_loss: 0.0272 - val_categorical_accuracy: 0.9443
Epoch 28/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0254 - categorical_accuracy: 0.9516 - val_loss: 0.0275 - val_categorical_accuracy: 0.9396
Epoch 29/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0249 - categorical_accuracy: 0.9514 - val_loss: 0.0276 - val_categorical_accuracy: 0.9371
Epoch 30/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0244 - categorical_accuracy: 0.9508 - val_loss: 0.0325 - val_categorical_accuracy: 0.9052
Epoch 31/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0238 - categorical_accuracy: 0.9503 - val_loss: 0.0255 - val_categorical_accuracy: 0.9443
Epoch 32/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0234 - categorical_accuracy: 0.9513 - val_loss: 0.0253 - val_categorical_accuracy: 0.9438
Epoch 33/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0229 - categorical_accuracy: 0.9527 - val_loss: 0.0262 - val_categorical_accuracy: 0.9360
Epoch 34/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0224 - categorical_accuracy: 0.9545 - val_loss: 0.0257 - val_categorical_accuracy: 0.9378
Epoch 35/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0224 - categorical_accuracy: 0.9519 - val_loss: 0.0244 - val_categorical_accuracy: 0.9437
Epoch 36/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0220 - categorical_accuracy: 0.9532 - val_loss: 0.0263 - val_categorical_accuracy: 0.9340
Epoch 37/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0216 - categorical_accuracy: 0.9544 - val_loss: 0.0248 - val_categorical_accuracy: 0.9396
Epoch 38/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0216 - categorical_accuracy: 0.9539 - val_loss: 0.0243 - val_categorical_accuracy: 0.9420
Epoch 39/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0213 - categorical_accuracy: 0.9552 - val_loss: 0.0227 - val_categorical_accuracy: 0.9490
Epoch 40/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0210 - categorical_accuracy: 0.9560 - val_loss: 0.0231 - val_categorical_accuracy: 0.9443
Epoch 41/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0210 - categorical_accuracy: 0.9551 - val_loss: 0.0275 - val_categorical_accuracy: 0.9239
Epoch 42/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0208 - categorical_accuracy: 0.9571 - val_loss: 0.0259 - val_categorical_accuracy: 0.9302
Epoch 43/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0207 - categorical_accuracy: 0.9559 - val_loss: 0.0232 - val_categorical_accuracy: 0.9458
Epoch 44/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0205 - categorical_accuracy: 0.9564 - val_loss: 0.0260 - val_categorical_accuracy: 0.9308
Epoch 45/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0209 - categorical_accuracy: 0.9544 - val_loss: 0.0229 - val_categorical_accuracy: 0.9475

Epoch 00045: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 46/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0175 - categorical_accuracy: 0.9735 - val_loss: 0.0198 - val_categorical_accuracy: 0.9618
Epoch 47/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0162 - categorical_accuracy: 0.9787 - val_loss: 0.0190 - val_categorical_accuracy: 0.9649
Epoch 48/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0155 - categorical_accuracy: 0.9809 - val_loss: 0.0186 - val_categorical_accuracy: 0.9655
Epoch 49/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0149 - categorical_accuracy: 0.9819 - val_loss: 0.0184 - val_categorical_accuracy: 0.9648
Epoch 50/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0144 - categorical_accuracy: 0.9827 - val_loss: 0.0179 - val_categorical_accuracy: 0.9663
Epoch 51/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0140 - categorical_accuracy: 0.9835 - val_loss: 0.0175 - val_categorical_accuracy: 0.9662
Epoch 52/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0136 - categorical_accuracy: 0.9844 - val_loss: 0.0173 - val_categorical_accuracy: 0.9657
Epoch 53/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0131 - categorical_accuracy: 0.9849 - val_loss: 0.0173 - val_categorical_accuracy: 0.9652
Epoch 54/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0127 - categorical_accuracy: 0.9858 - val_loss: 0.0170 - val_categorical_accuracy: 0.9651
Epoch 55/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0124 - categorical_accuracy: 0.9855 - val_loss: 0.0169 - val_categorical_accuracy: 0.9636
Epoch 56/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0121 - categorical_accuracy: 0.9855 - val_loss: 0.0166 - val_categorical_accuracy: 0.9632
Epoch 57/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0117 - categorical_accuracy: 0.9866 - val_loss: 0.0163 - val_categorical_accuracy: 0.9649
Epoch 58/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0114 - categorical_accuracy: 0.9868 - val_loss: 0.0159 - val_categorical_accuracy: 0.9661
Epoch 59/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0111 - categorical_accuracy: 0.9870 - val_loss: 0.0158 - val_categorical_accuracy: 0.9647
Epoch 60/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0108 - categorical_accuracy: 0.9877 - val_loss: 0.0159 - val_categorical_accuracy: 0.9635
Epoch 61/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0106 - categorical_accuracy: 0.9878 - val_loss: 0.0161 - val_categorical_accuracy: 0.9638
Epoch 62/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0105 - categorical_accuracy: 0.9871 - val_loss: 0.0153 - val_categorical_accuracy: 0.9661
Epoch 63/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0100 - categorical_accuracy: 0.9886 - val_loss: 0.0154 - val_categorical_accuracy: 0.9655
Epoch 64/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0099 - categorical_accuracy: 0.9881 - val_loss: 0.0152 - val_categorical_accuracy: 0.9648
Epoch 65/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0097 - categorical_accuracy: 0.9882 - val_loss: 0.0151 - val_categorical_accuracy: 0.9647
Epoch 66/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0094 - categorical_accuracy: 0.9885 - val_loss: 0.0147 - val_categorical_accuracy: 0.9662
Epoch 67/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0092 - categorical_accuracy: 0.9890 - val_loss: 0.0151 - val_categorical_accuracy: 0.9632
Epoch 68/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0091 - categorical_accuracy: 0.9885 - val_loss: 0.0154 - val_categorical_accuracy: 0.9617
Epoch 69/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0090 - categorical_accuracy: 0.9889 - val_loss: 0.0147 - val_categorical_accuracy: 0.9643
Epoch 70/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0089 - categorical_accuracy: 0.9879 - val_loss: 0.0146 - val_categorical_accuracy: 0.9640
Epoch 71/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0087 - categorical_accuracy: 0.9881 - val_loss: 0.0145 - val_categorical_accuracy: 0.9644
Epoch 72/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0085 - categorical_accuracy: 0.9894 - val_loss: 0.0146 - val_categorical_accuracy: 0.9620
Epoch 73/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0084 - categorical_accuracy: 0.9893 - val_loss: 0.0141 - val_categorical_accuracy: 0.9639
Epoch 74/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0083 - categorical_accuracy: 0.9890 - val_loss: 0.0151 - val_categorical_accuracy: 0.9615
Epoch 75/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0082 - categorical_accuracy: 0.9895 - val_loss: 0.0145 - val_categorical_accuracy: 0.9629
Epoch 76/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0081 - categorical_accuracy: 0.9897 - val_loss: 0.0142 - val_categorical_accuracy: 0.9631
Epoch 77/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0081 - categorical_accuracy: 0.9879 - val_loss: 0.0140 - val_categorical_accuracy: 0.9632
Epoch 78/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0080 - categorical_accuracy: 0.9888 - val_loss: 0.0140 - val_categorical_accuracy: 0.9642
Epoch 79/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0079 - categorical_accuracy: 0.9892 - val_loss: 0.0146 - val_categorical_accuracy: 0.9614
Epoch 80/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0079 - categorical_accuracy: 0.9888 - val_loss: 0.0144 - val_categorical_accuracy: 0.9593
Epoch 81/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0077 - categorical_accuracy: 0.9899 - val_loss: 0.0144 - val_categorical_accuracy: 0.9615
Epoch 82/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0077 - categorical_accuracy: 0.9897 - val_loss: 0.0145 - val_categorical_accuracy: 0.9599
Epoch 83/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0078 - categorical_accuracy: 0.9886 - val_loss: 0.0147 - val_categorical_accuracy: 0.9614

Epoch 00083: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 84/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0071 - categorical_accuracy: 0.9926 - val_loss: 0.0135 - val_categorical_accuracy: 0.9649
Epoch 85/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0068 - categorical_accuracy: 0.9939 - val_loss: 0.0135 - val_categorical_accuracy: 0.9654
Epoch 86/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0067 - categorical_accuracy: 0.9947 - val_loss: 0.0133 - val_categorical_accuracy: 0.9661
Epoch 87/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0066 - categorical_accuracy: 0.9949 - val_loss: 0.0134 - val_categorical_accuracy: 0.9667
Epoch 88/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0065 - categorical_accuracy: 0.9955 - val_loss: 0.0133 - val_categorical_accuracy: 0.9671
Epoch 89/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0064 - categorical_accuracy: 0.9956 - val_loss: 0.0132 - val_categorical_accuracy: 0.9669
Epoch 90/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0064 - categorical_accuracy: 0.9960 - val_loss: 0.0132 - val_categorical_accuracy: 0.9677
Epoch 91/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0063 - categorical_accuracy: 0.9962 - val_loss: 0.0134 - val_categorical_accuracy: 0.9675
Epoch 92/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0062 - categorical_accuracy: 0.9965 - val_loss: 0.0132 - val_categorical_accuracy: 0.9683
Epoch 93/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9962 - val_loss: 0.0131 - val_categorical_accuracy: 0.9671
Epoch 94/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9962 - val_loss: 0.0132 - val_categorical_accuracy: 0.9678
Epoch 95/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9964 - val_loss: 0.0131 - val_categorical_accuracy: 0.9668

Epoch 00095: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 96/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0060 - categorical_accuracy: 0.9967 - val_loss: 0.0131 - val_categorical_accuracy: 0.9675
Epoch 97/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0060 - categorical_accuracy: 0.9968 - val_loss: 0.0131 - val_categorical_accuracy: 0.9678
Epoch 98/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0059 - categorical_accuracy: 0.9971 - val_loss: 0.0131 - val_categorical_accuracy: 0.9674
Epoch 99/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0060 - categorical_accuracy: 0.9970 - val_loss: 0.0131 - val_categorical_accuracy: 0.9681
Epoch 100/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0059 - categorical_accuracy: 0.9971 - val_loss: 0.0131 - val_categorical_accuracy: 0.9684

Epoch 00100: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 101/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0059 - categorical_accuracy: 0.9972 - val_loss: 0.0131 - val_categorical_accuracy: 0.9682
Epoch 102/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0059 - categorical_accuracy: 0.9972 - val_loss: 0.0131 - val_categorical_accuracy: 0.9684
Epoch 103/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0059 - categorical_accuracy: 0.9973 - val_loss: 0.0131 - val_categorical_accuracy: 0.9686
Epoch 104/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0059 - categorical_accuracy: 0.9968 - val_loss: 0.0131 - val_categorical_accuracy: 0.9683
Epoch 105/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0059 - categorical_accuracy: 0.9974 - val_loss: 0.0131 - val_categorical_accuracy: 0.9682

Epoch 00105: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 106/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0059 - categorical_accuracy: 0.9968 - val_loss: 0.0131 - val_categorical_accuracy: 0.9681
Epoch 107/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0059 - categorical_accuracy: 0.9970 - val_loss: 0.0131 - val_categorical_accuracy: 0.9681
Epoch 108/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0059 - categorical_accuracy: 0.9971 - val_loss: 0.0131 - val_categorical_accuracy: 0.9681
Epoch 109/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0059 - categorical_accuracy: 0.9975 - val_loss: 0.0131 - val_categorical_accuracy: 0.9682
Epoch 110/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0059 - categorical_accuracy: 0.9974 - val_loss: 0.0131 - val_categorical_accuracy: 0.9682

Epoch 00110: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 00110: early stopping
========= generating oof predictions 08:45:30 =========
========= generating test set predictions 08:45:34 =========
========= fitting 2 th model 08:46:22 =========
Epoch 1/65535
420/420 [==============================] - 57s 135ms/step - loss: 0.5350 - categorical_accuracy: 0.1774 - val_loss: 0.4737 - val_categorical_accuracy: 0.3608
Epoch 2/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.4253 - categorical_accuracy: 0.5051 - val_loss: 0.3826 - val_categorical_accuracy: 0.6081
Epoch 3/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.3459 - categorical_accuracy: 0.6883 - val_loss: 0.3170 - val_categorical_accuracy: 0.7201
Epoch 4/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.2871 - categorical_accuracy: 0.7815 - val_loss: 0.2602 - val_categorical_accuracy: 0.8233
Epoch 5/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.2421 - categorical_accuracy: 0.8321 - val_loss: 0.2215 - val_categorical_accuracy: 0.8551
Epoch 6/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.2057 - categorical_accuracy: 0.8629 - val_loss: 0.1883 - val_categorical_accuracy: 0.8797
Epoch 7/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.1756 - categorical_accuracy: 0.8816 - val_loss: 0.1610 - val_categorical_accuracy: 0.8978
Epoch 8/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.1511 - categorical_accuracy: 0.8943 - val_loss: 0.1399 - val_categorical_accuracy: 0.8996
Epoch 9/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.1304 - categorical_accuracy: 0.9051 - val_loss: 0.1203 - val_categorical_accuracy: 0.9142
Epoch 10/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.1131 - categorical_accuracy: 0.9131 - val_loss: 0.1044 - val_categorical_accuracy: 0.9230
Epoch 11/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0986 - categorical_accuracy: 0.9185 - val_loss: 0.0952 - val_categorical_accuracy: 0.9037
Epoch 12/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0865 - categorical_accuracy: 0.9228 - val_loss: 0.0840 - val_categorical_accuracy: 0.9087
Epoch 13/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0762 - categorical_accuracy: 0.9280 - val_loss: 0.0731 - val_categorical_accuracy: 0.9205
Epoch 14/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0676 - categorical_accuracy: 0.9313 - val_loss: 0.0665 - val_categorical_accuracy: 0.9181
Epoch 15/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0604 - categorical_accuracy: 0.9334 - val_loss: 0.0595 - val_categorical_accuracy: 0.9229
Epoch 16/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0544 - categorical_accuracy: 0.9358 - val_loss: 0.0531 - val_categorical_accuracy: 0.9311
Epoch 17/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0491 - categorical_accuracy: 0.9391 - val_loss: 0.0486 - val_categorical_accuracy: 0.9289
Epoch 18/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0448 - categorical_accuracy: 0.9407 - val_loss: 0.0448 - val_categorical_accuracy: 0.9324
Epoch 19/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0416 - categorical_accuracy: 0.9391 - val_loss: 0.0454 - val_categorical_accuracy: 0.9122
Epoch 20/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0383 - categorical_accuracy: 0.9439 - val_loss: 0.0413 - val_categorical_accuracy: 0.9220
Epoch 21/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0358 - categorical_accuracy: 0.9437 - val_loss: 0.0395 - val_categorical_accuracy: 0.9216
Epoch 22/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0335 - categorical_accuracy: 0.9452 - val_loss: 0.0348 - val_categorical_accuracy: 0.9340
Epoch 23/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0321 - categorical_accuracy: 0.9437 - val_loss: 0.0342 - val_categorical_accuracy: 0.9268
Epoch 24/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0303 - categorical_accuracy: 0.9465 - val_loss: 0.0384 - val_categorical_accuracy: 0.9040
Epoch 25/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0285 - categorical_accuracy: 0.9496 - val_loss: 0.0351 - val_categorical_accuracy: 0.9114
Epoch 26/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0276 - categorical_accuracy: 0.9482 - val_loss: 0.0287 - val_categorical_accuracy: 0.9398
Epoch 27/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0264 - categorical_accuracy: 0.9504 - val_loss: 0.0281 - val_categorical_accuracy: 0.9400
Epoch 28/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0257 - categorical_accuracy: 0.9495 - val_loss: 0.0292 - val_categorical_accuracy: 0.9293
Epoch 29/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0252 - categorical_accuracy: 0.9494 - val_loss: 0.0290 - val_categorical_accuracy: 0.9295
Epoch 30/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0244 - categorical_accuracy: 0.9505 - val_loss: 0.0269 - val_categorical_accuracy: 0.9386
Epoch 31/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0233 - categorical_accuracy: 0.9540 - val_loss: 0.0271 - val_categorical_accuracy: 0.9338
Epoch 32/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0233 - categorical_accuracy: 0.9516 - val_loss: 0.0269 - val_categorical_accuracy: 0.9356
Epoch 33/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0232 - categorical_accuracy: 0.9514 - val_loss: 0.0264 - val_categorical_accuracy: 0.9365
Epoch 34/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0229 - categorical_accuracy: 0.9519 - val_loss: 0.0302 - val_categorical_accuracy: 0.9148
Epoch 35/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0221 - categorical_accuracy: 0.9543 - val_loss: 0.0268 - val_categorical_accuracy: 0.9311
Epoch 36/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0220 - categorical_accuracy: 0.9540 - val_loss: 0.0243 - val_categorical_accuracy: 0.9419
Epoch 37/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0217 - categorical_accuracy: 0.9542 - val_loss: 0.0307 - val_categorical_accuracy: 0.9087
Epoch 38/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0216 - categorical_accuracy: 0.9552 - val_loss: 0.0260 - val_categorical_accuracy: 0.9327
Epoch 39/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0214 - categorical_accuracy: 0.9557 - val_loss: 0.0246 - val_categorical_accuracy: 0.9398
Epoch 40/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0211 - categorical_accuracy: 0.9558 - val_loss: 0.0252 - val_categorical_accuracy: 0.9361
Epoch 41/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0212 - categorical_accuracy: 0.9547 - val_loss: 0.0240 - val_categorical_accuracy: 0.9407
Epoch 42/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0208 - categorical_accuracy: 0.9552 - val_loss: 0.0302 - val_categorical_accuracy: 0.9100
Epoch 43/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0206 - categorical_accuracy: 0.9566 - val_loss: 0.0280 - val_categorical_accuracy: 0.9242
Epoch 44/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0209 - categorical_accuracy: 0.9550 - val_loss: 0.0252 - val_categorical_accuracy: 0.9337
Epoch 45/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0204 - categorical_accuracy: 0.9570 - val_loss: 0.0234 - val_categorical_accuracy: 0.9439
Epoch 46/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0205 - categorical_accuracy: 0.9563 - val_loss: 0.0245 - val_categorical_accuracy: 0.9387
Epoch 47/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0204 - categorical_accuracy: 0.9567 - val_loss: 0.0271 - val_categorical_accuracy: 0.9229
Epoch 48/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0202 - categorical_accuracy: 0.9582 - val_loss: 0.0242 - val_categorical_accuracy: 0.9363
Epoch 49/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0202 - categorical_accuracy: 0.9565 - val_loss: 0.0233 - val_categorical_accuracy: 0.9439
Epoch 50/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0202 - categorical_accuracy: 0.9575 - val_loss: 0.0229 - val_categorical_accuracy: 0.9425
Epoch 51/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0201 - categorical_accuracy: 0.9580 - val_loss: 0.0248 - val_categorical_accuracy: 0.9375
Epoch 52/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0201 - categorical_accuracy: 0.9578 - val_loss: 0.0228 - val_categorical_accuracy: 0.9453
Epoch 53/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0200 - categorical_accuracy: 0.9585 - val_loss: 0.0253 - val_categorical_accuracy: 0.9305
Epoch 54/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0198 - categorical_accuracy: 0.9589 - val_loss: 0.0280 - val_categorical_accuracy: 0.9262
Epoch 55/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0198 - categorical_accuracy: 0.9591 - val_loss: 0.0226 - val_categorical_accuracy: 0.9462
Epoch 56/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0197 - categorical_accuracy: 0.9593 - val_loss: 0.0244 - val_categorical_accuracy: 0.9388
Epoch 57/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0198 - categorical_accuracy: 0.9582 - val_loss: 0.0233 - val_categorical_accuracy: 0.9393
Epoch 58/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0197 - categorical_accuracy: 0.9596 - val_loss: 0.0239 - val_categorical_accuracy: 0.9386
Epoch 59/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0195 - categorical_accuracy: 0.9604 - val_loss: 0.0232 - val_categorical_accuracy: 0.9431
Epoch 60/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0199 - categorical_accuracy: 0.9570 - val_loss: 0.0237 - val_categorical_accuracy: 0.9375
Epoch 61/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0196 - categorical_accuracy: 0.9594 - val_loss: 0.0232 - val_categorical_accuracy: 0.9419

Epoch 00061: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 62/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0165 - categorical_accuracy: 0.9769 - val_loss: 0.0188 - val_categorical_accuracy: 0.9627
Epoch 63/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0153 - categorical_accuracy: 0.9811 - val_loss: 0.0183 - val_categorical_accuracy: 0.9649
Epoch 64/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0148 - categorical_accuracy: 0.9825 - val_loss: 0.0179 - val_categorical_accuracy: 0.9649
Epoch 65/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0142 - categorical_accuracy: 0.9836 - val_loss: 0.0176 - val_categorical_accuracy: 0.9647
Epoch 66/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0137 - categorical_accuracy: 0.9845 - val_loss: 0.0172 - val_categorical_accuracy: 0.9649
Epoch 67/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0133 - categorical_accuracy: 0.9849 - val_loss: 0.0173 - val_categorical_accuracy: 0.9631
Epoch 68/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0129 - categorical_accuracy: 0.9855 - val_loss: 0.0169 - val_categorical_accuracy: 0.9631
Epoch 69/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0125 - categorical_accuracy: 0.9860 - val_loss: 0.0166 - val_categorical_accuracy: 0.9636
Epoch 70/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0121 - categorical_accuracy: 0.9865 - val_loss: 0.0162 - val_categorical_accuracy: 0.9656
Epoch 71/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0117 - categorical_accuracy: 0.9875 - val_loss: 0.0159 - val_categorical_accuracy: 0.9655
Epoch 72/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0115 - categorical_accuracy: 0.9873 - val_loss: 0.0158 - val_categorical_accuracy: 0.9652
Epoch 73/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0111 - categorical_accuracy: 0.9879 - val_loss: 0.0155 - val_categorical_accuracy: 0.9661
Epoch 74/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0108 - categorical_accuracy: 0.9882 - val_loss: 0.0155 - val_categorical_accuracy: 0.9646
Epoch 75/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0106 - categorical_accuracy: 0.9880 - val_loss: 0.0153 - val_categorical_accuracy: 0.9646
Epoch 76/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0103 - categorical_accuracy: 0.9882 - val_loss: 0.0152 - val_categorical_accuracy: 0.9636
Epoch 77/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0100 - categorical_accuracy: 0.9886 - val_loss: 0.0150 - val_categorical_accuracy: 0.9635
Epoch 78/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0098 - categorical_accuracy: 0.9886 - val_loss: 0.0145 - val_categorical_accuracy: 0.9652
Epoch 79/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0096 - categorical_accuracy: 0.9891 - val_loss: 0.0145 - val_categorical_accuracy: 0.9650
Epoch 80/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0094 - categorical_accuracy: 0.9891 - val_loss: 0.0153 - val_categorical_accuracy: 0.9619
Epoch 81/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0091 - categorical_accuracy: 0.9895 - val_loss: 0.0151 - val_categorical_accuracy: 0.9610
Epoch 82/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0089 - categorical_accuracy: 0.9896 - val_loss: 0.0142 - val_categorical_accuracy: 0.9645
Epoch 83/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0087 - categorical_accuracy: 0.9898 - val_loss: 0.0144 - val_categorical_accuracy: 0.9638
Epoch 84/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0086 - categorical_accuracy: 0.9896 - val_loss: 0.0138 - val_categorical_accuracy: 0.9641
Epoch 85/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0084 - categorical_accuracy: 0.9899 - val_loss: 0.0142 - val_categorical_accuracy: 0.9622
Epoch 86/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0084 - categorical_accuracy: 0.9894 - val_loss: 0.0137 - val_categorical_accuracy: 0.9626
Epoch 87/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0082 - categorical_accuracy: 0.9901 - val_loss: 0.0137 - val_categorical_accuracy: 0.9644
Epoch 88/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0080 - categorical_accuracy: 0.9904 - val_loss: 0.0140 - val_categorical_accuracy: 0.9635
Epoch 89/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0079 - categorical_accuracy: 0.9897 - val_loss: 0.0143 - val_categorical_accuracy: 0.9602
Epoch 90/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0079 - categorical_accuracy: 0.9894 - val_loss: 0.0145 - val_categorical_accuracy: 0.9576

Epoch 00090: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 91/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0074 - categorical_accuracy: 0.9925 - val_loss: 0.0130 - val_categorical_accuracy: 0.9666
Epoch 92/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0072 - categorical_accuracy: 0.9934 - val_loss: 0.0129 - val_categorical_accuracy: 0.9660
Epoch 93/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0071 - categorical_accuracy: 0.9938 - val_loss: 0.0129 - val_categorical_accuracy: 0.9659
Epoch 94/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0070 - categorical_accuracy: 0.9942 - val_loss: 0.0128 - val_categorical_accuracy: 0.9659
Epoch 95/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0070 - categorical_accuracy: 0.9940 - val_loss: 0.0128 - val_categorical_accuracy: 0.9663
Epoch 96/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9945 - val_loss: 0.0128 - val_categorical_accuracy: 0.9662
Epoch 97/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0067 - categorical_accuracy: 0.9949 - val_loss: 0.0128 - val_categorical_accuracy: 0.9662
Epoch 98/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0067 - categorical_accuracy: 0.9951 - val_loss: 0.0127 - val_categorical_accuracy: 0.9673
Epoch 99/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9952 - val_loss: 0.0128 - val_categorical_accuracy: 0.9668
Epoch 100/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9954 - val_loss: 0.0127 - val_categorical_accuracy: 0.9670
Epoch 101/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0065 - categorical_accuracy: 0.9954 - val_loss: 0.0127 - val_categorical_accuracy: 0.9671
Epoch 102/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0065 - categorical_accuracy: 0.9956 - val_loss: 0.0128 - val_categorical_accuracy: 0.9669
Epoch 103/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0064 - categorical_accuracy: 0.9957 - val_loss: 0.0128 - val_categorical_accuracy: 0.9668
Epoch 104/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0064 - categorical_accuracy: 0.9958 - val_loss: 0.0127 - val_categorical_accuracy: 0.9667

Epoch 00104: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 105/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0064 - categorical_accuracy: 0.9955 - val_loss: 0.0127 - val_categorical_accuracy: 0.9665
Epoch 106/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0063 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9672
Epoch 107/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0063 - categorical_accuracy: 0.9958 - val_loss: 0.0127 - val_categorical_accuracy: 0.9672
Epoch 108/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0063 - categorical_accuracy: 0.9963 - val_loss: 0.0127 - val_categorical_accuracy: 0.9669
Epoch 109/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9962 - val_loss: 0.0127 - val_categorical_accuracy: 0.9672

Epoch 00109: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 110/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9960 - val_loss: 0.0127 - val_categorical_accuracy: 0.9670
Epoch 111/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0127 - val_categorical_accuracy: 0.9671
Epoch 112/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0062 - categorical_accuracy: 0.9960 - val_loss: 0.0127 - val_categorical_accuracy: 0.9672
Epoch 113/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9671
Epoch 114/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9962 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674

Epoch 00114: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 115/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9670
Epoch 116/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0062 - categorical_accuracy: 0.9965 - val_loss: 0.0127 - val_categorical_accuracy: 0.9676
Epoch 117/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674
Epoch 118/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0063 - categorical_accuracy: 0.9959 - val_loss: 0.0127 - val_categorical_accuracy: 0.9677
Epoch 119/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0127 - val_categorical_accuracy: 0.9677

Epoch 00119: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 120/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0127 - val_categorical_accuracy: 0.9677
Epoch 121/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9969 - val_loss: 0.0127 - val_categorical_accuracy: 0.9676
Epoch 122/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9962 - val_loss: 0.0127 - val_categorical_accuracy: 0.9677
Epoch 123/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9962 - val_loss: 0.0127 - val_categorical_accuracy: 0.9675
Epoch 124/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9965 - val_loss: 0.0127 - val_categorical_accuracy: 0.9678

Epoch 00124: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 125/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9961 - val_loss: 0.0127 - val_categorical_accuracy: 0.9678
Epoch 126/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9677
Epoch 127/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9961 - val_loss: 0.0127 - val_categorical_accuracy: 0.9676
Epoch 128/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9962 - val_loss: 0.0127 - val_categorical_accuracy: 0.9676
Epoch 129/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9965 - val_loss: 0.0127 - val_categorical_accuracy: 0.9676

Epoch 00129: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 130/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0127 - val_categorical_accuracy: 0.9675
Epoch 131/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9967 - val_loss: 0.0127 - val_categorical_accuracy: 0.9675
Epoch 132/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9675
Epoch 133/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9965 - val_loss: 0.0127 - val_categorical_accuracy: 0.9673
Epoch 134/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9675

Epoch 00134: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 135/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0062 - categorical_accuracy: 0.9963 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674
Epoch 136/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9963 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674
Epoch 137/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0127 - val_categorical_accuracy: 0.9673
Epoch 138/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9962 - val_loss: 0.0127 - val_categorical_accuracy: 0.9672
Epoch 139/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9968 - val_loss: 0.0127 - val_categorical_accuracy: 0.9675

Epoch 00139: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 140/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9965 - val_loss: 0.0127 - val_categorical_accuracy: 0.9675
Epoch 141/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674
Epoch 142/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9967 - val_loss: 0.0127 - val_categorical_accuracy: 0.9677
Epoch 143/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9960 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674
Epoch 144/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9961 - val_loss: 0.0127 - val_categorical_accuracy: 0.9673

Epoch 00144: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 145/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9672
Epoch 146/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0061 - categorical_accuracy: 0.9969 - val_loss: 0.0127 - val_categorical_accuracy: 0.9673
Epoch 147/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9672
Epoch 148/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9968 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674
Epoch 149/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9676

Epoch 00149: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 150/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0126 - val_categorical_accuracy: 0.9673
Epoch 151/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675
Epoch 152/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0126 - val_categorical_accuracy: 0.9673
Epoch 153/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9968 - val_loss: 0.0126 - val_categorical_accuracy: 0.9673
Epoch 154/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9964 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675

Epoch 00154: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 155/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9969 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675
Epoch 156/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9968 - val_loss: 0.0126 - val_categorical_accuracy: 0.9673
Epoch 157/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9969 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675
Epoch 158/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9962 - val_loss: 0.0126 - val_categorical_accuracy: 0.9674
Epoch 159/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9968 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675

Epoch 00159: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 160/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9966 - val_loss: 0.0126 - val_categorical_accuracy: 0.9674
Epoch 161/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9968 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675
Epoch 162/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9965 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675
Epoch 163/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9967 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675
Epoch 164/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9969 - val_loss: 0.0127 - val_categorical_accuracy: 0.9675

Epoch 00164: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 165/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9965 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674
Epoch 166/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9962 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675
Epoch 167/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0126 - val_categorical_accuracy: 0.9674
Epoch 168/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9969 - val_loss: 0.0127 - val_categorical_accuracy: 0.9673
Epoch 169/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9963 - val_loss: 0.0126 - val_categorical_accuracy: 0.9674

Epoch 00169: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 170/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9963 - val_loss: 0.0126 - val_categorical_accuracy: 0.9675
Epoch 171/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9964 - val_loss: 0.0126 - val_categorical_accuracy: 0.9676
Epoch 00171: early stopping
========= generating oof predictions 11:24:47 =========
========= generating test set predictions 11:24:51 =========
========= fitting 3 th model 11:25:40 =========
Epoch 1/65535
420/420 [==============================] - 57s 135ms/step - loss: 0.5345 - categorical_accuracy: 0.1888 - val_loss: 0.4794 - val_categorical_accuracy: 0.3350
Epoch 2/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.4264 - categorical_accuracy: 0.5024 - val_loss: 0.3836 - val_categorical_accuracy: 0.6024
Epoch 3/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.3473 - categorical_accuracy: 0.6838 - val_loss: 0.3124 - val_categorical_accuracy: 0.7620
Epoch 4/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.2884 - categorical_accuracy: 0.7786 - val_loss: 0.2636 - val_categorical_accuracy: 0.8083
Epoch 5/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.2427 - categorical_accuracy: 0.8321 - val_loss: 0.2225 - val_categorical_accuracy: 0.8497
Epoch 6/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.2062 - categorical_accuracy: 0.8593 - val_loss: 0.1891 - val_categorical_accuracy: 0.8804
Epoch 7/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.1763 - categorical_accuracy: 0.8810 - val_loss: 0.1619 - val_categorical_accuracy: 0.8959
Epoch 8/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.1514 - categorical_accuracy: 0.8960 - val_loss: 0.1400 - val_categorical_accuracy: 0.9037
Epoch 9/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.1308 - categorical_accuracy: 0.9048 - val_loss: 0.1235 - val_categorical_accuracy: 0.8990
Epoch 10/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.1131 - categorical_accuracy: 0.9144 - val_loss: 0.1061 - val_categorical_accuracy: 0.9132
Epoch 11/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0987 - categorical_accuracy: 0.9199 - val_loss: 0.0918 - val_categorical_accuracy: 0.9256
Epoch 12/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0862 - categorical_accuracy: 0.9252 - val_loss: 0.0814 - val_categorical_accuracy: 0.9259
Epoch 13/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0761 - categorical_accuracy: 0.9284 - val_loss: 0.0722 - val_categorical_accuracy: 0.9282
Epoch 14/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0674 - categorical_accuracy: 0.9315 - val_loss: 0.0649 - val_categorical_accuracy: 0.9264
Epoch 15/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0601 - categorical_accuracy: 0.9345 - val_loss: 0.0604 - val_categorical_accuracy: 0.9172
Epoch 16/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0541 - categorical_accuracy: 0.9351 - val_loss: 0.0525 - val_categorical_accuracy: 0.9354
Epoch 17/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0490 - categorical_accuracy: 0.9387 - val_loss: 0.0488 - val_categorical_accuracy: 0.9280
Epoch 18/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0447 - categorical_accuracy: 0.9397 - val_loss: 0.0456 - val_categorical_accuracy: 0.9265
Epoch 19/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0414 - categorical_accuracy: 0.9400 - val_loss: 0.0412 - val_categorical_accuracy: 0.9347
Epoch 20/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0381 - categorical_accuracy: 0.9418 - val_loss: 0.0380 - val_categorical_accuracy: 0.9372
Epoch 21/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0355 - categorical_accuracy: 0.9435 - val_loss: 0.0385 - val_categorical_accuracy: 0.9242
Epoch 22/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0332 - categorical_accuracy: 0.9457 - val_loss: 0.0369 - val_categorical_accuracy: 0.9230
Epoch 23/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0319 - categorical_accuracy: 0.9433 - val_loss: 0.0354 - val_categorical_accuracy: 0.9227
Epoch 24/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0300 - categorical_accuracy: 0.9467 - val_loss: 0.0314 - val_categorical_accuracy: 0.9382
Epoch 25/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0285 - categorical_accuracy: 0.9475 - val_loss: 0.0301 - val_categorical_accuracy: 0.9411
Epoch 26/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0271 - categorical_accuracy: 0.9500 - val_loss: 0.0292 - val_categorical_accuracy: 0.9358
Epoch 27/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0265 - categorical_accuracy: 0.9482 - val_loss: 0.0278 - val_categorical_accuracy: 0.9435
Epoch 28/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0258 - categorical_accuracy: 0.9482 - val_loss: 0.0282 - val_categorical_accuracy: 0.9370
Epoch 29/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0248 - categorical_accuracy: 0.9502 - val_loss: 0.0315 - val_categorical_accuracy: 0.9131
Epoch 30/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0242 - categorical_accuracy: 0.9510 - val_loss: 0.0273 - val_categorical_accuracy: 0.9344
Epoch 31/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0237 - categorical_accuracy: 0.9510 - val_loss: 0.0309 - val_categorical_accuracy: 0.9151
Epoch 32/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0232 - categorical_accuracy: 0.9519 - val_loss: 0.0276 - val_categorical_accuracy: 0.9293
Epoch 33/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0229 - categorical_accuracy: 0.9513 - val_loss: 0.0309 - val_categorical_accuracy: 0.9124
Epoch 34/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0222 - categorical_accuracy: 0.9533 - val_loss: 0.0335 - val_categorical_accuracy: 0.8993
Epoch 35/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0222 - categorical_accuracy: 0.9536 - val_loss: 0.0269 - val_categorical_accuracy: 0.9274
Epoch 36/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0221 - categorical_accuracy: 0.9520 - val_loss: 0.0275 - val_categorical_accuracy: 0.9250
Epoch 37/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0216 - categorical_accuracy: 0.9550 - val_loss: 0.0273 - val_categorical_accuracy: 0.9250
Epoch 38/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0214 - categorical_accuracy: 0.9543 - val_loss: 0.0236 - val_categorical_accuracy: 0.9442
Epoch 39/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0213 - categorical_accuracy: 0.9548 - val_loss: 0.0267 - val_categorical_accuracy: 0.9268
Epoch 40/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0211 - categorical_accuracy: 0.9556 - val_loss: 0.0233 - val_categorical_accuracy: 0.9446
Epoch 41/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0208 - categorical_accuracy: 0.9558 - val_loss: 0.0256 - val_categorical_accuracy: 0.9359
Epoch 42/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0207 - categorical_accuracy: 0.9560 - val_loss: 0.0264 - val_categorical_accuracy: 0.9272
Epoch 43/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0207 - categorical_accuracy: 0.9555 - val_loss: 0.0228 - val_categorical_accuracy: 0.9493
Epoch 44/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0206 - categorical_accuracy: 0.9552 - val_loss: 0.0231 - val_categorical_accuracy: 0.9456
Epoch 45/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0205 - categorical_accuracy: 0.9557 - val_loss: 0.0237 - val_categorical_accuracy: 0.9429
Epoch 46/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0202 - categorical_accuracy: 0.9573 - val_loss: 0.0302 - val_categorical_accuracy: 0.9076
Epoch 47/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0199 - categorical_accuracy: 0.9585 - val_loss: 0.0261 - val_categorical_accuracy: 0.9298
Epoch 48/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0204 - categorical_accuracy: 0.9552 - val_loss: 0.0235 - val_categorical_accuracy: 0.9424
Epoch 49/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0201 - categorical_accuracy: 0.9573 - val_loss: 0.0256 - val_categorical_accuracy: 0.9322

Epoch 00049: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 50/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0171 - categorical_accuracy: 0.9746 - val_loss: 0.0190 - val_categorical_accuracy: 0.9642
Epoch 51/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0159 - categorical_accuracy: 0.9791 - val_loss: 0.0186 - val_categorical_accuracy: 0.9648
Epoch 52/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0153 - categorical_accuracy: 0.9811 - val_loss: 0.0183 - val_categorical_accuracy: 0.9647
Epoch 53/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0146 - categorical_accuracy: 0.9825 - val_loss: 0.0179 - val_categorical_accuracy: 0.9644
Epoch 54/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0141 - categorical_accuracy: 0.9830 - val_loss: 0.0176 - val_categorical_accuracy: 0.9648
Epoch 55/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0137 - categorical_accuracy: 0.9833 - val_loss: 0.0172 - val_categorical_accuracy: 0.9659
Epoch 56/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0133 - categorical_accuracy: 0.9845 - val_loss: 0.0168 - val_categorical_accuracy: 0.9657
Epoch 57/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0129 - categorical_accuracy: 0.9851 - val_loss: 0.0167 - val_categorical_accuracy: 0.9662
Epoch 58/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0125 - categorical_accuracy: 0.9854 - val_loss: 0.0166 - val_categorical_accuracy: 0.9639
Epoch 59/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0121 - categorical_accuracy: 0.9863 - val_loss: 0.0161 - val_categorical_accuracy: 0.9651
Epoch 60/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0117 - categorical_accuracy: 0.9864 - val_loss: 0.0157 - val_categorical_accuracy: 0.9667
Epoch 61/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0115 - categorical_accuracy: 0.9869 - val_loss: 0.0161 - val_categorical_accuracy: 0.9638
Epoch 62/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0112 - categorical_accuracy: 0.9871 - val_loss: 0.0156 - val_categorical_accuracy: 0.9656
Epoch 63/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0109 - categorical_accuracy: 0.9875 - val_loss: 0.0157 - val_categorical_accuracy: 0.9630
Epoch 64/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0107 - categorical_accuracy: 0.9874 - val_loss: 0.0155 - val_categorical_accuracy: 0.9641
Epoch 65/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0104 - categorical_accuracy: 0.9872 - val_loss: 0.0152 - val_categorical_accuracy: 0.9636
Epoch 66/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0101 - categorical_accuracy: 0.9880 - val_loss: 0.0152 - val_categorical_accuracy: 0.9649
Epoch 67/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0099 - categorical_accuracy: 0.9882 - val_loss: 0.0151 - val_categorical_accuracy: 0.9645
Epoch 68/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0096 - categorical_accuracy: 0.9884 - val_loss: 0.0149 - val_categorical_accuracy: 0.9644
Epoch 69/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0094 - categorical_accuracy: 0.9894 - val_loss: 0.0147 - val_categorical_accuracy: 0.9633
Epoch 70/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0093 - categorical_accuracy: 0.9887 - val_loss: 0.0147 - val_categorical_accuracy: 0.9639
Epoch 71/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0091 - categorical_accuracy: 0.9887 - val_loss: 0.0146 - val_categorical_accuracy: 0.9623
Epoch 72/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0088 - categorical_accuracy: 0.9893 - val_loss: 0.0146 - val_categorical_accuracy: 0.9626
Epoch 73/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0088 - categorical_accuracy: 0.9885 - val_loss: 0.0141 - val_categorical_accuracy: 0.9648
Epoch 74/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0085 - categorical_accuracy: 0.9893 - val_loss: 0.0140 - val_categorical_accuracy: 0.9648
Epoch 75/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0084 - categorical_accuracy: 0.9898 - val_loss: 0.0141 - val_categorical_accuracy: 0.9642
Epoch 76/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0082 - categorical_accuracy: 0.9898 - val_loss: 0.0140 - val_categorical_accuracy: 0.9635
Epoch 77/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0082 - categorical_accuracy: 0.9893 - val_loss: 0.0140 - val_categorical_accuracy: 0.9629
Epoch 78/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0080 - categorical_accuracy: 0.9895 - val_loss: 0.0144 - val_categorical_accuracy: 0.9628
Epoch 79/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0081 - categorical_accuracy: 0.9888 - val_loss: 0.0149 - val_categorical_accuracy: 0.9595
Epoch 80/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0080 - categorical_accuracy: 0.9890 - val_loss: 0.0145 - val_categorical_accuracy: 0.9613
Epoch 81/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0078 - categorical_accuracy: 0.9899 - val_loss: 0.0146 - val_categorical_accuracy: 0.9606
Epoch 82/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0078 - categorical_accuracy: 0.9893 - val_loss: 0.0141 - val_categorical_accuracy: 0.9627
Epoch 83/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0078 - categorical_accuracy: 0.9889 - val_loss: 0.0139 - val_categorical_accuracy: 0.9610

Epoch 00083: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 84/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0073 - categorical_accuracy: 0.9921 - val_loss: 0.0130 - val_categorical_accuracy: 0.9658
Epoch 85/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0070 - categorical_accuracy: 0.9936 - val_loss: 0.0129 - val_categorical_accuracy: 0.9662
Epoch 86/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0068 - categorical_accuracy: 0.9944 - val_loss: 0.0131 - val_categorical_accuracy: 0.9662
Epoch 87/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0067 - categorical_accuracy: 0.9951 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674
Epoch 88/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0067 - categorical_accuracy: 0.9947 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674
Epoch 89/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9951 - val_loss: 0.0128 - val_categorical_accuracy: 0.9671
Epoch 90/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0065 - categorical_accuracy: 0.9958 - val_loss: 0.0129 - val_categorical_accuracy: 0.9665
Epoch 91/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0064 - categorical_accuracy: 0.9957 - val_loss: 0.0128 - val_categorical_accuracy: 0.9668
Epoch 92/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0064 - categorical_accuracy: 0.9960 - val_loss: 0.0128 - val_categorical_accuracy: 0.9665
Epoch 93/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0063 - categorical_accuracy: 0.9961 - val_loss: 0.0127 - val_categorical_accuracy: 0.9674

Epoch 00093: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 94/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9967 - val_loss: 0.0127 - val_categorical_accuracy: 0.9680
Epoch 95/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9961 - val_loss: 0.0128 - val_categorical_accuracy: 0.9674
Epoch 96/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9964 - val_loss: 0.0127 - val_categorical_accuracy: 0.9677
Epoch 97/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9966 - val_loss: 0.0128 - val_categorical_accuracy: 0.9679
Epoch 98/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9968 - val_loss: 0.0127 - val_categorical_accuracy: 0.9680

Epoch 00098: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 99/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0062 - categorical_accuracy: 0.9965 - val_loss: 0.0127 - val_categorical_accuracy: 0.9683
Epoch 100/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9968 - val_loss: 0.0127 - val_categorical_accuracy: 0.9680
Epoch 101/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9967 - val_loss: 0.0127 - val_categorical_accuracy: 0.9680
Epoch 102/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9971 - val_loss: 0.0127 - val_categorical_accuracy: 0.9679
Epoch 103/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0061 - categorical_accuracy: 0.9970 - val_loss: 0.0127 - val_categorical_accuracy: 0.9680

Epoch 00103: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 00103: early stopping
========= generating oof predictions 13:01:18 =========
========= generating test set predictions 13:01:22 =========
========= fitting 4 th model 13:02:10 =========
Epoch 1/65535
420/420 [==============================] - 57s 135ms/step - loss: 0.5331 - categorical_accuracy: 0.1956 - val_loss: 0.4696 - val_categorical_accuracy: 0.3995
Epoch 2/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.4241 - categorical_accuracy: 0.5160 - val_loss: 0.3798 - val_categorical_accuracy: 0.6255
Epoch 3/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.3444 - categorical_accuracy: 0.7033 - val_loss: 0.3107 - val_categorical_accuracy: 0.7682
Epoch 4/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.2866 - categorical_accuracy: 0.7877 - val_loss: 0.2620 - val_categorical_accuracy: 0.8196
Epoch 5/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.2418 - categorical_accuracy: 0.8351 - val_loss: 0.2221 - val_categorical_accuracy: 0.8549
Epoch 6/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.2058 - categorical_accuracy: 0.8639 - val_loss: 0.1874 - val_categorical_accuracy: 0.8884
Epoch 7/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.1762 - categorical_accuracy: 0.8794 - val_loss: 0.1627 - val_categorical_accuracy: 0.8919
Epoch 8/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.1512 - categorical_accuracy: 0.8952 - val_loss: 0.1400 - val_categorical_accuracy: 0.9025
Epoch 9/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.1306 - categorical_accuracy: 0.9047 - val_loss: 0.1218 - val_categorical_accuracy: 0.9058
Epoch 10/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.1132 - categorical_accuracy: 0.9137 - val_loss: 0.1052 - val_categorical_accuracy: 0.9182
Epoch 11/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0986 - categorical_accuracy: 0.9188 - val_loss: 0.0926 - val_categorical_accuracy: 0.9200
Epoch 12/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0862 - categorical_accuracy: 0.9252 - val_loss: 0.0809 - val_categorical_accuracy: 0.9273
Epoch 13/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0761 - categorical_accuracy: 0.9280 - val_loss: 0.0717 - val_categorical_accuracy: 0.9302
Epoch 14/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0675 - categorical_accuracy: 0.9317 - val_loss: 0.0636 - val_categorical_accuracy: 0.9352
Epoch 15/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0602 - categorical_accuracy: 0.9347 - val_loss: 0.0666 - val_categorical_accuracy: 0.8864
Epoch 16/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0541 - categorical_accuracy: 0.9373 - val_loss: 0.0580 - val_categorical_accuracy: 0.9030
Epoch 17/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0492 - categorical_accuracy: 0.9380 - val_loss: 0.0572 - val_categorical_accuracy: 0.8860
Epoch 18/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0447 - categorical_accuracy: 0.9402 - val_loss: 0.0458 - val_categorical_accuracy: 0.9249
Epoch 19/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0411 - categorical_accuracy: 0.9406 - val_loss: 0.0418 - val_categorical_accuracy: 0.9340
Epoch 20/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0382 - categorical_accuracy: 0.9425 - val_loss: 0.0411 - val_categorical_accuracy: 0.9204
Epoch 21/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0356 - categorical_accuracy: 0.9431 - val_loss: 0.0372 - val_categorical_accuracy: 0.9296
Epoch 22/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0333 - categorical_accuracy: 0.9452 - val_loss: 0.0357 - val_categorical_accuracy: 0.9279
Epoch 23/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0311 - categorical_accuracy: 0.9470 - val_loss: 0.0323 - val_categorical_accuracy: 0.9378
Epoch 24/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0298 - categorical_accuracy: 0.9473 - val_loss: 0.0350 - val_categorical_accuracy: 0.9203
Epoch 25/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0287 - categorical_accuracy: 0.9465 - val_loss: 0.0347 - val_categorical_accuracy: 0.9172
Epoch 26/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0274 - categorical_accuracy: 0.9482 - val_loss: 0.0311 - val_categorical_accuracy: 0.9261
Epoch 27/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0264 - categorical_accuracy: 0.9491 - val_loss: 0.0293 - val_categorical_accuracy: 0.9346
Epoch 28/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0255 - categorical_accuracy: 0.9491 - val_loss: 0.0281 - val_categorical_accuracy: 0.9372
Epoch 29/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0247 - categorical_accuracy: 0.9503 - val_loss: 0.0262 - val_categorical_accuracy: 0.9434
Epoch 30/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0242 - categorical_accuracy: 0.9510 - val_loss: 0.0255 - val_categorical_accuracy: 0.9440
Epoch 31/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0237 - categorical_accuracy: 0.9513 - val_loss: 0.0246 - val_categorical_accuracy: 0.9465
Epoch 32/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0231 - categorical_accuracy: 0.9523 - val_loss: 0.0295 - val_categorical_accuracy: 0.9239
Epoch 33/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0230 - categorical_accuracy: 0.9512 - val_loss: 0.0274 - val_categorical_accuracy: 0.9338
Epoch 34/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0225 - categorical_accuracy: 0.9530 - val_loss: 0.0275 - val_categorical_accuracy: 0.9323
Epoch 35/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0222 - categorical_accuracy: 0.9531 - val_loss: 0.0270 - val_categorical_accuracy: 0.9296
Epoch 36/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0219 - categorical_accuracy: 0.9536 - val_loss: 0.0252 - val_categorical_accuracy: 0.9372
Epoch 37/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0214 - categorical_accuracy: 0.9555 - val_loss: 0.0270 - val_categorical_accuracy: 0.9301

Epoch 00037: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 38/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0183 - categorical_accuracy: 0.9715 - val_loss: 0.0202 - val_categorical_accuracy: 0.9615
Epoch 39/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0170 - categorical_accuracy: 0.9774 - val_loss: 0.0195 - val_categorical_accuracy: 0.9624
Epoch 40/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0163 - categorical_accuracy: 0.9792 - val_loss: 0.0191 - val_categorical_accuracy: 0.9623
Epoch 41/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0158 - categorical_accuracy: 0.9801 - val_loss: 0.0189 - val_categorical_accuracy: 0.9632
Epoch 42/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0152 - categorical_accuracy: 0.9817 - val_loss: 0.0183 - val_categorical_accuracy: 0.9638
Epoch 43/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0147 - categorical_accuracy: 0.9821 - val_loss: 0.0183 - val_categorical_accuracy: 0.9629
Epoch 44/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0142 - categorical_accuracy: 0.9830 - val_loss: 0.0177 - val_categorical_accuracy: 0.9637
Epoch 45/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0138 - categorical_accuracy: 0.9837 - val_loss: 0.0175 - val_categorical_accuracy: 0.9646
Epoch 46/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0134 - categorical_accuracy: 0.9845 - val_loss: 0.0172 - val_categorical_accuracy: 0.9645
Epoch 47/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0130 - categorical_accuracy: 0.9848 - val_loss: 0.0170 - val_categorical_accuracy: 0.9641
Epoch 48/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0126 - categorical_accuracy: 0.9854 - val_loss: 0.0166 - val_categorical_accuracy: 0.9642
Epoch 49/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0124 - categorical_accuracy: 0.9851 - val_loss: 0.0166 - val_categorical_accuracy: 0.9635
Epoch 50/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0120 - categorical_accuracy: 0.9861 - val_loss: 0.0163 - val_categorical_accuracy: 0.9636
Epoch 51/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0117 - categorical_accuracy: 0.9859 - val_loss: 0.0160 - val_categorical_accuracy: 0.9641
Epoch 52/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0113 - categorical_accuracy: 0.9866 - val_loss: 0.0157 - val_categorical_accuracy: 0.9658
Epoch 53/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0112 - categorical_accuracy: 0.9861 - val_loss: 0.0156 - val_categorical_accuracy: 0.9644
Epoch 54/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0109 - categorical_accuracy: 0.9868 - val_loss: 0.0155 - val_categorical_accuracy: 0.9641
Epoch 55/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0106 - categorical_accuracy: 0.9875 - val_loss: 0.0153 - val_categorical_accuracy: 0.9648
Epoch 56/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0104 - categorical_accuracy: 0.9867 - val_loss: 0.0153 - val_categorical_accuracy: 0.9630
Epoch 57/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0101 - categorical_accuracy: 0.9874 - val_loss: 0.0148 - val_categorical_accuracy: 0.9644
Epoch 58/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0100 - categorical_accuracy: 0.9876 - val_loss: 0.0150 - val_categorical_accuracy: 0.9636
Epoch 59/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0098 - categorical_accuracy: 0.9873 - val_loss: 0.0150 - val_categorical_accuracy: 0.9617
Epoch 60/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0096 - categorical_accuracy: 0.9876 - val_loss: 0.0145 - val_categorical_accuracy: 0.9628
Epoch 61/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0095 - categorical_accuracy: 0.9877 - val_loss: 0.0145 - val_categorical_accuracy: 0.9639
Epoch 62/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0093 - categorical_accuracy: 0.9878 - val_loss: 0.0147 - val_categorical_accuracy: 0.9633
Epoch 63/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0091 - categorical_accuracy: 0.9877 - val_loss: 0.0145 - val_categorical_accuracy: 0.9624
Epoch 64/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0090 - categorical_accuracy: 0.9876 - val_loss: 0.0145 - val_categorical_accuracy: 0.9636
Epoch 65/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0089 - categorical_accuracy: 0.9875 - val_loss: 0.0149 - val_categorical_accuracy: 0.9588
Epoch 66/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0090 - categorical_accuracy: 0.9863 - val_loss: 0.0141 - val_categorical_accuracy: 0.9636
Epoch 67/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0088 - categorical_accuracy: 0.9866 - val_loss: 0.0141 - val_categorical_accuracy: 0.9618
Epoch 68/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0087 - categorical_accuracy: 0.9867 - val_loss: 0.0138 - val_categorical_accuracy: 0.9625
Epoch 69/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0085 - categorical_accuracy: 0.9874 - val_loss: 0.0142 - val_categorical_accuracy: 0.9614
Epoch 70/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0086 - categorical_accuracy: 0.9868 - val_loss: 0.0154 - val_categorical_accuracy: 0.9574
Epoch 71/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0084 - categorical_accuracy: 0.9878 - val_loss: 0.0147 - val_categorical_accuracy: 0.9595
Epoch 72/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0084 - categorical_accuracy: 0.9871 - val_loss: 0.0138 - val_categorical_accuracy: 0.9618
Epoch 73/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0083 - categorical_accuracy: 0.9874 - val_loss: 0.0140 - val_categorical_accuracy: 0.9609
Epoch 74/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0083 - categorical_accuracy: 0.9867 - val_loss: 0.0142 - val_categorical_accuracy: 0.9589

Epoch 00074: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 75/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0076 - categorical_accuracy: 0.9910 - val_loss: 0.0131 - val_categorical_accuracy: 0.9659
Epoch 76/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0073 - categorical_accuracy: 0.9924 - val_loss: 0.0128 - val_categorical_accuracy: 0.9672
Epoch 77/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0072 - categorical_accuracy: 0.9927 - val_loss: 0.0128 - val_categorical_accuracy: 0.9673
Epoch 78/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0070 - categorical_accuracy: 0.9939 - val_loss: 0.0129 - val_categorical_accuracy: 0.9668
Epoch 79/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0070 - categorical_accuracy: 0.9937 - val_loss: 0.0130 - val_categorical_accuracy: 0.9672
Epoch 80/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0069 - categorical_accuracy: 0.9943 - val_loss: 0.0129 - val_categorical_accuracy: 0.9673
Epoch 81/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9941 - val_loss: 0.0130 - val_categorical_accuracy: 0.9668
Epoch 82/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0068 - categorical_accuracy: 0.9949 - val_loss: 0.0128 - val_categorical_accuracy: 0.9669

Epoch 00082: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 83/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0067 - categorical_accuracy: 0.9945 - val_loss: 0.0128 - val_categorical_accuracy: 0.9674
Epoch 84/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0066 - categorical_accuracy: 0.9955 - val_loss: 0.0128 - val_categorical_accuracy: 0.9674
Epoch 85/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9951 - val_loss: 0.0128 - val_categorical_accuracy: 0.9674
Epoch 86/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9951 - val_loss: 0.0128 - val_categorical_accuracy: 0.9675
Epoch 87/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9950 - val_loss: 0.0128 - val_categorical_accuracy: 0.9680

Epoch 00087: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 88/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9956 - val_loss: 0.0128 - val_categorical_accuracy: 0.9679
Epoch 89/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0065 - categorical_accuracy: 0.9959 - val_loss: 0.0128 - val_categorical_accuracy: 0.9675
Epoch 90/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9954 - val_loss: 0.0128 - val_categorical_accuracy: 0.9678
Epoch 91/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0065 - categorical_accuracy: 0.9955 - val_loss: 0.0128 - val_categorical_accuracy: 0.9677
Epoch 92/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0065 - categorical_accuracy: 0.9958 - val_loss: 0.0128 - val_categorical_accuracy: 0.9675

Epoch 00092: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 93/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9950 - val_loss: 0.0128 - val_categorical_accuracy: 0.9676
Epoch 94/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0066 - categorical_accuracy: 0.9951 - val_loss: 0.0128 - val_categorical_accuracy: 0.9675
Epoch 95/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0065 - categorical_accuracy: 0.9958 - val_loss: 0.0128 - val_categorical_accuracy: 0.9674
Epoch 96/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0065 - categorical_accuracy: 0.9960 - val_loss: 0.0128 - val_categorical_accuracy: 0.9673
Epoch 97/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0065 - categorical_accuracy: 0.9956 - val_loss: 0.0128 - val_categorical_accuracy: 0.9674

Epoch 00097: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 98/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0066 - categorical_accuracy: 0.9955 - val_loss: 0.0128 - val_categorical_accuracy: 0.9677
Epoch 99/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0066 - categorical_accuracy: 0.9956 - val_loss: 0.0128 - val_categorical_accuracy: 0.9675
Epoch 00099: early stopping
========= generating oof predictions 14:34:17 =========
========= generating test set predictions 14:34:21 =========
========= fitting 5 th model 14:35:09 =========
Epoch 1/65535
420/420 [==============================] - 57s 135ms/step - loss: 0.5383 - categorical_accuracy: 0.1625 - val_loss: 0.4779 - val_categorical_accuracy: 0.3346
Epoch 2/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.4290 - categorical_accuracy: 0.4863 - val_loss: 0.3771 - val_categorical_accuracy: 0.6482
Epoch 3/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.3451 - categorical_accuracy: 0.6995 - val_loss: 0.3087 - val_categorical_accuracy: 0.7778
Epoch 4/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.2861 - categorical_accuracy: 0.7926 - val_loss: 0.2592 - val_categorical_accuracy: 0.8328
Epoch 5/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.2415 - categorical_accuracy: 0.8358 - val_loss: 0.2173 - val_categorical_accuracy: 0.8805
Epoch 6/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.2053 - categorical_accuracy: 0.8655 - val_loss: 0.1874 - val_categorical_accuracy: 0.8882
Epoch 7/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.1756 - categorical_accuracy: 0.8844 - val_loss: 0.1596 - val_categorical_accuracy: 0.9081
Epoch 8/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.1507 - categorical_accuracy: 0.8999 - val_loss: 0.1374 - val_categorical_accuracy: 0.9135
Epoch 9/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.1301 - categorical_accuracy: 0.9075 - val_loss: 0.1208 - val_categorical_accuracy: 0.9123
Epoch 10/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.1127 - categorical_accuracy: 0.9169 - val_loss: 0.1051 - val_categorical_accuracy: 0.9175
Epoch 11/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0984 - categorical_accuracy: 0.9200 - val_loss: 0.0908 - val_categorical_accuracy: 0.9300
Epoch 12/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0859 - categorical_accuracy: 0.9265 - val_loss: 0.0819 - val_categorical_accuracy: 0.9209
Epoch 13/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0759 - categorical_accuracy: 0.9293 - val_loss: 0.0721 - val_categorical_accuracy: 0.9284
Epoch 14/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0673 - categorical_accuracy: 0.9318 - val_loss: 0.0649 - val_categorical_accuracy: 0.9258
Epoch 15/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0599 - categorical_accuracy: 0.9358 - val_loss: 0.0583 - val_categorical_accuracy: 0.9286
Epoch 16/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0538 - categorical_accuracy: 0.9379 - val_loss: 0.0524 - val_categorical_accuracy: 0.9333
Epoch 17/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0484 - categorical_accuracy: 0.9411 - val_loss: 0.0502 - val_categorical_accuracy: 0.9205
Epoch 18/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0445 - categorical_accuracy: 0.9413 - val_loss: 0.0456 - val_categorical_accuracy: 0.9268
Epoch 19/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0408 - categorical_accuracy: 0.9429 - val_loss: 0.0411 - val_categorical_accuracy: 0.9331
Epoch 20/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0376 - categorical_accuracy: 0.9454 - val_loss: 0.0388 - val_categorical_accuracy: 0.9328
Epoch 21/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0353 - categorical_accuracy: 0.9444 - val_loss: 0.0355 - val_categorical_accuracy: 0.9405
Epoch 22/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0332 - categorical_accuracy: 0.9444 - val_loss: 0.0359 - val_categorical_accuracy: 0.9291
Epoch 23/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0310 - categorical_accuracy: 0.9477 - val_loss: 0.0340 - val_categorical_accuracy: 0.9329
Epoch 24/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0295 - categorical_accuracy: 0.9484 - val_loss: 0.0360 - val_categorical_accuracy: 0.9163
Epoch 25/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0281 - categorical_accuracy: 0.9491 - val_loss: 0.0304 - val_categorical_accuracy: 0.9370
Epoch 26/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0272 - categorical_accuracy: 0.9488 - val_loss: 0.0409 - val_categorical_accuracy: 0.8811
Epoch 27/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0263 - categorical_accuracy: 0.9490 - val_loss: 0.0291 - val_categorical_accuracy: 0.9331
Epoch 28/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0256 - categorical_accuracy: 0.9499 - val_loss: 0.0281 - val_categorical_accuracy: 0.9346
Epoch 29/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0249 - categorical_accuracy: 0.9492 - val_loss: 0.0316 - val_categorical_accuracy: 0.9194
Epoch 30/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0239 - categorical_accuracy: 0.9526 - val_loss: 0.0295 - val_categorical_accuracy: 0.9257
Epoch 31/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0236 - categorical_accuracy: 0.9515 - val_loss: 0.0283 - val_categorical_accuracy: 0.9312
Epoch 32/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0232 - categorical_accuracy: 0.9512 - val_loss: 0.0276 - val_categorical_accuracy: 0.9293
Epoch 33/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0228 - categorical_accuracy: 0.9513 - val_loss: 0.0253 - val_categorical_accuracy: 0.9399
Epoch 34/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0220 - categorical_accuracy: 0.9541 - val_loss: 0.0250 - val_categorical_accuracy: 0.9389
Epoch 35/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0219 - categorical_accuracy: 0.9533 - val_loss: 0.0243 - val_categorical_accuracy: 0.9422
Epoch 36/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0217 - categorical_accuracy: 0.9537 - val_loss: 0.0273 - val_categorical_accuracy: 0.9238
Epoch 37/65535
420/420 [==============================] - 57s 137ms/step - loss: 0.0214 - categorical_accuracy: 0.9542 - val_loss: 0.0236 - val_categorical_accuracy: 0.9454
Epoch 38/65535
420/420 [==============================] - 57s 137ms/step - loss: 0.0211 - categorical_accuracy: 0.9556 - val_loss: 0.0341 - val_categorical_accuracy: 0.8951
Epoch 39/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0211 - categorical_accuracy: 0.9544 - val_loss: 0.0252 - val_categorical_accuracy: 0.9355
Epoch 40/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0208 - categorical_accuracy: 0.9557 - val_loss: 0.0238 - val_categorical_accuracy: 0.9410
Epoch 41/65535
420/420 [==============================] - 57s 135ms/step - loss: 0.0206 - categorical_accuracy: 0.9560 - val_loss: 0.0269 - val_categorical_accuracy: 0.9272
Epoch 42/65535
420/420 [==============================] - 57s 136ms/step - loss: 0.0206 - categorical_accuracy: 0.9562 - val_loss: 0.0264 - val_categorical_accuracy: 0.9301
Epoch 43/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0205 - categorical_accuracy: 0.9570 - val_loss: 0.0265 - val_categorical_accuracy: 0.9308

Epoch 00043: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 44/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0174 - categorical_accuracy: 0.9737 - val_loss: 0.0197 - val_categorical_accuracy: 0.9600
Epoch 45/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0161 - categorical_accuracy: 0.9789 - val_loss: 0.0190 - val_categorical_accuracy: 0.9623
Epoch 46/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0154 - categorical_accuracy: 0.9803 - val_loss: 0.0186 - val_categorical_accuracy: 0.9632
Epoch 47/65535
420/420 [==============================] - 57s 136ms/step - loss: 0.0149 - categorical_accuracy: 0.9817 - val_loss: 0.0183 - val_categorical_accuracy: 0.9634
Epoch 48/65535
420/420 [==============================] - 56s 134ms/step - loss: 0.0143 - categorical_accuracy: 0.9829 - val_loss: 0.0186 - val_categorical_accuracy: 0.9625
Epoch 49/65535
420/420 [==============================] - 57s 136ms/step - loss: 0.0139 - categorical_accuracy: 0.9831 - val_loss: 0.0176 - val_categorical_accuracy: 0.9633
Epoch 50/65535
420/420 [==============================] - 58s 139ms/step - loss: 0.0135 - categorical_accuracy: 0.9835 - val_loss: 0.0174 - val_categorical_accuracy: 0.9641
Epoch 51/65535
420/420 [==============================] - 58s 139ms/step - loss: 0.0131 - categorical_accuracy: 0.9844 - val_loss: 0.0175 - val_categorical_accuracy: 0.9615
Epoch 52/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0126 - categorical_accuracy: 0.9854 - val_loss: 0.0173 - val_categorical_accuracy: 0.9626
Epoch 53/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0124 - categorical_accuracy: 0.9849 - val_loss: 0.0174 - val_categorical_accuracy: 0.9616
Epoch 54/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0120 - categorical_accuracy: 0.9854 - val_loss: 0.0168 - val_categorical_accuracy: 0.9630
Epoch 55/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0117 - categorical_accuracy: 0.9860 - val_loss: 0.0164 - val_categorical_accuracy: 0.9632
Epoch 56/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0114 - categorical_accuracy: 0.9866 - val_loss: 0.0163 - val_categorical_accuracy: 0.9635
Epoch 57/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0111 - categorical_accuracy: 0.9871 - val_loss: 0.0162 - val_categorical_accuracy: 0.9630
Epoch 58/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0108 - categorical_accuracy: 0.9870 - val_loss: 0.0156 - val_categorical_accuracy: 0.9653
Epoch 59/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0105 - categorical_accuracy: 0.9880 - val_loss: 0.0159 - val_categorical_accuracy: 0.9620
Epoch 60/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0103 - categorical_accuracy: 0.9871 - val_loss: 0.0158 - val_categorical_accuracy: 0.9615
Epoch 61/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0101 - categorical_accuracy: 0.9879 - val_loss: 0.0154 - val_categorical_accuracy: 0.9630
Epoch 62/65535
420/420 [==============================] - 56s 134ms/step - loss: 0.0099 - categorical_accuracy: 0.9881 - val_loss: 0.0155 - val_categorical_accuracy: 0.9628
Epoch 63/65535
420/420 [==============================] - 57s 135ms/step - loss: 0.0097 - categorical_accuracy: 0.9872 - val_loss: 0.0152 - val_categorical_accuracy: 0.9626
Epoch 64/65535
420/420 [==============================] - 58s 137ms/step - loss: 0.0094 - categorical_accuracy: 0.9886 - val_loss: 0.0152 - val_categorical_accuracy: 0.9613
Epoch 65/65535
420/420 [==============================] - 57s 136ms/step - loss: 0.0093 - categorical_accuracy: 0.9883 - val_loss: 0.0155 - val_categorical_accuracy: 0.9604
Epoch 66/65535
420/420 [==============================] - 56s 134ms/step - loss: 0.0091 - categorical_accuracy: 0.9882 - val_loss: 0.0151 - val_categorical_accuracy: 0.9621
Epoch 67/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0091 - categorical_accuracy: 0.9876 - val_loss: 0.0143 - val_categorical_accuracy: 0.9634
Epoch 68/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0089 - categorical_accuracy: 0.9881 - val_loss: 0.0153 - val_categorical_accuracy: 0.9589
Epoch 69/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0086 - categorical_accuracy: 0.9893 - val_loss: 0.0153 - val_categorical_accuracy: 0.9585
Epoch 70/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0084 - categorical_accuracy: 0.9895 - val_loss: 0.0145 - val_categorical_accuracy: 0.9616
Epoch 71/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0084 - categorical_accuracy: 0.9886 - val_loss: 0.0147 - val_categorical_accuracy: 0.9597
Epoch 72/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0083 - categorical_accuracy: 0.9885 - val_loss: 0.0145 - val_categorical_accuracy: 0.9611
Epoch 73/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0082 - categorical_accuracy: 0.9884 - val_loss: 0.0148 - val_categorical_accuracy: 0.9585

Epoch 00073: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 74/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0077 - categorical_accuracy: 0.9916 - val_loss: 0.0136 - val_categorical_accuracy: 0.9647
Epoch 75/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0075 - categorical_accuracy: 0.9924 - val_loss: 0.0136 - val_categorical_accuracy: 0.9635
Epoch 76/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0073 - categorical_accuracy: 0.9935 - val_loss: 0.0137 - val_categorical_accuracy: 0.9644
Epoch 77/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0072 - categorical_accuracy: 0.9935 - val_loss: 0.0135 - val_categorical_accuracy: 0.9650
Epoch 78/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0072 - categorical_accuracy: 0.9941 - val_loss: 0.0136 - val_categorical_accuracy: 0.9644
Epoch 79/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0071 - categorical_accuracy: 0.9941 - val_loss: 0.0137 - val_categorical_accuracy: 0.9655
Epoch 80/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0070 - categorical_accuracy: 0.9943 - val_loss: 0.0136 - val_categorical_accuracy: 0.9662

Epoch 00080: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 81/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0069 - categorical_accuracy: 0.9948 - val_loss: 0.0134 - val_categorical_accuracy: 0.9671
Epoch 82/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0070 - categorical_accuracy: 0.9947 - val_loss: 0.0134 - val_categorical_accuracy: 0.9669
Epoch 83/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0069 - categorical_accuracy: 0.9949 - val_loss: 0.0134 - val_categorical_accuracy: 0.9670
Epoch 84/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0069 - categorical_accuracy: 0.9950 - val_loss: 0.0135 - val_categorical_accuracy: 0.9664
Epoch 85/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9956 - val_loss: 0.0135 - val_categorical_accuracy: 0.9665
Epoch 86/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0069 - categorical_accuracy: 0.9948 - val_loss: 0.0135 - val_categorical_accuracy: 0.9659
Epoch 87/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9949 - val_loss: 0.0134 - val_categorical_accuracy: 0.9667

Epoch 00087: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 88/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9954 - val_loss: 0.0134 - val_categorical_accuracy: 0.9667
Epoch 89/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0068 - categorical_accuracy: 0.9955 - val_loss: 0.0134 - val_categorical_accuracy: 0.9669
Epoch 90/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9957 - val_loss: 0.0134 - val_categorical_accuracy: 0.9668
Epoch 91/65535
420/420 [==============================] - 56s 133ms/step - loss: 0.0068 - categorical_accuracy: 0.9951 - val_loss: 0.0134 - val_categorical_accuracy: 0.9670
Epoch 92/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9953 - val_loss: 0.0134 - val_categorical_accuracy: 0.9668

Epoch 00092: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 93/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9948 - val_loss: 0.0134 - val_categorical_accuracy: 0.9670
Epoch 94/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9950 - val_loss: 0.0134 - val_categorical_accuracy: 0.9670
Epoch 95/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9951 - val_loss: 0.0134 - val_categorical_accuracy: 0.9672
Epoch 96/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9954 - val_loss: 0.0134 - val_categorical_accuracy: 0.9670
Epoch 97/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9953 - val_loss: 0.0134 - val_categorical_accuracy: 0.9671

Epoch 00097: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 98/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9951 - val_loss: 0.0134 - val_categorical_accuracy: 0.9670
Epoch 99/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9952 - val_loss: 0.0134 - val_categorical_accuracy: 0.9673
Epoch 100/65535
420/420 [==============================] - 55s 131ms/step - loss: 0.0068 - categorical_accuracy: 0.9955 - val_loss: 0.0134 - val_categorical_accuracy: 0.9673
Epoch 101/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9952 - val_loss: 0.0134 - val_categorical_accuracy: 0.9670
Epoch 102/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9955 - val_loss: 0.0134 - val_categorical_accuracy: 0.9670

Epoch 00102: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 103/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9951 - val_loss: 0.0134 - val_categorical_accuracy: 0.9673
Epoch 104/65535
420/420 [==============================] - 55s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9955 - val_loss: 0.0134 - val_categorical_accuracy: 0.9671
Epoch 105/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9950 - val_loss: 0.0134 - val_categorical_accuracy: 0.9672
Epoch 106/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9951 - val_loss: 0.0134 - val_categorical_accuracy: 0.9672
Epoch 107/65535
420/420 [==============================] - 56s 132ms/step - loss: 0.0068 - categorical_accuracy: 0.9951 - val_loss: 0.0134 - val_categorical_accuracy: 0.9670

Epoch 00107: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 00107: early stopping
========= generating oof predictions 16:14:52 =========
========= generating test set predictions 16:14:56 =========
train loss avg 0.006290070815386471 -- std 0.0003334396241429092, val loss avg 0.012940854520369244 -- std 0.00028581065269909475
train acc avg 0.9963090222291484 -- std 0.0008564099840524098, val acc avg 0.9676850166644947 -- std 0.0004141639756710654
mean nb epochs 118.0
dump oof predicted probs
dump test set predicted probs
