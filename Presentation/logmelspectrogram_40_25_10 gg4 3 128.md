ren (master *+) python $ python train.py logmelspectrogram_40_25_10 gg4 3 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (101, 40) ===========
------------- SUMMARY OF MODEL -------------
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 101, 40)      0
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 101, 40, 1)   0           input_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 101, 40, 18)  198         reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 101, 40, 18)  72          conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 101, 40, 18)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 101, 40, 18)  3258        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 101, 40, 18)  72          conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 101, 40, 18)  0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 50, 20, 18)   0           activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 50, 20, 36)   6516        max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 50, 20, 36)   144         conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 50, 20, 36)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 50, 20, 36)   12996       activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 50, 20, 36)   144         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 50, 20, 36)   0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 25, 10, 36)   0           activation_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 25, 10, 72)   25992       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 25, 10, 72)   288         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 25, 10, 72)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 25, 10, 72)   51912       activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 25, 10, 72)   288         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 25, 10, 72)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 12, 5, 72)    0           activation_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 12, 5, 144)   103824      max_pooling2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 12, 5, 144)   576         conv2d_7[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 12, 5, 144)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 12, 5, 144)   207504      activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 12, 5, 144)   576         conv2d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 12, 5, 144)   0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 6, 2, 144)    0           activation_8[0][0]
__________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalM (None, 144)          0           max_pooling2d_4[0][0]
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 144)          0           max_pooling2d_4[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 288)          0           global_max_pooling2d_1[0][0]
                                                                 global_average_pooling2d_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          36992       concatenate_1[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 128)          512         dense_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 128)          0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 128)          16512       dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 128)          512         dense_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 128)          0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 31)           3999        dropout_2[0][0]
==================================================================================================
Total params: 472,887
Trainable params: 471,295
Non-trainable params: 1,592
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 09:34:34 =========
Epoch 1/65535
420/420 [==============================] - 25s 60ms/step - loss: 0.1008 - categorical_accuracy: 0.3884 - val_loss: 0.0369 - val_categorical_accuracy: 0.7970
Epoch 2/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0352 - categorical_accuracy: 0.8056 - val_loss: 0.0188 - val_categorical_accuracy: 0.9052
Epoch 3/65535
420/420 [==============================] - 19s 46ms/step - loss: 0.0224 - categorical_accuracy: 0.8837 - val_loss: 0.0157 - val_categorical_accuracy: 0.9128
Epoch 4/65535
420/420 [==============================] - 19s 46ms/step - loss: 0.0176 - categorical_accuracy: 0.9076 - val_loss: 0.0143 - val_categorical_accuracy: 0.9170
Epoch 5/65535
420/420 [==============================] - 19s 46ms/step - loss: 0.0146 - categorical_accuracy: 0.9253 - val_loss: 0.0136 - val_categorical_accuracy: 0.9256
Epoch 6/65535
420/420 [==============================] - 19s 46ms/step - loss: 0.0126 - categorical_accuracy: 0.9354 - val_loss: 0.0113 - val_categorical_accuracy: 0.9417
Epoch 7/65535
420/420 [==============================] - 20s 46ms/step - loss: 0.0111 - categorical_accuracy: 0.9424 - val_loss: 0.0094 - val_categorical_accuracy: 0.9487
Epoch 8/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0098 - categorical_accuracy: 0.9501 - val_loss: 0.0083 - val_categorical_accuracy: 0.9538
Epoch 9/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0089 - categorical_accuracy: 0.9540 - val_loss: 0.0087 - val_categorical_accuracy: 0.9541
Epoch 10/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0083 - categorical_accuracy: 0.9578 - val_loss: 0.0080 - val_categorical_accuracy: 0.9562
Epoch 11/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0075 - categorical_accuracy: 0.9608 - val_loss: 0.0095 - val_categorical_accuracy: 0.9498
Epoch 12/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0069 - categorical_accuracy: 0.9652 - val_loss: 0.0084 - val_categorical_accuracy: 0.9554
Epoch 13/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0064 - categorical_accuracy: 0.9683 - val_loss: 0.0077 - val_categorical_accuracy: 0.9583
Epoch 14/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0058 - categorical_accuracy: 0.9706 - val_loss: 0.0096 - val_categorical_accuracy: 0.9535
Epoch 15/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0055 - categorical_accuracy: 0.9725 - val_loss: 0.0084 - val_categorical_accuracy: 0.9585
Epoch 16/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0050 - categorical_accuracy: 0.9747 - val_loss: 0.0078 - val_categorical_accuracy: 0.9604
Epoch 17/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0047 - categorical_accuracy: 0.9763 - val_loss: 0.0076 - val_categorical_accuracy: 0.9614
Epoch 18/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0044 - categorical_accuracy: 0.9774 - val_loss: 0.0106 - val_categorical_accuracy: 0.9528
Epoch 19/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0038 - categorical_accuracy: 0.9800 - val_loss: 0.0131 - val_categorical_accuracy: 0.9470

Epoch 00019: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 20/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0029 - categorical_accuracy: 0.9858 - val_loss: 0.0066 - val_categorical_accuracy: 0.9684
Epoch 21/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0025 - categorical_accuracy: 0.9880 - val_loss: 0.0067 - val_categorical_accuracy: 0.9678
Epoch 22/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0022 - categorical_accuracy: 0.9898 - val_loss: 0.0068 - val_categorical_accuracy: 0.9684
Epoch 23/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0021 - categorical_accuracy: 0.9909 - val_loss: 0.0069 - val_categorical_accuracy: 0.9674
Epoch 24/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0020 - categorical_accuracy: 0.9906 - val_loss: 0.0067 - val_categorical_accuracy: 0.9690
Epoch 25/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0019 - categorical_accuracy: 0.9921 - val_loss: 0.0067 - val_categorical_accuracy: 0.9699
Epoch 26/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0018 - categorical_accuracy: 0.9924 - val_loss: 0.0069 - val_categorical_accuracy: 0.9688

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 27/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0016 - categorical_accuracy: 0.9931 - val_loss: 0.0069 - val_categorical_accuracy: 0.9687
Epoch 28/65535
420/420 [==============================] - 20s 49ms/step - loss: 0.0016 - categorical_accuracy: 0.9932 - val_loss: 0.0068 - val_categorical_accuracy: 0.9688
Epoch 29/65535
420/420 [==============================] - 22s 52ms/step - loss: 0.0016 - categorical_accuracy: 0.9936 - val_loss: 0.0068 - val_categorical_accuracy: 0.9687
Epoch 30/65535
420/420 [==============================] - 23s 55ms/step - loss: 0.0015 - categorical_accuracy: 0.9938 - val_loss: 0.0069 - val_categorical_accuracy: 0.9692
Epoch 31/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0015 - categorical_accuracy: 0.9935 - val_loss: 0.0070 - val_categorical_accuracy: 0.9692

Epoch 00031: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 32/65535
420/420 [==============================] - 23s 55ms/step - loss: 0.0016 - categorical_accuracy: 0.9934 - val_loss: 0.0069 - val_categorical_accuracy: 0.9691
Epoch 33/65535
420/420 [==============================] - 22s 54ms/step - loss: 0.0014 - categorical_accuracy: 0.9946 - val_loss: 0.0069 - val_categorical_accuracy: 0.9694
Epoch 34/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0015 - categorical_accuracy: 0.9939 - val_loss: 0.0069 - val_categorical_accuracy: 0.9693
Epoch 35/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0015 - categorical_accuracy: 0.9941 - val_loss: 0.0069 - val_categorical_accuracy: 0.9696
Epoch 00035: early stopping
========= generating oof predictions 09:46:37 =========
========= generating test set predictions 09:46:39 =========
========= fitting 2 th model 09:47:08 =========
Epoch 1/65535
420/420 [==============================] - 24s 56ms/step - loss: 0.0959 - categorical_accuracy: 0.4183 - val_loss: 0.0375 - val_categorical_accuracy: 0.7976
Epoch 2/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0337 - categorical_accuracy: 0.8158 - val_loss: 0.0228 - val_categorical_accuracy: 0.8785
Epoch 3/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0219 - categorical_accuracy: 0.8857 - val_loss: 0.0156 - val_categorical_accuracy: 0.9119
Epoch 4/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0167 - categorical_accuracy: 0.9147 - val_loss: 0.0125 - val_categorical_accuracy: 0.9324
Epoch 5/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0140 - categorical_accuracy: 0.9274 - val_loss: 0.0114 - val_categorical_accuracy: 0.9374
Epoch 6/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0121 - categorical_accuracy: 0.9393 - val_loss: 0.0120 - val_categorical_accuracy: 0.9361
Epoch 7/65535
420/420 [==============================] - 22s 54ms/step - loss: 0.0106 - categorical_accuracy: 0.9453 - val_loss: 0.0101 - val_categorical_accuracy: 0.9471
Epoch 8/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0096 - categorical_accuracy: 0.9511 - val_loss: 0.0117 - val_categorical_accuracy: 0.9387
Epoch 9/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0085 - categorical_accuracy: 0.9568 - val_loss: 0.0096 - val_categorical_accuracy: 0.9486
Epoch 10/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0078 - categorical_accuracy: 0.9598 - val_loss: 0.0123 - val_categorical_accuracy: 0.9337
Epoch 11/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0072 - categorical_accuracy: 0.9626 - val_loss: 0.0106 - val_categorical_accuracy: 0.9456
Epoch 12/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0065 - categorical_accuracy: 0.9668 - val_loss: 0.0082 - val_categorical_accuracy: 0.9568
Epoch 13/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0062 - categorical_accuracy: 0.9682 - val_loss: 0.0109 - val_categorical_accuracy: 0.9407
Epoch 14/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0056 - categorical_accuracy: 0.9713 - val_loss: 0.0083 - val_categorical_accuracy: 0.9557
Epoch 15/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0054 - categorical_accuracy: 0.9723 - val_loss: 0.0079 - val_categorical_accuracy: 0.9584
Epoch 16/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0049 - categorical_accuracy: 0.9751 - val_loss: 0.0089 - val_categorical_accuracy: 0.9554
Epoch 17/65535
420/420 [==============================] - 22s 52ms/step - loss: 0.0047 - categorical_accuracy: 0.9755 - val_loss: 0.0083 - val_categorical_accuracy: 0.9571
Epoch 18/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0042 - categorical_accuracy: 0.9786 - val_loss: 0.0081 - val_categorical_accuracy: 0.9595
Epoch 19/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0038 - categorical_accuracy: 0.9809 - val_loss: 0.0090 - val_categorical_accuracy: 0.9560
Epoch 20/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0036 - categorical_accuracy: 0.9815 - val_loss: 0.0101 - val_categorical_accuracy: 0.9528
Epoch 21/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0034 - categorical_accuracy: 0.9830 - val_loss: 0.0084 - val_categorical_accuracy: 0.9588

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 22/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0024 - categorical_accuracy: 0.9884 - val_loss: 0.0071 - val_categorical_accuracy: 0.9658
Epoch 23/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0020 - categorical_accuracy: 0.9911 - val_loss: 0.0069 - val_categorical_accuracy: 0.9669
Epoch 24/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0018 - categorical_accuracy: 0.9917 - val_loss: 0.0070 - val_categorical_accuracy: 0.9675
Epoch 25/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0017 - categorical_accuracy: 0.9930 - val_loss: 0.0071 - val_categorical_accuracy: 0.9664
Epoch 26/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0016 - categorical_accuracy: 0.9937 - val_loss: 0.0072 - val_categorical_accuracy: 0.9664
Epoch 27/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0014 - categorical_accuracy: 0.9942 - val_loss: 0.0070 - val_categorical_accuracy: 0.9672
Epoch 28/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0014 - categorical_accuracy: 0.9939 - val_loss: 0.0072 - val_categorical_accuracy: 0.9668
Epoch 29/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0013 - categorical_accuracy: 0.9948 - val_loss: 0.0073 - val_categorical_accuracy: 0.9666

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 30/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0012 - categorical_accuracy: 0.9953 - val_loss: 0.0069 - val_categorical_accuracy: 0.9678
Epoch 31/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0012 - categorical_accuracy: 0.9954 - val_loss: 0.0069 - val_categorical_accuracy: 0.9676
Epoch 32/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0012 - categorical_accuracy: 0.9955 - val_loss: 0.0070 - val_categorical_accuracy: 0.9673
Epoch 33/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0012 - categorical_accuracy: 0.9956 - val_loss: 0.0070 - val_categorical_accuracy: 0.9681
Epoch 34/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0011 - categorical_accuracy: 0.9956 - val_loss: 0.0069 - val_categorical_accuracy: 0.9684

Epoch 00034: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 35/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0011 - categorical_accuracy: 0.9959 - val_loss: 0.0069 - val_categorical_accuracy: 0.9680
Epoch 36/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0012 - categorical_accuracy: 0.9958 - val_loss: 0.0069 - val_categorical_accuracy: 0.9679
Epoch 37/65535
420/420 [==============================] - 21s 51ms/step - loss: 0.0011 - categorical_accuracy: 0.9961 - val_loss: 0.0069 - val_categorical_accuracy: 0.9681
Epoch 38/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0011 - categorical_accuracy: 0.9959 - val_loss: 0.0069 - val_categorical_accuracy: 0.9678
Epoch 00038: early stopping
========= generating oof predictions 10:01:21 =========
========= generating test set predictions 10:01:23 =========
========= fitting 3 th model 10:01:45 =========
Epoch 1/65535
420/420 [==============================] - 20s 49ms/step - loss: 0.1100 - categorical_accuracy: 0.3133 - val_loss: 0.0538 - val_categorical_accuracy: 0.6808
Epoch 2/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0428 - categorical_accuracy: 0.7555 - val_loss: 0.0374 - val_categorical_accuracy: 0.7718
Epoch 3/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0270 - categorical_accuracy: 0.8540 - val_loss: 0.0160 - val_categorical_accuracy: 0.9131
Epoch 4/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0211 - categorical_accuracy: 0.8892 - val_loss: 0.0168 - val_categorical_accuracy: 0.9072
Epoch 5/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0175 - categorical_accuracy: 0.9084 - val_loss: 0.0149 - val_categorical_accuracy: 0.9195
Epoch 6/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0151 - categorical_accuracy: 0.9213 - val_loss: 0.0137 - val_categorical_accuracy: 0.9260
Epoch 7/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0134 - categorical_accuracy: 0.9311 - val_loss: 0.0113 - val_categorical_accuracy: 0.9369
Epoch 8/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0121 - categorical_accuracy: 0.9381 - val_loss: 0.0102 - val_categorical_accuracy: 0.9447
Epoch 9/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0108 - categorical_accuracy: 0.9449 - val_loss: 0.0103 - val_categorical_accuracy: 0.9446
Epoch 10/65535
420/420 [==============================] - 20s 49ms/step - loss: 0.0098 - categorical_accuracy: 0.9499 - val_loss: 0.0121 - val_categorical_accuracy: 0.9379
Epoch 11/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0091 - categorical_accuracy: 0.9530 - val_loss: 0.0101 - val_categorical_accuracy: 0.9475
Epoch 12/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0083 - categorical_accuracy: 0.9565 - val_loss: 0.0094 - val_categorical_accuracy: 0.9493
Epoch 13/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0076 - categorical_accuracy: 0.9600 - val_loss: 0.0088 - val_categorical_accuracy: 0.9553
Epoch 14/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0072 - categorical_accuracy: 0.9632 - val_loss: 0.0099 - val_categorical_accuracy: 0.9481
Epoch 15/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0067 - categorical_accuracy: 0.9650 - val_loss: 0.0092 - val_categorical_accuracy: 0.9531
Epoch 16/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0064 - categorical_accuracy: 0.9670 - val_loss: 0.0107 - val_categorical_accuracy: 0.9456
Epoch 17/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0058 - categorical_accuracy: 0.9697 - val_loss: 0.0105 - val_categorical_accuracy: 0.9475
Epoch 18/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0053 - categorical_accuracy: 0.9721 - val_loss: 0.0094 - val_categorical_accuracy: 0.9533
Epoch 19/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0053 - categorical_accuracy: 0.9725 - val_loss: 0.0089 - val_categorical_accuracy: 0.9572

Epoch 00019: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 20/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0038 - categorical_accuracy: 0.9809 - val_loss: 0.0073 - val_categorical_accuracy: 0.9639
Epoch 21/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0033 - categorical_accuracy: 0.9843 - val_loss: 0.0071 - val_categorical_accuracy: 0.9651
Epoch 22/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0031 - categorical_accuracy: 0.9851 - val_loss: 0.0072 - val_categorical_accuracy: 0.9651
Epoch 23/65535
420/420 [==============================] - 21s 50ms/step - loss: 0.0029 - categorical_accuracy: 0.9863 - val_loss: 0.0072 - val_categorical_accuracy: 0.9668
Epoch 24/65535
420/420 [==============================] - 24s 56ms/step - loss: 0.0028 - categorical_accuracy: 0.9865 - val_loss: 0.0072 - val_categorical_accuracy: 0.9662
Epoch 25/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0027 - categorical_accuracy: 0.9875 - val_loss: 0.0074 - val_categorical_accuracy: 0.9665
Epoch 26/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0026 - categorical_accuracy: 0.9876 - val_loss: 0.0072 - val_categorical_accuracy: 0.9668
Epoch 27/65535
420/420 [==============================] - 24s 57ms/step - loss: 0.0025 - categorical_accuracy: 0.9880 - val_loss: 0.0073 - val_categorical_accuracy: 0.9667

Epoch 00027: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 28/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0023 - categorical_accuracy: 0.9898 - val_loss: 0.0070 - val_categorical_accuracy: 0.9686
Epoch 29/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0023 - categorical_accuracy: 0.9893 - val_loss: 0.0070 - val_categorical_accuracy: 0.9681
Epoch 30/65535
420/420 [==============================] - 23s 55ms/step - loss: 0.0023 - categorical_accuracy: 0.9895 - val_loss: 0.0070 - val_categorical_accuracy: 0.9675
Epoch 31/65535
420/420 [==============================] - 23s 56ms/step - loss: 0.0022 - categorical_accuracy: 0.9907 - val_loss: 0.0070 - val_categorical_accuracy: 0.9680
Epoch 32/65535
420/420 [==============================] - 24s 57ms/step - loss: 0.0022 - categorical_accuracy: 0.9909 - val_loss: 0.0070 - val_categorical_accuracy: 0.9680
Epoch 33/65535
420/420 [==============================] - 24s 56ms/step - loss: 0.0022 - categorical_accuracy: 0.9901 - val_loss: 0.0070 - val_categorical_accuracy: 0.9686
Epoch 34/65535
420/420 [==============================] - 24s 56ms/step - loss: 0.0021 - categorical_accuracy: 0.9907 - val_loss: 0.0071 - val_categorical_accuracy: 0.9679

Epoch 00034: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 35/65535
420/420 [==============================] - 24s 56ms/step - loss: 0.0021 - categorical_accuracy: 0.9907 - val_loss: 0.0070 - val_categorical_accuracy: 0.9682
Epoch 36/65535
420/420 [==============================] - 24s 57ms/step - loss: 0.0021 - categorical_accuracy: 0.9906 - val_loss: 0.0070 - val_categorical_accuracy: 0.9683
Epoch 37/65535
420/420 [==============================] - 23s 56ms/step - loss: 0.0021 - categorical_accuracy: 0.9910 - val_loss: 0.0070 - val_categorical_accuracy: 0.9683
Epoch 38/65535
420/420 [==============================] - 23s 56ms/step - loss: 0.0020 - categorical_accuracy: 0.9914 - val_loss: 0.0070 - val_categorical_accuracy: 0.9683
Epoch 39/65535
420/420 [==============================] - 23s 55ms/step - loss: 0.0022 - categorical_accuracy: 0.9906 - val_loss: 0.0070 - val_categorical_accuracy: 0.9683

Epoch 00039: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 40/65535
420/420 [==============================] - 20s 49ms/step - loss: 0.0020 - categorical_accuracy: 0.9919 - val_loss: 0.0070 - val_categorical_accuracy: 0.9683
Epoch 41/65535
420/420 [==============================] - 21s 50ms/step - loss: 0.0021 - categorical_accuracy: 0.9907 - val_loss: 0.0070 - val_categorical_accuracy: 0.9686
Epoch 42/65535
420/420 [==============================] - 21s 51ms/step - loss: 0.0021 - categorical_accuracy: 0.9912 - val_loss: 0.0070 - val_categorical_accuracy: 0.9686
Epoch 43/65535
420/420 [==============================] - 22s 52ms/step - loss: 0.0021 - categorical_accuracy: 0.9905 - val_loss: 0.0070 - val_categorical_accuracy: 0.9686
Epoch 00043: early stopping
========= generating oof predictions 10:17:10 =========
========= generating test set predictions 10:17:12 =========
========= fitting 4 th model 10:17:36 =========
Epoch 1/65535
420/420 [==============================] - 22s 52ms/step - loss: 0.0949 - categorical_accuracy: 0.4247 - val_loss: 0.0373 - val_categorical_accuracy: 0.7965
Epoch 2/65535
420/420 [==============================] - 22s 51ms/step - loss: 0.0343 - categorical_accuracy: 0.8125 - val_loss: 0.0190 - val_categorical_accuracy: 0.8968
Epoch 3/65535
420/420 [==============================] - 21s 50ms/step - loss: 0.0224 - categorical_accuracy: 0.8819 - val_loss: 0.0133 - val_categorical_accuracy: 0.9311
Epoch 4/65535
420/420 [==============================] - 21s 50ms/step - loss: 0.0174 - categorical_accuracy: 0.9100 - val_loss: 0.0146 - val_categorical_accuracy: 0.9191
Epoch 5/65535
420/420 [==============================] - 23s 55ms/step - loss: 0.0146 - categorical_accuracy: 0.9253 - val_loss: 0.0102 - val_categorical_accuracy: 0.9453
Epoch 6/65535
420/420 [==============================] - 24s 58ms/step - loss: 0.0126 - categorical_accuracy: 0.9347 - val_loss: 0.0098 - val_categorical_accuracy: 0.9466
Epoch 7/65535
420/420 [==============================] - 24s 57ms/step - loss: 0.0113 - categorical_accuracy: 0.9418 - val_loss: 0.0088 - val_categorical_accuracy: 0.9518
Epoch 8/65535
420/420 [==============================] - 23s 56ms/step - loss: 0.0101 - categorical_accuracy: 0.9484 - val_loss: 0.0085 - val_categorical_accuracy: 0.9529
Epoch 9/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0088 - categorical_accuracy: 0.9553 - val_loss: 0.0090 - val_categorical_accuracy: 0.9525
Epoch 10/65535
420/420 [==============================] - 23s 56ms/step - loss: 0.0081 - categorical_accuracy: 0.9597 - val_loss: 0.0088 - val_categorical_accuracy: 0.9517
Epoch 11/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0074 - categorical_accuracy: 0.9627 - val_loss: 0.0082 - val_categorical_accuracy: 0.9551
Epoch 12/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0068 - categorical_accuracy: 0.9651 - val_loss: 0.0082 - val_categorical_accuracy: 0.9548
Epoch 13/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0063 - categorical_accuracy: 0.9679 - val_loss: 0.0082 - val_categorical_accuracy: 0.9578
Epoch 14/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0060 - categorical_accuracy: 0.9688 - val_loss: 0.0076 - val_categorical_accuracy: 0.9586
Epoch 15/65535
420/420 [==============================] - 22s 54ms/step - loss: 0.0055 - categorical_accuracy: 0.9720 - val_loss: 0.0080 - val_categorical_accuracy: 0.9568
Epoch 16/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0051 - categorical_accuracy: 0.9738 - val_loss: 0.0074 - val_categorical_accuracy: 0.9628
Epoch 17/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0046 - categorical_accuracy: 0.9758 - val_loss: 0.0075 - val_categorical_accuracy: 0.9619
Epoch 18/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0044 - categorical_accuracy: 0.9780 - val_loss: 0.0091 - val_categorical_accuracy: 0.9560
Epoch 19/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0039 - categorical_accuracy: 0.9803 - val_loss: 0.0069 - val_categorical_accuracy: 0.9654
Epoch 20/65535
420/420 [==============================] - 21s 51ms/step - loss: 0.0037 - categorical_accuracy: 0.9813 - val_loss: 0.0085 - val_categorical_accuracy: 0.9567
Epoch 21/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0037 - categorical_accuracy: 0.9814 - val_loss: 0.0083 - val_categorical_accuracy: 0.9607
Epoch 22/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0033 - categorical_accuracy: 0.9830 - val_loss: 0.0088 - val_categorical_accuracy: 0.9568
Epoch 23/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0030 - categorical_accuracy: 0.9848 - val_loss: 0.0087 - val_categorical_accuracy: 0.9577
Epoch 24/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0030 - categorical_accuracy: 0.9841 - val_loss: 0.0076 - val_categorical_accuracy: 0.9644
Epoch 25/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0028 - categorical_accuracy: 0.9860 - val_loss: 0.0081 - val_categorical_accuracy: 0.9626

Epoch 00025: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 26/65535
420/420 [==============================] - 21s 51ms/step - loss: 0.0019 - categorical_accuracy: 0.9907 - val_loss: 0.0071 - val_categorical_accuracy: 0.9661
Epoch 27/65535
420/420 [==============================] - 21s 49ms/step - loss: 0.0016 - categorical_accuracy: 0.9928 - val_loss: 0.0068 - val_categorical_accuracy: 0.9675
Epoch 28/65535
420/420 [==============================] - 21s 50ms/step - loss: 0.0014 - categorical_accuracy: 0.9938 - val_loss: 0.0067 - val_categorical_accuracy: 0.9685
Epoch 29/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0013 - categorical_accuracy: 0.9943 - val_loss: 0.0069 - val_categorical_accuracy: 0.9682
Epoch 30/65535
420/420 [==============================] - 22s 52ms/step - loss: 0.0012 - categorical_accuracy: 0.9951 - val_loss: 0.0069 - val_categorical_accuracy: 0.9682
Epoch 31/65535
420/420 [==============================] - 23s 55ms/step - loss: 0.0012 - categorical_accuracy: 0.9954 - val_loss: 0.0068 - val_categorical_accuracy: 0.9684
Epoch 32/65535
420/420 [==============================] - 21s 51ms/step - loss: 0.0011 - categorical_accuracy: 0.9957 - val_loss: 0.0069 - val_categorical_accuracy: 0.9683
Epoch 33/65535
420/420 [==============================] - 22s 53ms/step - loss: 0.0011 - categorical_accuracy: 0.9957 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688
Epoch 34/65535
420/420 [==============================] - 23s 54ms/step - loss: 0.0011 - categorical_accuracy: 0.9956 - val_loss: 0.0069 - val_categorical_accuracy: 0.9679

Epoch 00034: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 35/65535
420/420 [==============================] - 22s 51ms/step - loss: 9.3663e-04 - categorical_accuracy: 0.9967 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688
Epoch 36/65535
420/420 [==============================] - 23s 54ms/step - loss: 9.5798e-04 - categorical_accuracy: 0.9966 - val_loss: 0.0067 - val_categorical_accuracy: 0.9682
Epoch 37/65535
420/420 [==============================] - 21s 51ms/step - loss: 9.3784e-04 - categorical_accuracy: 0.9966 - val_loss: 0.0067 - val_categorical_accuracy: 0.9682
Epoch 38/65535
420/420 [==============================] - 21s 51ms/step - loss: 9.1804e-04 - categorical_accuracy: 0.9969 - val_loss: 0.0067 - val_categorical_accuracy: 0.9685
Epoch 39/65535
420/420 [==============================] - 22s 52ms/step - loss: 8.9047e-04 - categorical_accuracy: 0.9969 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688

Epoch 00039: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 40/65535
420/420 [==============================] - 22s 52ms/step - loss: 8.8668e-04 - categorical_accuracy: 0.9969 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688
Epoch 41/65535
420/420 [==============================] - 22s 53ms/step - loss: 8.9329e-04 - categorical_accuracy: 0.9969 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688
Epoch 42/65535
420/420 [==============================] - 23s 55ms/step - loss: 9.3505e-04 - categorical_accuracy: 0.9967 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688
Epoch 43/65535
420/420 [==============================] - 26s 61ms/step - loss: 8.8779e-04 - categorical_accuracy: 0.9970 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688
Epoch 44/65535
420/420 [==============================] - 26s 61ms/step - loss: 8.8303e-04 - categorical_accuracy: 0.9971 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688

Epoch 00044: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 45/65535
420/420 [==============================] - 26s 62ms/step - loss: 8.6769e-04 - categorical_accuracy: 0.9968 - val_loss: 0.0067 - val_categorical_accuracy: 0.9689
Epoch 46/65535
420/420 [==============================] - 26s 62ms/step - loss: 8.9647e-04 - categorical_accuracy: 0.9966 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688
Epoch 47/65535
420/420 [==============================] - 26s 62ms/step - loss: 8.6607e-04 - categorical_accuracy: 0.9970 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688
Epoch 48/65535
420/420 [==============================] - 25s 60ms/step - loss: 8.6805e-04 - categorical_accuracy: 0.9970 - val_loss: 0.0067 - val_categorical_accuracy: 0.9688
Epoch 00048: early stopping
========= generating oof predictions 10:35:35 =========
========= generating test set predictions 10:35:37 =========
========= fitting 5 th model 10:36:09 =========
Epoch 1/65535
420/420 [==============================] - 27s 65ms/step - loss: 0.1063 - categorical_accuracy: 0.3477 - val_loss: 0.0535 - val_categorical_accuracy: 0.6694
Epoch 2/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.0374 - categorical_accuracy: 0.7921 - val_loss: 0.0249 - val_categorical_accuracy: 0.8707
Epoch 3/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0232 - categorical_accuracy: 0.8779 - val_loss: 0.0177 - val_categorical_accuracy: 0.9022
Epoch 4/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0176 - categorical_accuracy: 0.9082 - val_loss: 0.0202 - val_categorical_accuracy: 0.8857
Epoch 5/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0149 - categorical_accuracy: 0.9237 - val_loss: 0.0098 - val_categorical_accuracy: 0.9457
Epoch 6/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0128 - categorical_accuracy: 0.9343 - val_loss: 0.0165 - val_categorical_accuracy: 0.9090
Epoch 7/65535
420/420 [==============================] - 27s 64ms/step - loss: 0.0115 - categorical_accuracy: 0.9414 - val_loss: 0.0127 - val_categorical_accuracy: 0.9357
Epoch 8/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.0100 - categorical_accuracy: 0.9483 - val_loss: 0.0106 - val_categorical_accuracy: 0.9423
Epoch 9/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0093 - categorical_accuracy: 0.9517 - val_loss: 0.0087 - val_categorical_accuracy: 0.9526
Epoch 10/65535
420/420 [==============================] - 27s 64ms/step - loss: 0.0084 - categorical_accuracy: 0.9571 - val_loss: 0.0108 - val_categorical_accuracy: 0.9436
Epoch 11/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.0078 - categorical_accuracy: 0.9590 - val_loss: 0.0096 - val_categorical_accuracy: 0.9470
Epoch 12/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0070 - categorical_accuracy: 0.9643 - val_loss: 0.0085 - val_categorical_accuracy: 0.9547
Epoch 13/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0064 - categorical_accuracy: 0.9673 - val_loss: 0.0082 - val_categorical_accuracy: 0.9553
Epoch 14/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.0060 - categorical_accuracy: 0.9691 - val_loss: 0.0085 - val_categorical_accuracy: 0.9567
Epoch 15/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.0055 - categorical_accuracy: 0.9716 - val_loss: 0.0109 - val_categorical_accuracy: 0.9509
Epoch 16/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.0053 - categorical_accuracy: 0.9729 - val_loss: 0.0081 - val_categorical_accuracy: 0.9580
Epoch 17/65535
420/420 [==============================] - 25s 61ms/step - loss: 0.0050 - categorical_accuracy: 0.9739 - val_loss: 0.0077 - val_categorical_accuracy: 0.9622
Epoch 18/65535
420/420 [==============================] - 26s 61ms/step - loss: 0.0045 - categorical_accuracy: 0.9769 - val_loss: 0.0085 - val_categorical_accuracy: 0.9567
Epoch 19/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0044 - categorical_accuracy: 0.9781 - val_loss: 0.0078 - val_categorical_accuracy: 0.9617
Epoch 20/65535
420/420 [==============================] - 27s 64ms/step - loss: 0.0038 - categorical_accuracy: 0.9807 - val_loss: 0.0079 - val_categorical_accuracy: 0.9622
Epoch 21/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0038 - categorical_accuracy: 0.9804 - val_loss: 0.0089 - val_categorical_accuracy: 0.9577
Epoch 22/65535
420/420 [==============================] - 26s 62ms/step - loss: 0.0036 - categorical_accuracy: 0.9818 - val_loss: 0.0084 - val_categorical_accuracy: 0.9606
Epoch 23/65535
420/420 [==============================] - 26s 63ms/step - loss: 0.0033 - categorical_accuracy: 0.9827 - val_loss: 0.0089 - val_categorical_accuracy: 0.9596

Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 24/65535
420/420 [==============================] - 24s 57ms/step - loss: 0.0023 - categorical_accuracy: 0.9888 - val_loss: 0.0069 - val_categorical_accuracy: 0.9681
Epoch 25/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0019 - categorical_accuracy: 0.9908 - val_loss: 0.0070 - val_categorical_accuracy: 0.9685
Epoch 26/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0018 - categorical_accuracy: 0.9919 - val_loss: 0.0071 - val_categorical_accuracy: 0.9681
Epoch 27/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0016 - categorical_accuracy: 0.9925 - val_loss: 0.0072 - val_categorical_accuracy: 0.9685
Epoch 28/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0016 - categorical_accuracy: 0.9929 - val_loss: 0.0072 - val_categorical_accuracy: 0.9690
Epoch 29/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0014 - categorical_accuracy: 0.9939 - val_loss: 0.0072 - val_categorical_accuracy: 0.9674
Epoch 30/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0014 - categorical_accuracy: 0.9940 - val_loss: 0.0072 - val_categorical_accuracy: 0.9688

Epoch 00030: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 31/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0013 - categorical_accuracy: 0.9950 - val_loss: 0.0071 - val_categorical_accuracy: 0.9695
Epoch 32/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0013 - categorical_accuracy: 0.9947 - val_loss: 0.0072 - val_categorical_accuracy: 0.9695
Epoch 33/65535
420/420 [==============================] - 20s 49ms/step - loss: 0.0013 - categorical_accuracy: 0.9947 - val_loss: 0.0072 - val_categorical_accuracy: 0.9697
Epoch 34/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0012 - categorical_accuracy: 0.9952 - val_loss: 0.0072 - val_categorical_accuracy: 0.9697
Epoch 35/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0012 - categorical_accuracy: 0.9955 - val_loss: 0.0072 - val_categorical_accuracy: 0.9694

Epoch 00035: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 36/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0012 - categorical_accuracy: 0.9950 - val_loss: 0.0072 - val_categorical_accuracy: 0.9694
Epoch 37/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0012 - categorical_accuracy: 0.9952 - val_loss: 0.0072 - val_categorical_accuracy: 0.9700
Epoch 38/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0012 - categorical_accuracy: 0.9954 - val_loss: 0.0072 - val_categorical_accuracy: 0.9699
Epoch 39/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0011 - categorical_accuracy: 0.9962 - val_loss: 0.0072 - val_categorical_accuracy: 0.9697
Epoch 00039: early stopping
========= generating oof predictions 10:51:42 =========
========= generating test set predictions 10:51:44 =========
train loss avg 0.001351262494714203 -- std 0.0004314073962773064, val loss avg 0.006950727925110821 -- std 0.00015277639587709334
train acc avg 0.9947146849129883 -- std 0.002324864223442957, val acc avg 0.9688921074586915 -- std 0.0006717911775506614
mean nb epochs 40.6
dump oof predicted probs
dump test set predicted probs
