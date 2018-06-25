ren (master *+) python $ python train.py logmelspectrogram_40_25_10 gg4 s 128
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
conv2d_1 (Conv2D)               (None, 101, 40, 16)  176         reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 101, 40, 16)  64          conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 101, 40, 16)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 101, 40, 16)  2576        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 101, 40, 16)  64          conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 101, 40, 16)  0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 50, 20, 16)   0           activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 50, 20, 32)   5152        max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 50, 20, 32)   128         conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 50, 20, 32)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 50, 20, 32)   10272       activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 50, 20, 32)   128         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 50, 20, 32)   0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 25, 10, 32)   0           activation_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 25, 10, 64)   20544       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 25, 10, 64)   256         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 25, 10, 64)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 25, 10, 64)   41024       activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 25, 10, 64)   256         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 25, 10, 64)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 12, 5, 64)    0           activation_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 12, 5, 128)   82048       max_pooling2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 12, 5, 128)   512         conv2d_7[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 12, 5, 128)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 12, 5, 128)   163968      activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 12, 5, 128)   512         conv2d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 12, 5, 128)   0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 6, 2, 128)    0           activation_8[0][0]
__________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalM (None, 128)          0           max_pooling2d_4[0][0]
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 128)          0           max_pooling2d_4[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 256)          0           global_max_pooling2d_1[0][0]
                                                                 global_average_pooling2d_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 256)          65792       concatenate_1[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 256)          1024        dense_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 256)          0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 256)          65792       dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 256)          1024        dense_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 256)          0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 31)           7967        dropout_2[0][0]
==================================================================================================
Total params: 469,279
Trainable params: 467,295
Non-trainable params: 1,984
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 23:20:42 =========
Epoch 1/65535
420/420 [==============================] - 17s 41ms/step - loss: 0.0991 - categorical_accuracy: 0.4007 - val_loss: 0.0518 - val_categorical_accuracy: 0.7192
Epoch 2/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0288 - categorical_accuracy: 0.8383 - val_loss: 0.0377 - val_categorical_accuracy: 0.7923
Epoch 3/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0188 - categorical_accuracy: 0.8987 - val_loss: 0.0128 - val_categorical_accuracy: 0.9271
Epoch 4/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0144 - categorical_accuracy: 0.9225 - val_loss: 0.0152 - val_categorical_accuracy: 0.9172
Epoch 5/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0121 - categorical_accuracy: 0.9350 - val_loss: 0.0121 - val_categorical_accuracy: 0.9306
Epoch 6/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0103 - categorical_accuracy: 0.9460 - val_loss: 0.0098 - val_categorical_accuracy: 0.9472
Epoch 7/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0093 - categorical_accuracy: 0.9501 - val_loss: 0.0103 - val_categorical_accuracy: 0.9416
Epoch 8/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0080 - categorical_accuracy: 0.9582 - val_loss: 0.0109 - val_categorical_accuracy: 0.9433
Epoch 9/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0074 - categorical_accuracy: 0.9612 - val_loss: 0.0109 - val_categorical_accuracy: 0.9460
Epoch 10/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0067 - categorical_accuracy: 0.9640 - val_loss: 0.0087 - val_categorical_accuracy: 0.9522
Epoch 11/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0060 - categorical_accuracy: 0.9686 - val_loss: 0.0083 - val_categorical_accuracy: 0.9544
Epoch 12/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0056 - categorical_accuracy: 0.9697 - val_loss: 0.0125 - val_categorical_accuracy: 0.9472
Epoch 13/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0052 - categorical_accuracy: 0.9721 - val_loss: 0.0079 - val_categorical_accuracy: 0.9564
Epoch 14/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0049 - categorical_accuracy: 0.9738 - val_loss: 0.0138 - val_categorical_accuracy: 0.9399
Epoch 15/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0045 - categorical_accuracy: 0.9765 - val_loss: 0.0134 - val_categorical_accuracy: 0.9440
Epoch 16/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0041 - categorical_accuracy: 0.9778 - val_loss: 0.0097 - val_categorical_accuracy: 0.9523
Epoch 17/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0037 - categorical_accuracy: 0.9802 - val_loss: 0.0136 - val_categorical_accuracy: 0.9335
Epoch 18/65535
420/420 [==============================] - 17s 41ms/step - loss: 0.0035 - categorical_accuracy: 0.9816 - val_loss: 0.0108 - val_categorical_accuracy: 0.9549
Epoch 19/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0032 - categorical_accuracy: 0.9830 - val_loss: 0.0084 - val_categorical_accuracy: 0.9585

Epoch 00019: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 20/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0022 - categorical_accuracy: 0.9885 - val_loss: 0.0076 - val_categorical_accuracy: 0.9634
Epoch 21/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9917 - val_loss: 0.0075 - val_categorical_accuracy: 0.9632
Epoch 22/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0016 - categorical_accuracy: 0.9928 - val_loss: 0.0075 - val_categorical_accuracy: 0.9641
Epoch 23/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0015 - categorical_accuracy: 0.9932 - val_loss: 0.0075 - val_categorical_accuracy: 0.9644
Epoch 24/65535
420/420 [==============================] - 17s 41ms/step - loss: 0.0013 - categorical_accuracy: 0.9943 - val_loss: 0.0076 - val_categorical_accuracy: 0.9641
Epoch 25/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0012 - categorical_accuracy: 0.9950 - val_loss: 0.0077 - val_categorical_accuracy: 0.9636
Epoch 26/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0012 - categorical_accuracy: 0.9949 - val_loss: 0.0076 - val_categorical_accuracy: 0.9640

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 27/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0011 - categorical_accuracy: 0.9959 - val_loss: 0.0076 - val_categorical_accuracy: 0.9648
Epoch 28/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0010 - categorical_accuracy: 0.9960 - val_loss: 0.0076 - val_categorical_accuracy: 0.9644
Epoch 29/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0010 - categorical_accuracy: 0.9962 - val_loss: 0.0076 - val_categorical_accuracy: 0.9646
Epoch 30/65535
420/420 [==============================] - 17s 40ms/step - loss: 9.9146e-04 - categorical_accuracy: 0.9962 - val_loss: 0.0076 - val_categorical_accuracy: 0.9653
Epoch 31/65535
420/420 [==============================] - 17s 40ms/step - loss: 9.6430e-04 - categorical_accuracy: 0.9969 - val_loss: 0.0077 - val_categorical_accuracy: 0.9649

Epoch 00031: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 32/65535
420/420 [==============================] - 17s 40ms/step - loss: 9.4857e-04 - categorical_accuracy: 0.9966 - val_loss: 0.0076 - val_categorical_accuracy: 0.9650
Epoch 33/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0010 - categorical_accuracy: 0.9961 - val_loss: 0.0077 - val_categorical_accuracy: 0.9648
Epoch 34/65535
420/420 [==============================] - 17s 40ms/step - loss: 9.5641e-04 - categorical_accuracy: 0.9964 - val_loss: 0.0077 - val_categorical_accuracy: 0.9648
Epoch 35/65535
420/420 [==============================] - 17s 39ms/step - loss: 9.4550e-04 - categorical_accuracy: 0.9963 - val_loss: 0.0077 - val_categorical_accuracy: 0.9647
Epoch 36/65535
420/420 [==============================] - 17s 40ms/step - loss: 9.2736e-04 - categorical_accuracy: 0.9965 - val_loss: 0.0077 - val_categorical_accuracy: 0.9649

Epoch 00036: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 37/65535
420/420 [==============================] - 17s 40ms/step - loss: 9.3865e-04 - categorical_accuracy: 0.9964 - val_loss: 0.0077 - val_categorical_accuracy: 0.9648
Epoch 00037: early stopping
========= generating oof predictions 23:30:57 =========
========= generating test set predictions 23:30:58 =========
========= fitting 2 th model 23:31:18 =========
Epoch 1/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0993 - categorical_accuracy: 0.3992 - val_loss: 0.0536 - val_categorical_accuracy: 0.6707
Epoch 2/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0284 - categorical_accuracy: 0.8404 - val_loss: 0.0208 - val_categorical_accuracy: 0.8842
Epoch 3/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0181 - categorical_accuracy: 0.9008 - val_loss: 0.0191 - val_categorical_accuracy: 0.8897
Epoch 4/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0141 - categorical_accuracy: 0.9249 - val_loss: 0.0158 - val_categorical_accuracy: 0.9250
Epoch 5/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0116 - categorical_accuracy: 0.9382 - val_loss: 0.0117 - val_categorical_accuracy: 0.9327
Epoch 6/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0101 - categorical_accuracy: 0.9466 - val_loss: 0.0094 - val_categorical_accuracy: 0.9471
Epoch 7/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0089 - categorical_accuracy: 0.9528 - val_loss: 0.0084 - val_categorical_accuracy: 0.9515
Epoch 8/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0080 - categorical_accuracy: 0.9574 - val_loss: 0.0081 - val_categorical_accuracy: 0.9532
Epoch 9/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0073 - categorical_accuracy: 0.9620 - val_loss: 0.0085 - val_categorical_accuracy: 0.9527
Epoch 10/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0065 - categorical_accuracy: 0.9661 - val_loss: 0.0098 - val_categorical_accuracy: 0.9440
Epoch 11/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0061 - categorical_accuracy: 0.9681 - val_loss: 0.0116 - val_categorical_accuracy: 0.9424
Epoch 12/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0056 - categorical_accuracy: 0.9716 - val_loss: 0.0119 - val_categorical_accuracy: 0.9388
Epoch 13/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0050 - categorical_accuracy: 0.9737 - val_loss: 0.0090 - val_categorical_accuracy: 0.9510
Epoch 14/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0047 - categorical_accuracy: 0.9756 - val_loss: 0.0094 - val_categorical_accuracy: 0.9495

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 15/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0034 - categorical_accuracy: 0.9826 - val_loss: 0.0072 - val_categorical_accuracy: 0.9621
Epoch 16/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0029 - categorical_accuracy: 0.9857 - val_loss: 0.0073 - val_categorical_accuracy: 0.9614
Epoch 17/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0027 - categorical_accuracy: 0.9872 - val_loss: 0.0073 - val_categorical_accuracy: 0.9627
Epoch 18/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0025 - categorical_accuracy: 0.9881 - val_loss: 0.0073 - val_categorical_accuracy: 0.9623
Epoch 19/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0024 - categorical_accuracy: 0.9883 - val_loss: 0.0075 - val_categorical_accuracy: 0.9618
Epoch 20/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0023 - categorical_accuracy: 0.9897 - val_loss: 0.0075 - val_categorical_accuracy: 0.9625
Epoch 21/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0021 - categorical_accuracy: 0.9903 - val_loss: 0.0074 - val_categorical_accuracy: 0.9630

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 22/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0020 - categorical_accuracy: 0.9911 - val_loss: 0.0070 - val_categorical_accuracy: 0.9641
Epoch 23/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0019 - categorical_accuracy: 0.9913 - val_loss: 0.0070 - val_categorical_accuracy: 0.9649
Epoch 24/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9922 - val_loss: 0.0070 - val_categorical_accuracy: 0.9649
Epoch 25/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9920 - val_loss: 0.0070 - val_categorical_accuracy: 0.9655
Epoch 26/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9919 - val_loss: 0.0070 - val_categorical_accuracy: 0.9649
Epoch 27/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9920 - val_loss: 0.0070 - val_categorical_accuracy: 0.9648
Epoch 28/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9917 - val_loss: 0.0070 - val_categorical_accuracy: 0.9651

Epoch 00028: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 29/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9927 - val_loss: 0.0070 - val_categorical_accuracy: 0.9651
Epoch 30/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9926 - val_loss: 0.0070 - val_categorical_accuracy: 0.9647
Epoch 31/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9924 - val_loss: 0.0070 - val_categorical_accuracy: 0.9648
Epoch 32/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9925 - val_loss: 0.0070 - val_categorical_accuracy: 0.9647
Epoch 33/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0016 - categorical_accuracy: 0.9934 - val_loss: 0.0070 - val_categorical_accuracy: 0.9648

Epoch 00033: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 34/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9927 - val_loss: 0.0070 - val_categorical_accuracy: 0.9650
Epoch 35/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9929 - val_loss: 0.0070 - val_categorical_accuracy: 0.9649
Epoch 36/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0017 - categorical_accuracy: 0.9927 - val_loss: 0.0070 - val_categorical_accuracy: 0.9649
Epoch 37/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9929 - val_loss: 0.0070 - val_categorical_accuracy: 0.9650
Epoch 38/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0016 - categorical_accuracy: 0.9932 - val_loss: 0.0070 - val_categorical_accuracy: 0.9649

Epoch 00038: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 00038: early stopping
========= generating oof predictions 23:41:46 =========
========= generating test set predictions 23:41:47 =========
========= fitting 3 th model 23:42:06 =========
Epoch 1/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0916 - categorical_accuracy: 0.4518 - val_loss: 0.0313 - val_categorical_accuracy: 0.8185
Epoch 2/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0264 - categorical_accuracy: 0.8536 - val_loss: 0.0234 - val_categorical_accuracy: 0.8735
Epoch 3/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0174 - categorical_accuracy: 0.9071 - val_loss: 0.0161 - val_categorical_accuracy: 0.9087
Epoch 4/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0138 - categorical_accuracy: 0.9275 - val_loss: 0.0147 - val_categorical_accuracy: 0.9186
Epoch 5/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0116 - categorical_accuracy: 0.9378 - val_loss: 0.0140 - val_categorical_accuracy: 0.9218
Epoch 6/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0100 - categorical_accuracy: 0.9478 - val_loss: 0.0101 - val_categorical_accuracy: 0.9435
Epoch 7/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0088 - categorical_accuracy: 0.9530 - val_loss: 0.0089 - val_categorical_accuracy: 0.9519
Epoch 8/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0080 - categorical_accuracy: 0.9576 - val_loss: 0.0079 - val_categorical_accuracy: 0.9559
Epoch 9/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0073 - categorical_accuracy: 0.9618 - val_loss: 0.0100 - val_categorical_accuracy: 0.9499
Epoch 10/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0065 - categorical_accuracy: 0.9646 - val_loss: 0.0095 - val_categorical_accuracy: 0.9478
Epoch 11/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0062 - categorical_accuracy: 0.9664 - val_loss: 0.0085 - val_categorical_accuracy: 0.9535
Epoch 12/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0056 - categorical_accuracy: 0.9706 - val_loss: 0.0080 - val_categorical_accuracy: 0.9594
Epoch 13/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0051 - categorical_accuracy: 0.9725 - val_loss: 0.0093 - val_categorical_accuracy: 0.9538
Epoch 14/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0047 - categorical_accuracy: 0.9755 - val_loss: 0.0082 - val_categorical_accuracy: 0.9590

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 15/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0033 - categorical_accuracy: 0.9826 - val_loss: 0.0068 - val_categorical_accuracy: 0.9645
Epoch 16/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0030 - categorical_accuracy: 0.9849 - val_loss: 0.0068 - val_categorical_accuracy: 0.9649
Epoch 17/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0027 - categorical_accuracy: 0.9866 - val_loss: 0.0067 - val_categorical_accuracy: 0.9647
Epoch 18/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0025 - categorical_accuracy: 0.9876 - val_loss: 0.0069 - val_categorical_accuracy: 0.9648
Epoch 19/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0024 - categorical_accuracy: 0.9882 - val_loss: 0.0069 - val_categorical_accuracy: 0.9645
Epoch 20/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0022 - categorical_accuracy: 0.9895 - val_loss: 0.0071 - val_categorical_accuracy: 0.9645
Epoch 21/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0022 - categorical_accuracy: 0.9893 - val_loss: 0.0071 - val_categorical_accuracy: 0.9648

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 22/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0020 - categorical_accuracy: 0.9908 - val_loss: 0.0069 - val_categorical_accuracy: 0.9649
Epoch 23/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0019 - categorical_accuracy: 0.9908 - val_loss: 0.0068 - val_categorical_accuracy: 0.9652
Epoch 24/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0019 - categorical_accuracy: 0.9912 - val_loss: 0.0069 - val_categorical_accuracy: 0.9654
Epoch 25/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9918 - val_loss: 0.0068 - val_categorical_accuracy: 0.9657
Epoch 26/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9920 - val_loss: 0.0068 - val_categorical_accuracy: 0.9655

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 27/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9920 - val_loss: 0.0068 - val_categorical_accuracy: 0.9649
Epoch 28/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9927 - val_loss: 0.0068 - val_categorical_accuracy: 0.9651
Epoch 29/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9925 - val_loss: 0.0068 - val_categorical_accuracy: 0.9651
Epoch 30/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9920 - val_loss: 0.0068 - val_categorical_accuracy: 0.9654
Epoch 31/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9921 - val_loss: 0.0068 - val_categorical_accuracy: 0.9652

Epoch 00031: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 32/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9923 - val_loss: 0.0068 - val_categorical_accuracy: 0.9650
Epoch 00032: early stopping
========= generating oof predictions 23:50:56 =========
========= generating test set predictions 23:50:57 =========
========= fitting 4 th model 23:51:17 =========
Epoch 1/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0848 - categorical_accuracy: 0.4945 - val_loss: 0.0352 - val_categorical_accuracy: 0.7899
Epoch 2/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0253 - categorical_accuracy: 0.8589 - val_loss: 0.0170 - val_categorical_accuracy: 0.9036
Epoch 3/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0170 - categorical_accuracy: 0.9087 - val_loss: 0.0131 - val_categorical_accuracy: 0.9270
Epoch 4/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0135 - categorical_accuracy: 0.9280 - val_loss: 0.0117 - val_categorical_accuracy: 0.9367
Epoch 5/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0111 - categorical_accuracy: 0.9411 - val_loss: 0.0101 - val_categorical_accuracy: 0.9440
Epoch 6/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0096 - categorical_accuracy: 0.9498 - val_loss: 0.0100 - val_categorical_accuracy: 0.9448
Epoch 7/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0084 - categorical_accuracy: 0.9554 - val_loss: 0.0089 - val_categorical_accuracy: 0.9497
Epoch 8/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0076 - categorical_accuracy: 0.9601 - val_loss: 0.0081 - val_categorical_accuracy: 0.9562
Epoch 9/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0068 - categorical_accuracy: 0.9640 - val_loss: 0.0089 - val_categorical_accuracy: 0.9507
Epoch 10/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0064 - categorical_accuracy: 0.9659 - val_loss: 0.0102 - val_categorical_accuracy: 0.9448
Epoch 11/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0058 - categorical_accuracy: 0.9688 - val_loss: 0.0078 - val_categorical_accuracy: 0.9568
Epoch 12/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0052 - categorical_accuracy: 0.9721 - val_loss: 0.0108 - val_categorical_accuracy: 0.9453
Epoch 13/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0048 - categorical_accuracy: 0.9750 - val_loss: 0.0087 - val_categorical_accuracy: 0.9522
Epoch 14/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0045 - categorical_accuracy: 0.9761 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 15/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0043 - categorical_accuracy: 0.9774 - val_loss: 0.0090 - val_categorical_accuracy: 0.9524
Epoch 16/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0039 - categorical_accuracy: 0.9794 - val_loss: 0.0098 - val_categorical_accuracy: 0.9495
Epoch 17/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0034 - categorical_accuracy: 0.9824 - val_loss: 0.0098 - val_categorical_accuracy: 0.9464
Epoch 18/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0032 - categorical_accuracy: 0.9832 - val_loss: 0.0091 - val_categorical_accuracy: 0.9577
Epoch 19/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0029 - categorical_accuracy: 0.9848 - val_loss: 0.0083 - val_categorical_accuracy: 0.9577
Epoch 20/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0027 - categorical_accuracy: 0.9865 - val_loss: 0.0108 - val_categorical_accuracy: 0.9499

Epoch 00020: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 21/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0018 - categorical_accuracy: 0.9909 - val_loss: 0.0068 - val_categorical_accuracy: 0.9662
Epoch 22/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0015 - categorical_accuracy: 0.9927 - val_loss: 0.0067 - val_categorical_accuracy: 0.9660
Epoch 23/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0013 - categorical_accuracy: 0.9942 - val_loss: 0.0067 - val_categorical_accuracy: 0.9669
Epoch 24/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9945 - val_loss: 0.0066 - val_categorical_accuracy: 0.9675
Epoch 25/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0012 - categorical_accuracy: 0.9953 - val_loss: 0.0068 - val_categorical_accuracy: 0.9677
Epoch 26/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0011 - categorical_accuracy: 0.9958 - val_loss: 0.0069 - val_categorical_accuracy: 0.9677
Epoch 27/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0010 - categorical_accuracy: 0.9959 - val_loss: 0.0068 - val_categorical_accuracy: 0.9673
Epoch 28/65535
420/420 [==============================] - 17s 40ms/step - loss: 9.5506e-04 - categorical_accuracy: 0.9965 - val_loss: 0.0069 - val_categorical_accuracy: 0.9669
Epoch 29/65535
420/420 [==============================] - 17s 39ms/step - loss: 9.5360e-04 - categorical_accuracy: 0.9963 - val_loss: 0.0069 - val_categorical_accuracy: 0.9665
Epoch 30/65535
420/420 [==============================] - 16s 39ms/step - loss: 9.0126e-04 - categorical_accuracy: 0.9964 - val_loss: 0.0069 - val_categorical_accuracy: 0.9677

Epoch 00030: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 31/65535
420/420 [==============================] - 17s 40ms/step - loss: 8.2541e-04 - categorical_accuracy: 0.9972 - val_loss: 0.0069 - val_categorical_accuracy: 0.9673
Epoch 32/65535
420/420 [==============================] - 17s 39ms/step - loss: 8.2366e-04 - categorical_accuracy: 0.9971 - val_loss: 0.0068 - val_categorical_accuracy: 0.9673
Epoch 33/65535
420/420 [==============================] - 17s 39ms/step - loss: 7.4063e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0068 - val_categorical_accuracy: 0.9676
Epoch 34/65535
420/420 [==============================] - 17s 39ms/step - loss: 7.4080e-04 - categorical_accuracy: 0.9974 - val_loss: 0.0068 - val_categorical_accuracy: 0.9679
Epoch 35/65535
420/420 [==============================] - 17s 40ms/step - loss: 7.3089e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0068 - val_categorical_accuracy: 0.9677

Epoch 00035: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 36/65535
420/420 [==============================] - 17s 39ms/step - loss: 7.1347e-04 - categorical_accuracy: 0.9979 - val_loss: 0.0068 - val_categorical_accuracy: 0.9677
Epoch 37/65535
420/420 [==============================] - 17s 40ms/step - loss: 6.8151e-04 - categorical_accuracy: 0.9978 - val_loss: 0.0068 - val_categorical_accuracy: 0.9675
Epoch 38/65535
420/420 [==============================] - 17s 39ms/step - loss: 7.1091e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0068 - val_categorical_accuracy: 0.9676
Epoch 39/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.0031e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0069 - val_categorical_accuracy: 0.9677
Epoch 00039: early stopping
========= generating oof predictions 00:02:05 =========
========= generating test set predictions 00:02:07 =========
========= fitting 5 th model 00:02:27 =========
Epoch 1/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0883 - categorical_accuracy: 0.4707 - val_loss: 0.0442 - val_categorical_accuracy: 0.7403
Epoch 2/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0267 - categorical_accuracy: 0.8513 - val_loss: 0.0209 - val_categorical_accuracy: 0.8889
Epoch 3/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0177 - categorical_accuracy: 0.9057 - val_loss: 0.0301 - val_categorical_accuracy: 0.8662
Epoch 4/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0139 - categorical_accuracy: 0.9248 - val_loss: 0.0133 - val_categorical_accuracy: 0.9275
Epoch 5/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0116 - categorical_accuracy: 0.9379 - val_loss: 0.0115 - val_categorical_accuracy: 0.9370
Epoch 6/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0100 - categorical_accuracy: 0.9470 - val_loss: 0.0119 - val_categorical_accuracy: 0.9388
Epoch 7/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0087 - categorical_accuracy: 0.9538 - val_loss: 0.0155 - val_categorical_accuracy: 0.9131
Epoch 8/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0077 - categorical_accuracy: 0.9589 - val_loss: 0.0146 - val_categorical_accuracy: 0.9191
Epoch 9/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0071 - categorical_accuracy: 0.9620 - val_loss: 0.0107 - val_categorical_accuracy: 0.9441
Epoch 10/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0066 - categorical_accuracy: 0.9648 - val_loss: 0.0120 - val_categorical_accuracy: 0.9398
Epoch 11/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0060 - categorical_accuracy: 0.9683 - val_loss: 0.0181 - val_categorical_accuracy: 0.9114
Epoch 12/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0053 - categorical_accuracy: 0.9718 - val_loss: 0.0091 - val_categorical_accuracy: 0.9511
Epoch 13/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0049 - categorical_accuracy: 0.9739 - val_loss: 0.0124 - val_categorical_accuracy: 0.9357
Epoch 14/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0047 - categorical_accuracy: 0.9747 - val_loss: 0.0109 - val_categorical_accuracy: 0.9430
Epoch 15/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0043 - categorical_accuracy: 0.9770 - val_loss: 0.0091 - val_categorical_accuracy: 0.9539
Epoch 16/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0039 - categorical_accuracy: 0.9792 - val_loss: 0.0120 - val_categorical_accuracy: 0.9407
Epoch 17/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0036 - categorical_accuracy: 0.9814 - val_loss: 0.0083 - val_categorical_accuracy: 0.9597
Epoch 18/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0032 - categorical_accuracy: 0.9834 - val_loss: 0.0103 - val_categorical_accuracy: 0.9486
Epoch 19/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0030 - categorical_accuracy: 0.9847 - val_loss: 0.0094 - val_categorical_accuracy: 0.9577
Epoch 20/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0026 - categorical_accuracy: 0.9866 - val_loss: 0.0089 - val_categorical_accuracy: 0.9565
Epoch 21/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0026 - categorical_accuracy: 0.9867 - val_loss: 0.0128 - val_categorical_accuracy: 0.9405
Epoch 22/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0025 - categorical_accuracy: 0.9870 - val_loss: 0.0099 - val_categorical_accuracy: 0.9536
Epoch 23/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0023 - categorical_accuracy: 0.9885 - val_loss: 0.0085 - val_categorical_accuracy: 0.9597

Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 24/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0014 - categorical_accuracy: 0.9931 - val_loss: 0.0077 - val_categorical_accuracy: 0.9639
Epoch 25/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0012 - categorical_accuracy: 0.9949 - val_loss: 0.0078 - val_categorical_accuracy: 0.9648
Epoch 26/65535
420/420 [==============================] - 16s 39ms/step - loss: 9.9061e-04 - categorical_accuracy: 0.9961 - val_loss: 0.0077 - val_categorical_accuracy: 0.9656
Epoch 27/65535
420/420 [==============================] - 16s 39ms/step - loss: 9.2818e-04 - categorical_accuracy: 0.9963 - val_loss: 0.0078 - val_categorical_accuracy: 0.9650
Epoch 28/65535
420/420 [==============================] - 16s 39ms/step - loss: 8.6046e-04 - categorical_accuracy: 0.9965 - val_loss: 0.0076 - val_categorical_accuracy: 0.9665
Epoch 29/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.8354e-04 - categorical_accuracy: 0.9969 - val_loss: 0.0077 - val_categorical_accuracy: 0.9665
Epoch 30/65535
420/420 [==============================] - 17s 39ms/step - loss: 7.6990e-04 - categorical_accuracy: 0.9972 - val_loss: 0.0077 - val_categorical_accuracy: 0.9662

Epoch 00030: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 31/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.6200e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670
Epoch 32/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.7959e-04 - categorical_accuracy: 0.9975 - val_loss: 0.0076 - val_categorical_accuracy: 0.9671
Epoch 33/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.4099e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0076 - val_categorical_accuracy: 0.9671
Epoch 34/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.3566e-04 - categorical_accuracy: 0.9980 - val_loss: 0.0076 - val_categorical_accuracy: 0.9673
Epoch 35/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.2279e-04 - categorical_accuracy: 0.9980 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670

Epoch 00035: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 36/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.0880e-04 - categorical_accuracy: 0.9979 - val_loss: 0.0076 - val_categorical_accuracy: 0.9667
Epoch 37/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.1632e-04 - categorical_accuracy: 0.9979 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670
Epoch 38/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.0567e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0076 - val_categorical_accuracy: 0.9669
Epoch 39/65535
420/420 [==============================] - 17s 40ms/step - loss: 6.1259e-04 - categorical_accuracy: 0.9980 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670
Epoch 40/65535
420/420 [==============================] - 16s 39ms/step - loss: 5.8450e-04 - categorical_accuracy: 0.9981 - val_loss: 0.0076 - val_categorical_accuracy: 0.9668

Epoch 00040: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 41/65535
420/420 [==============================] - 16s 39ms/step - loss: 5.8837e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670
Epoch 42/65535
420/420 [==============================] - 17s 39ms/step - loss: 5.7350e-04 - categorical_accuracy: 0.9981 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670
Epoch 43/65535
420/420 [==============================] - 16s 39ms/step - loss: 5.9221e-04 - categorical_accuracy: 0.9983 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670
Epoch 44/65535
420/420 [==============================] - 17s 39ms/step - loss: 5.9388e-04 - categorical_accuracy: 0.9981 - val_loss: 0.0076 - val_categorical_accuracy: 0.9669
Epoch 45/65535
420/420 [==============================] - 16s 39ms/step - loss: 5.7616e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670

Epoch 00045: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 46/65535
420/420 [==============================] - 16s 39ms/step - loss: 5.8221e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670
Epoch 47/65535
420/420 [==============================] - 17s 39ms/step - loss: 5.7700e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0076 - val_categorical_accuracy: 0.9670
Epoch 00047: early stopping
========= generating oof predictions 00:15:22 =========
========= generating test set predictions 00:15:23 =========
train loss avg 0.0011152167523245924 -- std 0.0004770237960336771, val loss avg 0.007204285948556594 -- std 0.00037423572102743193
train acc avg 0.9955638133156652 -- std 0.00239281417618311, val acc avg 0.9658682881364713 -- std 0.0012345221198331493
mean nb epochs 38.6
dump oof predicted probs
dump test set predicted probs
