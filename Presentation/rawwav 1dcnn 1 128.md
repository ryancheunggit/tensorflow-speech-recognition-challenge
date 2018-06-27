ren (master *+) python $ python train.py rawwav 1dcnn 1 128
------------- SUMMARY OF MODEL -------------
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 16000, 1)     0
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 16000, 4)     20          input_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 16000, 4)     16          conv1d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 16000, 4)     0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 16000, 4)     68          activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 16000, 4)     16          conv1d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 16000, 4)     0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 8000, 4)      0           activation_2[0][0]
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 8000, 8)      136         max_pooling1d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 8000, 8)      32          conv1d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 8000, 8)      0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 8000, 8)      264         activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 8000, 8)      32          conv1d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 8000, 8)      0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)  (None, 4000, 8)      0           activation_4[0][0]
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 4000, 16)     528         max_pooling1d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 4000, 16)     64          conv1d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 4000, 16)     0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 4000, 16)     1040        activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 4000, 16)     64          conv1d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 4000, 16)     0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)  (None, 2000, 16)     0           activation_6[0][0]
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 2000, 32)     2080        max_pooling1d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 2000, 32)     128         conv1d_7[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 2000, 32)     0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 2000, 32)     4128        activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 2000, 32)     128         conv1d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 2000, 32)     0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
max_pooling1d_4 (MaxPooling1D)  (None, 1000, 32)     0           activation_8[0][0]
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 1000, 64)     8256        max_pooling1d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 1000, 64)     256         conv1d_9[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 1000, 64)     0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv1d_10 (Conv1D)              (None, 1000, 64)     16448       activation_9[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 1000, 64)     256         conv1d_10[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 1000, 64)     0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
max_pooling1d_5 (MaxPooling1D)  (None, 500, 64)      0           activation_10[0][0]
__________________________________________________________________________________________________
conv1d_11 (Conv1D)              (None, 500, 128)     32896       max_pooling1d_5[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 500, 128)     512         conv1d_11[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 500, 128)     0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
conv1d_12 (Conv1D)              (None, 500, 128)     65664       activation_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 500, 128)     512         conv1d_12[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 500, 128)     0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
max_pooling1d_6 (MaxPooling1D)  (None, 250, 128)     0           activation_12[0][0]
__________________________________________________________________________________________________
conv1d_13 (Conv1D)              (None, 250, 256)     131328      max_pooling1d_6[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 250, 256)     1024        conv1d_13[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 250, 256)     0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
conv1d_14 (Conv1D)              (None, 250, 256)     262400      activation_13[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 250, 256)     1024        conv1d_14[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 250, 256)     0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
max_pooling1d_7 (MaxPooling1D)  (None, 125, 256)     0           activation_14[0][0]
__________________________________________________________________________________________________
conv1d_15 (Conv1D)              (None, 125, 512)     524800      max_pooling1d_7[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 125, 512)     2048        conv1d_15[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 125, 512)     0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
conv1d_16 (Conv1D)              (None, 125, 512)     1049088     activation_15[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 125, 512)     2048        conv1d_16[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 125, 512)     0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
max_pooling1d_8 (MaxPooling1D)  (None, 62, 512)      0           activation_16[0][0]
__________________________________________________________________________________________________
conv1d_17 (Conv1D)              (None, 62, 1024)     2098176     max_pooling1d_8[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 62, 1024)     4096        conv1d_17[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 62, 1024)     0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
conv1d_18 (Conv1D)              (None, 62, 1024)     4195328     activation_17[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 62, 1024)     4096        conv1d_18[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 62, 1024)     0           batch_normalization_18[0][0]
__________________________________________________________________________________________________
max_pooling1d_9 (MaxPooling1D)  (None, 31, 1024)     0           activation_18[0][0]
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 1024)         0           max_pooling1d_9[0][0]
__________________________________________________________________________________________________
global_average_pooling1d_1 (Glo (None, 1024)         0           max_pooling1d_9[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 2048)         0           global_max_pooling1d_1[0][0]
                                                                 global_average_pooling1d_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1024)         2098176     concatenate_1[0][0]
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 1024)         4096        dense_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 1024)         0           batch_normalization_19[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1024)         1049600     dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 1024)         4096        dense_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 1024)         0           batch_normalization_20[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 31)           31775       dropout_2[0][0]
==================================================================================================
Total params: 11,596,743
Trainable params: 11,584,471
Non-trainable params: 12,272
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 21:04:50 =========
Epoch 1/65535
420/420 [==============================] - 410s 975ms/step - loss: 0.1158 - categorical_accuracy: 0.2922 - val_loss: 0.1320 - val_categorical_accuracy: 0.2593
Epoch 2/65535
420/420 [==============================] - 393s 936ms/step - loss: 0.0495 - categorical_accuracy: 0.7118 - val_loss: 0.0660 - val_categorical_accuracy: 0.6461
Epoch 3/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0329 - categorical_accuracy: 0.8090 - val_loss: 0.0357 - val_categorical_accuracy: 0.7941
Epoch 4/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0268 - categorical_accuracy: 0.8461 - val_loss: 0.0391 - val_categorical_accuracy: 0.7895
Epoch 5/65535
420/420 [==============================] - 383s 911ms/step - loss: 0.0235 - categorical_accuracy: 0.8639 - val_loss: 0.0325 - val_categorical_accuracy: 0.8109
Epoch 6/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0210 - categorical_accuracy: 0.8809 - val_loss: 0.0245 - val_categorical_accuracy: 0.8637
Epoch 7/65535
420/420 [==============================] - 383s 911ms/step - loss: 0.0194 - categorical_accuracy: 0.8913 - val_loss: 0.0303 - val_categorical_accuracy: 0.8216
Epoch 8/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0182 - categorical_accuracy: 0.8968 - val_loss: 0.0266 - val_categorical_accuracy: 0.8529
Epoch 9/65535
420/420 [==============================] - 383s 913ms/step - loss: 0.0169 - categorical_accuracy: 0.9040 - val_loss: 0.0221 - val_categorical_accuracy: 0.8759
Epoch 10/65535
420/420 [==============================] - 383s 913ms/step - loss: 0.0161 - categorical_accuracy: 0.9102 - val_loss: 0.0214 - val_categorical_accuracy: 0.8809
Epoch 11/65535
420/420 [==============================] - 386s 919ms/step - loss: 0.0154 - categorical_accuracy: 0.9131 - val_loss: 0.0165 - val_categorical_accuracy: 0.9105
Epoch 12/65535
420/420 [==============================] - 384s 915ms/step - loss: 0.0146 - categorical_accuracy: 0.9169 - val_loss: 0.0165 - val_categorical_accuracy: 0.9070
Epoch 13/65535
420/420 [==============================] - 388s 924ms/step - loss: 0.0140 - categorical_accuracy: 0.9203 - val_loss: 0.0196 - val_categorical_accuracy: 0.8902
Epoch 14/65535
420/420 [==============================] - 384s 913ms/step - loss: 0.0139 - categorical_accuracy: 0.9216 - val_loss: 0.0188 - val_categorical_accuracy: 0.8966
Epoch 15/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0131 - categorical_accuracy: 0.9266 - val_loss: 0.0160 - val_categorical_accuracy: 0.9132
Epoch 16/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0128 - categorical_accuracy: 0.9282 - val_loss: 0.0177 - val_categorical_accuracy: 0.9065
Epoch 17/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0122 - categorical_accuracy: 0.9312 - val_loss: 0.0142 - val_categorical_accuracy: 0.9225
Epoch 18/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0119 - categorical_accuracy: 0.9328 - val_loss: 0.0139 - val_categorical_accuracy: 0.9230
Epoch 19/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0115 - categorical_accuracy: 0.9346 - val_loss: 0.0149 - val_categorical_accuracy: 0.9210
Epoch 20/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0110 - categorical_accuracy: 0.9381 - val_loss: 0.0138 - val_categorical_accuracy: 0.9221
Epoch 21/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0111 - categorical_accuracy: 0.9374 - val_loss: 0.0138 - val_categorical_accuracy: 0.9250
Epoch 22/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0108 - categorical_accuracy: 0.9393 - val_loss: 0.0150 - val_categorical_accuracy: 0.9190
Epoch 23/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0102 - categorical_accuracy: 0.9421 - val_loss: 0.0141 - val_categorical_accuracy: 0.9253
Epoch 24/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0101 - categorical_accuracy: 0.9439 - val_loss: 0.0144 - val_categorical_accuracy: 0.9264
Epoch 25/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0097 - categorical_accuracy: 0.9460 - val_loss: 0.0141 - val_categorical_accuracy: 0.9285
Epoch 26/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0097 - categorical_accuracy: 0.9448 - val_loss: 0.0136 - val_categorical_accuracy: 0.9285
Epoch 27/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0095 - categorical_accuracy: 0.9471 - val_loss: 0.0127 - val_categorical_accuracy: 0.9336
Epoch 28/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0092 - categorical_accuracy: 0.9478 - val_loss: 0.0114 - val_categorical_accuracy: 0.9387
Epoch 29/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0089 - categorical_accuracy: 0.9506 - val_loss: 0.0159 - val_categorical_accuracy: 0.9169
Epoch 30/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0089 - categorical_accuracy: 0.9501 - val_loss: 0.0138 - val_categorical_accuracy: 0.9280
Epoch 31/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0087 - categorical_accuracy: 0.9509 - val_loss: 0.0142 - val_categorical_accuracy: 0.9236
Epoch 32/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0084 - categorical_accuracy: 0.9532 - val_loss: 0.0119 - val_categorical_accuracy: 0.9392
Epoch 33/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0084 - categorical_accuracy: 0.9535 - val_loss: 0.0136 - val_categorical_accuracy: 0.9300
Epoch 34/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0080 - categorical_accuracy: 0.9551 - val_loss: 0.0116 - val_categorical_accuracy: 0.9398
Epoch 00034: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 35/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0063 - categorical_accuracy: 0.9650 - val_loss: 0.0088 - val_categorical_accuracy: 0.9542
Epoch 36/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0057 - categorical_accuracy: 0.9692 - val_loss: 0.0090 - val_categorical_accuracy: 0.9546
Epoch 37/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0055 - categorical_accuracy: 0.9694 - val_loss: 0.0091 - val_categorical_accuracy: 0.9539
Epoch 38/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0053 - categorical_accuracy: 0.9713 - val_loss: 0.0085 - val_categorical_accuracy: 0.9561
Epoch 39/65535
420/420 [==============================] - 379s 901ms/step - loss: 0.0052 - categorical_accuracy: 0.9713 - val_loss: 0.0086 - val_categorical_accuracy: 0.9562
Epoch 40/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0052 - categorical_accuracy: 0.9718 - val_loss: 0.0084 - val_categorical_accuracy: 0.9573
Epoch 41/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0050 - categorical_accuracy: 0.9721 - val_loss: 0.0089 - val_categorical_accuracy: 0.9539
Epoch 42/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0051 - categorical_accuracy: 0.9721 - val_loss: 0.0087 - val_categorical_accuracy: 0.9553
Epoch 43/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0048 - categorical_accuracy: 0.9741 - val_loss: 0.0088 - val_categorical_accuracy: 0.9548
Epoch 44/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0047 - categorical_accuracy: 0.9736 - val_loss: 0.0085 - val_categorical_accuracy: 0.9568
Epoch 45/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0049 - categorical_accuracy: 0.9730 - val_loss: 0.0085 - val_categorical_accuracy: 0.9574
Epoch 46/65535
420/420 [==============================] - 379s 901ms/step - loss: 0.0048 - categorical_accuracy: 0.9728 - val_loss: 0.0090 - val_categorical_accuracy: 0.9543
Epoch 00046: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 47/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0046 - categorical_accuracy: 0.9742 - val_loss: 0.0087 - val_categorical_accuracy: 0.9557
Epoch 48/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0045 - categorical_accuracy: 0.9750 - val_loss: 0.0088 - val_categorical_accuracy: 0.9556
Epoch 49/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0044 - categorical_accuracy: 0.9749 - val_loss: 0.0087 - val_categorical_accuracy: 0.9562
Epoch 50/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0043 - categorical_accuracy: 0.9766 - val_loss: 0.0087 - val_categorical_accuracy: 0.9555
Epoch 51/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0044 - categorical_accuracy: 0.9753 - val_loss: 0.0087 - val_categorical_accuracy: 0.9552
Epoch 00051: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 52/65535
420/420 [==============================] - 379s 901ms/step - loss: 0.0043 - categorical_accuracy: 0.9757 - val_loss: 0.0086 - val_categorical_accuracy: 0.9557
Epoch 53/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0043 - categorical_accuracy: 0.9762 - val_loss: 0.0086 - val_categorical_accuracy: 0.9561
Epoch 54/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0042 - categorical_accuracy: 0.9779 - val_loss: 0.0086 - val_categorical_accuracy: 0.9561
Epoch 55/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0042 - categorical_accuracy: 0.9762 - val_loss: 0.0086 - val_categorical_accuracy: 0.9562
Epoch 00055: early stopping
========= generating oof predictions 02:53:37 =========
========= generating test set predictions 02:53:44 =========
========= fitting 2 th model 02:55:10 =========
Epoch 1/65535
420/420 [==============================] - 384s 914ms/step - loss: 0.1067 - categorical_accuracy: 0.3481 - val_loss: 0.1161 - val_categorical_accuracy: 0.4136
Epoch 2/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0435 - categorical_accuracy: 0.7445 - val_loss: 0.0529 - val_categorical_accuracy: 0.7028
Epoch 3/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0311 - categorical_accuracy: 0.8215 - val_loss: 0.0625 - val_categorical_accuracy: 0.6869
Epoch 4/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0256 - categorical_accuracy: 0.8532 - val_loss: 0.0340 - val_categorical_accuracy: 0.8109
Epoch 5/65535
420/420 [==============================] - 378s 899ms/step - loss: 0.0229 - categorical_accuracy: 0.8704 - val_loss: 0.0323 - val_categorical_accuracy: 0.8264
Epoch 6/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0203 - categorical_accuracy: 0.8848 - val_loss: 0.0213 - val_categorical_accuracy: 0.8812
Epoch 7/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0189 - categorical_accuracy: 0.8935 - val_loss: 0.0209 - val_categorical_accuracy: 0.8848
Epoch 8/65535
420/420 [==============================] - 378s 899ms/step - loss: 0.0176 - categorical_accuracy: 0.9003 - val_loss: 0.0177 - val_categorical_accuracy: 0.9037
Epoch 9/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0168 - categorical_accuracy: 0.9052 - val_loss: 0.0175 - val_categorical_accuracy: 0.9025
Epoch 10/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0156 - categorical_accuracy: 0.9117 - val_loss: 0.0166 - val_categorical_accuracy: 0.9068
Epoch 11/65535
420/420 [==============================] - 378s 899ms/step - loss: 0.0152 - categorical_accuracy: 0.9137 - val_loss: 0.0187 - val_categorical_accuracy: 0.8964
Epoch 12/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0145 - categorical_accuracy: 0.9180 - val_loss: 0.0144 - val_categorical_accuracy: 0.9189
Epoch 13/65535
420/420 [==============================] - 378s 899ms/step - loss: 0.0137 - categorical_accuracy: 0.9219 - val_loss: 0.0163 - val_categorical_accuracy: 0.9113
Epoch 14/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0128 - categorical_accuracy: 0.9278 - val_loss: 0.0148 - val_categorical_accuracy: 0.9189
Epoch 15/65535
420/420 [==============================] - 377s 899ms/step - loss: 0.0124 - categorical_accuracy: 0.9302 - val_loss: 0.0145 - val_categorical_accuracy: 0.9194
Epoch 16/65535
420/420 [==============================] - 380s 904ms/step - loss: 0.0124 - categorical_accuracy: 0.9305 - val_loss: 0.0140 - val_categorical_accuracy: 0.9247
Epoch 17/65535
420/420 [==============================] - 377s 899ms/step - loss: 0.0119 - categorical_accuracy: 0.9324 - val_loss: 0.0153 - val_categorical_accuracy: 0.9192
Epoch 18/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0115 - categorical_accuracy: 0.9350 - val_loss: 0.0130 - val_categorical_accuracy: 0.9265
Epoch 19/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0110 - categorical_accuracy: 0.9377 - val_loss: 0.0145 - val_categorical_accuracy: 0.9214
Epoch 20/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0110 - categorical_accuracy: 0.9374 - val_loss: 0.0130 - val_categorical_accuracy: 0.9279
Epoch 21/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0108 - categorical_accuracy: 0.9395 - val_loss: 0.0156 - val_categorical_accuracy: 0.9195
Epoch 22/65535
420/420 [==============================] - 377s 899ms/step - loss: 0.0103 - categorical_accuracy: 0.9423 - val_loss: 0.0133 - val_categorical_accuracy: 0.9286
Epoch 23/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0101 - categorical_accuracy: 0.9435 - val_loss: 0.0123 - val_categorical_accuracy: 0.9325
Epoch 24/65535
420/420 [==============================] - 378s 899ms/step - loss: 0.0100 - categorical_accuracy: 0.9432 - val_loss: 0.0123 - val_categorical_accuracy: 0.9313
Epoch 25/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0093 - categorical_accuracy: 0.9482 - val_loss: 0.0134 - val_categorical_accuracy: 0.9293
Epoch 26/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0092 - categorical_accuracy: 0.9476 - val_loss: 0.0142 - val_categorical_accuracy: 0.9212
Epoch 27/65535
420/420 [==============================] - 379s 901ms/step - loss: 0.0090 - categorical_accuracy: 0.9486 - val_loss: 0.0137 - val_categorical_accuracy: 0.9282
Epoch 28/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0088 - categorical_accuracy: 0.9502 - val_loss: 0.0130 - val_categorical_accuracy: 0.9332
Epoch 29/65535
420/420 [==============================] - 379s 904ms/step - loss: 0.0088 - categorical_accuracy: 0.9511 - val_loss: 0.0128 - val_categorical_accuracy: 0.9311
Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 30/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0069 - categorical_accuracy: 0.9611 - val_loss: 0.0092 - val_categorical_accuracy: 0.9506
Epoch 31/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0061 - categorical_accuracy: 0.9662 - val_loss: 0.0087 - val_categorical_accuracy: 0.9533
Epoch 32/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0060 - categorical_accuracy: 0.9667 - val_loss: 0.0088 - val_categorical_accuracy: 0.9527
Epoch 33/65535
420/420 [==============================] - 377s 899ms/step - loss: 0.0059 - categorical_accuracy: 0.9671 - val_loss: 0.0087 - val_categorical_accuracy: 0.9543
Epoch 34/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0058 - categorical_accuracy: 0.9673 - val_loss: 0.0095 - val_categorical_accuracy: 0.9497
Epoch 35/65535
420/420 [==============================] - 378s 899ms/step - loss: 0.0056 - categorical_accuracy: 0.9690 - val_loss: 0.0085 - val_categorical_accuracy: 0.9552
Epoch 36/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0056 - categorical_accuracy: 0.9693 - val_loss: 0.0091 - val_categorical_accuracy: 0.9526
Epoch 37/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0055 - categorical_accuracy: 0.9698 - val_loss: 0.0088 - val_categorical_accuracy: 0.9541
Epoch 38/65535
420/420 [==============================] - 382s 910ms/step - loss: 0.0054 - categorical_accuracy: 0.9702 - val_loss: 0.0090 - val_categorical_accuracy: 0.9521
Epoch 39/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0054 - categorical_accuracy: 0.9701 - val_loss: 0.0090 - val_categorical_accuracy: 0.9524
Epoch 40/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0053 - categorical_accuracy: 0.9712 - val_loss: 0.0097 - val_categorical_accuracy: 0.9508
Epoch 41/65535
420/420 [==============================] - 378s 901ms/step - loss: 0.0051 - categorical_accuracy: 0.9715 - val_loss: 0.0088 - val_categorical_accuracy: 0.9540
Epoch 00041: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 42/65535
420/420 [==============================] - 379s 902ms/step - loss: 0.0048 - categorical_accuracy: 0.9730 - val_loss: 0.0088 - val_categorical_accuracy: 0.9545
Epoch 43/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0049 - categorical_accuracy: 0.9730 - val_loss: 0.0089 - val_categorical_accuracy: 0.9532
Epoch 44/65535
420/420 [==============================] - 379s 903ms/step - loss: 0.0048 - categorical_accuracy: 0.9741 - val_loss: 0.0089 - val_categorical_accuracy: 0.9540
Epoch 45/65535
420/420 [==============================] - 395s 941ms/step - loss: 0.0047 - categorical_accuracy: 0.9742 - val_loss: 0.0086 - val_categorical_accuracy: 0.9564
Epoch 46/65535
420/420 [==============================] - 402s 957ms/step - loss: 0.0046 - categorical_accuracy: 0.9744 - val_loss: 0.0087 - val_categorical_accuracy: 0.9553
Epoch 00046: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 47/65535
420/420 [==============================] - 384s 915ms/step - loss: 0.0048 - categorical_accuracy: 0.9733 - val_loss: 0.0085 - val_categorical_accuracy: 0.9561
Epoch 48/65535
420/420 [==============================] - 378s 900ms/step - loss: 0.0045 - categorical_accuracy: 0.9747 - val_loss: 0.0086 - val_categorical_accuracy: 0.9561
Epoch 49/65535
420/420 [==============================] - 385s 916ms/step - loss: 0.0046 - categorical_accuracy: 0.9743 - val_loss: 0.0086 - val_categorical_accuracy: 0.9557
Epoch 50/65535
420/420 [==============================] - 391s 931ms/step - loss: 0.0045 - categorical_accuracy: 0.9749 - val_loss: 0.0086 - val_categorical_accuracy: 0.9560
Epoch 00050: early stopping
========= generating oof predictions 08:12:00 =========
========= generating test set predictions 08:12:07 =========
========= fitting 3 th model 11:24:17 =========
Epoch 1/65535
420/420 [==============================] - 396s 942ms/step - loss: 0.1111 - categorical_accuracy: 0.3225 - val_loss: 0.1086 - val_categorical_accuracy: 0.3790
Epoch 2/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0514 - categorical_accuracy: 0.7040 - val_loss: 0.0879 - val_categorical_accuracy: 0.5014
Epoch 3/65535
420/420 [==============================] - 385s 916ms/step - loss: 0.0348 - categorical_accuracy: 0.7961 - val_loss: 0.0442 - val_categorical_accuracy: 0.7519
Epoch 4/65535
420/420 [==============================] - 382s 911ms/step - loss: 0.0277 - categorical_accuracy: 0.8406 - val_loss: 0.0455 - val_categorical_accuracy: 0.7398
Epoch 5/65535
420/420 [==============================] - 383s 912ms/step - loss: 0.0235 - categorical_accuracy: 0.8648 - val_loss: 0.0256 - val_categorical_accuracy: 0.8555
Epoch 6/65535
420/420 [==============================] - 385s 917ms/step - loss: 0.0213 - categorical_accuracy: 0.8775 - val_loss: 0.0230 - val_categorical_accuracy: 0.8725
Epoch 7/65535
420/420 [==============================] - 384s 914ms/step - loss: 0.0191 - categorical_accuracy: 0.8917 - val_loss: 0.0243 - val_categorical_accuracy: 0.8658
Epoch 8/65535
420/420 [==============================] - 384s 914ms/step - loss: 0.0178 - categorical_accuracy: 0.8987 - val_loss: 0.0220 - val_categorical_accuracy: 0.8796
Epoch 9/65535
420/420 [==============================] - 386s 919ms/step - loss: 0.0173 - categorical_accuracy: 0.9023 - val_loss: 0.0189 - val_categorical_accuracy: 0.8966
Epoch 10/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0161 - categorical_accuracy: 0.9091 - val_loss: 0.0235 - val_categorical_accuracy: 0.8707
Epoch 11/65535
420/420 [==============================] - 387s 922ms/step - loss: 0.0151 - categorical_accuracy: 0.9151 - val_loss: 0.0180 - val_categorical_accuracy: 0.9015
Epoch 12/65535
420/420 [==============================] - 384s 913ms/step - loss: 0.0145 - categorical_accuracy: 0.9177 - val_loss: 0.0158 - val_categorical_accuracy: 0.9172
Epoch 13/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0141 - categorical_accuracy: 0.9210 - val_loss: 0.0199 - val_categorical_accuracy: 0.8953
Epoch 14/65535
420/420 [==============================] - 386s 918ms/step - loss: 0.0135 - categorical_accuracy: 0.9233 - val_loss: 0.0206 - val_categorical_accuracy: 0.8878
Epoch 15/65535
420/420 [==============================] - 383s 912ms/step - loss: 0.0129 - categorical_accuracy: 0.9266 - val_loss: 0.0147 - val_categorical_accuracy: 0.9209
Epoch 16/65535
420/420 [==============================] - 382s 910ms/step - loss: 0.0123 - categorical_accuracy: 0.9299 - val_loss: 0.0159 - val_categorical_accuracy: 0.9150
Epoch 17/65535
420/420 [==============================] - 386s 918ms/step - loss: 0.0118 - categorical_accuracy: 0.9335 - val_loss: 0.0165 - val_categorical_accuracy: 0.9113
Epoch 18/65535
420/420 [==============================] - 383s 912ms/step - loss: 0.0117 - categorical_accuracy: 0.9338 - val_loss: 0.0147 - val_categorical_accuracy: 0.9231
Epoch 19/65535
420/420 [==============================] - 382s 910ms/step - loss: 0.0111 - categorical_accuracy: 0.9362 - val_loss: 0.0143 - val_categorical_accuracy: 0.9215
Epoch 20/65535
420/420 [==============================] - 385s 916ms/step - loss: 0.0111 - categorical_accuracy: 0.9361 - val_loss: 0.0120 - val_categorical_accuracy: 0.9375
Epoch 21/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0107 - categorical_accuracy: 0.9391 - val_loss: 0.0149 - val_categorical_accuracy: 0.9230
Epoch 22/65535
420/420 [==============================] - 386s 918ms/step - loss: 0.0105 - categorical_accuracy: 0.9410 - val_loss: 0.0126 - val_categorical_accuracy: 0.9344
Epoch 23/65535
420/420 [==============================] - 382s 910ms/step - loss: 0.0102 - categorical_accuracy: 0.9425 - val_loss: 0.0151 - val_categorical_accuracy: 0.9208
Epoch 24/65535
420/420 [==============================] - 382s 910ms/step - loss: 0.0099 - categorical_accuracy: 0.9438 - val_loss: 0.0120 - val_categorical_accuracy: 0.9367
Epoch 25/65535
420/420 [==============================] - 382s 910ms/step - loss: 0.0098 - categorical_accuracy: 0.9444 - val_loss: 0.0149 - val_categorical_accuracy: 0.9202
Epoch 26/65535
420/420 [==============================] - 385s 917ms/step - loss: 0.0095 - categorical_accuracy: 0.9461 - val_loss: 0.0136 - val_categorical_accuracy: 0.9289
Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 27/65535
420/420 [==============================] - 387s 921ms/step - loss: 0.0075 - categorical_accuracy: 0.9583 - val_loss: 0.0092 - val_categorical_accuracy: 0.9522
Epoch 28/65535
420/420 [==============================] - 391s 930ms/step - loss: 0.0069 - categorical_accuracy: 0.9613 - val_loss: 0.0092 - val_categorical_accuracy: 0.9523
Epoch 29/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0065 - categorical_accuracy: 0.9637 - val_loss: 0.0093 - val_categorical_accuracy: 0.9520
Epoch 30/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0063 - categorical_accuracy: 0.9648 - val_loss: 0.0093 - val_categorical_accuracy: 0.9514
Epoch 31/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0062 - categorical_accuracy: 0.9656 - val_loss: 0.0089 - val_categorical_accuracy: 0.9540
Epoch 32/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0062 - categorical_accuracy: 0.9657 - val_loss: 0.0094 - val_categorical_accuracy: 0.9515
Epoch 33/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0060 - categorical_accuracy: 0.9669 - val_loss: 0.0089 - val_categorical_accuracy: 0.9537
Epoch 34/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0059 - categorical_accuracy: 0.9670 - val_loss: 0.0094 - val_categorical_accuracy: 0.9516
Epoch 35/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0059 - categorical_accuracy: 0.9671 - val_loss: 0.0087 - val_categorical_accuracy: 0.9540
Epoch 36/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0058 - categorical_accuracy: 0.9678 - val_loss: 0.0091 - val_categorical_accuracy: 0.9531
Epoch 37/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0057 - categorical_accuracy: 0.9680 - val_loss: 0.0089 - val_categorical_accuracy: 0.9537
Epoch 38/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0056 - categorical_accuracy: 0.9688 - val_loss: 0.0092 - val_categorical_accuracy: 0.9527
Epoch 39/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0055 - categorical_accuracy: 0.9689 - val_loss: 0.0093 - val_categorical_accuracy: 0.9528
Epoch 40/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0054 - categorical_accuracy: 0.9695 - val_loss: 0.0094 - val_categorical_accuracy: 0.9498
Epoch 41/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0052 - categorical_accuracy: 0.9704 - val_loss: 0.0092 - val_categorical_accuracy: 0.9539
Epoch 00041: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 42/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0052 - categorical_accuracy: 0.9712 - val_loss: 0.0089 - val_categorical_accuracy: 0.9547
Epoch 43/65535
420/420 [==============================] - 380s 904ms/step - loss: 0.0050 - categorical_accuracy: 0.9720 - val_loss: 0.0089 - val_categorical_accuracy: 0.9542
Epoch 44/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0051 - categorical_accuracy: 0.9721 - val_loss: 0.0087 - val_categorical_accuracy: 0.9549
Epoch 45/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0049 - categorical_accuracy: 0.9729 - val_loss: 0.0089 - val_categorical_accuracy: 0.9543
Epoch 46/65535
420/420 [==============================] - 380s 904ms/step - loss: 0.0049 - categorical_accuracy: 0.9721 - val_loss: 0.0085 - val_categorical_accuracy: 0.9563
Epoch 47/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0048 - categorical_accuracy: 0.9729 - val_loss: 0.0086 - val_categorical_accuracy: 0.9555
Epoch 48/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0047 - categorical_accuracy: 0.9741 - val_loss: 0.0087 - val_categorical_accuracy: 0.9557
Epoch 49/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0049 - categorical_accuracy: 0.9727 - val_loss: 0.0088 - val_categorical_accuracy: 0.9546
Epoch 50/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0048 - categorical_accuracy: 0.9731 - val_loss: 0.0087 - val_categorical_accuracy: 0.9554
Epoch 51/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0045 - categorical_accuracy: 0.9751 - val_loss: 0.0086 - val_categorical_accuracy: 0.9568
Epoch 52/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0047 - categorical_accuracy: 0.9738 - val_loss: 0.0089 - val_categorical_accuracy: 0.9548
Epoch 00052: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 53/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0047 - categorical_accuracy: 0.9739 - val_loss: 0.0088 - val_categorical_accuracy: 0.9549
Epoch 54/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0048 - categorical_accuracy: 0.9728 - val_loss: 0.0088 - val_categorical_accuracy: 0.9551
Epoch 55/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0047 - categorical_accuracy: 0.9734 - val_loss: 0.0087 - val_categorical_accuracy: 0.9555
Epoch 56/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0045 - categorical_accuracy: 0.9746 - val_loss: 0.0088 - val_categorical_accuracy: 0.9551
Epoch 57/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0046 - categorical_accuracy: 0.9748 - val_loss: 0.0089 - val_categorical_accuracy: 0.9548
Epoch 00057: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 58/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0046 - categorical_accuracy: 0.9744 - val_loss: 0.0088 - val_categorical_accuracy: 0.9551
Epoch 59/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0046 - categorical_accuracy: 0.9740 - val_loss: 0.0088 - val_categorical_accuracy: 0.9551
Epoch 60/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0046 - categorical_accuracy: 0.9748 - val_loss: 0.0088 - val_categorical_accuracy: 0.9554
Epoch 61/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0045 - categorical_accuracy: 0.9749 - val_loss: 0.0087 - val_categorical_accuracy: 0.9559
Epoch 00061: early stopping
========= generating oof predictions 17:53:01 =========
========= generating test set predictions 17:53:07 =========
========= fitting 4 th model 17:54:34 =========
Epoch 1/65535
420/420 [==============================] - 387s 921ms/step - loss: 0.1015 - categorical_accuracy: 0.3800 - val_loss: 0.0728 - val_categorical_accuracy: 0.5840
Epoch 2/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0431 - categorical_accuracy: 0.7459 - val_loss: 0.0640 - val_categorical_accuracy: 0.6829
Epoch 3/65535
420/420 [==============================] - 382s 908ms/step - loss: 0.0306 - categorical_accuracy: 0.8242 - val_loss: 0.0267 - val_categorical_accuracy: 0.8497
Epoch 4/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0254 - categorical_accuracy: 0.8521 - val_loss: 0.0251 - val_categorical_accuracy: 0.8601
Epoch 5/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0224 - categorical_accuracy: 0.8723 - val_loss: 0.0226 - val_categorical_accuracy: 0.8782
Epoch 6/65535
420/420 [==============================] - 382s 908ms/step - loss: 0.0199 - categorical_accuracy: 0.8865 - val_loss: 0.0178 - val_categorical_accuracy: 0.9055
Epoch 7/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0191 - categorical_accuracy: 0.8915 - val_loss: 0.0166 - val_categorical_accuracy: 0.9094
Epoch 8/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0173 - categorical_accuracy: 0.9031 - val_loss: 0.0182 - val_categorical_accuracy: 0.8974
Epoch 9/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0165 - categorical_accuracy: 0.9078 - val_loss: 0.0168 - val_categorical_accuracy: 0.9084
Epoch 10/65535
420/420 [==============================] - 382s 910ms/step - loss: 0.0157 - categorical_accuracy: 0.9119 - val_loss: 0.0147 - val_categorical_accuracy: 0.9194
Epoch 11/65535
420/420 [==============================] - 397s 945ms/step - loss: 0.0146 - categorical_accuracy: 0.9177 - val_loss: 0.0170 - val_categorical_accuracy: 0.9076
Epoch 12/65535
420/420 [==============================] - 386s 919ms/step - loss: 0.0145 - categorical_accuracy: 0.9191 - val_loss: 0.0148 - val_categorical_accuracy: 0.9203
Epoch 13/65535
420/420 [==============================] - 394s 937ms/step - loss: 0.0138 - categorical_accuracy: 0.9220 - val_loss: 0.0147 - val_categorical_accuracy: 0.9194
Epoch 14/65535
420/420 [==============================] - 395s 940ms/step - loss: 0.0131 - categorical_accuracy: 0.9257 - val_loss: 0.0159 - val_categorical_accuracy: 0.9142
Epoch 15/65535
420/420 [==============================] - 395s 941ms/step - loss: 0.0128 - categorical_accuracy: 0.9301 - val_loss: 0.0139 - val_categorical_accuracy: 0.9273
Epoch 16/65535
420/420 [==============================] - 385s 918ms/step - loss: 0.0122 - categorical_accuracy: 0.9319 - val_loss: 0.0128 - val_categorical_accuracy: 0.9335
Epoch 17/65535
420/420 [==============================] - 384s 914ms/step - loss: 0.0119 - categorical_accuracy: 0.9318 - val_loss: 0.0132 - val_categorical_accuracy: 0.9289
Epoch 18/65535
420/420 [==============================] - 385s 916ms/step - loss: 0.0115 - categorical_accuracy: 0.9352 - val_loss: 0.0130 - val_categorical_accuracy: 0.9304
Epoch 19/65535
420/420 [==============================] - 383s 912ms/step - loss: 0.0114 - categorical_accuracy: 0.9363 - val_loss: 0.0105 - val_categorical_accuracy: 0.9424
Epoch 20/65535
420/420 [==============================] - 383s 912ms/step - loss: 0.0109 - categorical_accuracy: 0.9389 - val_loss: 0.0111 - val_categorical_accuracy: 0.9387
Epoch 21/65535
420/420 [==============================] - 394s 938ms/step - loss: 0.0106 - categorical_accuracy: 0.9403 - val_loss: 0.0156 - val_categorical_accuracy: 0.9193
Epoch 22/65535
420/420 [==============================] - 385s 916ms/step - loss: 0.0103 - categorical_accuracy: 0.9421 - val_loss: 0.0119 - val_categorical_accuracy: 0.9369
Epoch 23/65535
420/420 [==============================] - 384s 915ms/step - loss: 0.0100 - categorical_accuracy: 0.9429 - val_loss: 0.0130 - val_categorical_accuracy: 0.9295
Epoch 24/65535
420/420 [==============================] - 384s 915ms/step - loss: 0.0098 - categorical_accuracy: 0.9443 - val_loss: 0.0130 - val_categorical_accuracy: 0.9317
Epoch 25/65535
420/420 [==============================] - 385s 916ms/step - loss: 0.0095 - categorical_accuracy: 0.9462 - val_loss: 0.0128 - val_categorical_accuracy: 0.9339

Epoch 00025: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 26/65535
420/420 [==============================] - 385s 916ms/step - loss: 0.0076 - categorical_accuracy: 0.9578 - val_loss: 0.0088 - val_categorical_accuracy: 0.9541
Epoch 27/65535
420/420 [==============================] - 385s 917ms/step - loss: 0.0070 - categorical_accuracy: 0.9612 - val_loss: 0.0085 - val_categorical_accuracy: 0.9559
Epoch 28/65535
420/420 [==============================] - 385s 916ms/step - loss: 0.0066 - categorical_accuracy: 0.9635 - val_loss: 0.0086 - val_categorical_accuracy: 0.9560
Epoch 29/65535
420/420 [==============================] - 392s 934ms/step - loss: 0.0065 - categorical_accuracy: 0.9634 - val_loss: 0.0092 - val_categorical_accuracy: 0.9507
Epoch 30/65535
420/420 [==============================] - 402s 958ms/step - loss: 0.0064 - categorical_accuracy: 0.9642 - val_loss: 0.0087 - val_categorical_accuracy: 0.9536
Epoch 31/65535
420/420 [==============================] - 401s 956ms/step - loss: 0.0064 - categorical_accuracy: 0.9644 - val_loss: 0.0085 - val_categorical_accuracy: 0.9552
Epoch 32/65535
420/420 [==============================] - 388s 924ms/step - loss: 0.0061 - categorical_accuracy: 0.9660 - val_loss: 0.0082 - val_categorical_accuracy: 0.9564
Epoch 33/65535
420/420 [==============================] - 394s 938ms/step - loss: 0.0060 - categorical_accuracy: 0.9669 - val_loss: 0.0083 - val_categorical_accuracy: 0.9565
Epoch 34/65535
420/420 [==============================] - 391s 932ms/step - loss: 0.0059 - categorical_accuracy: 0.9672 - val_loss: 0.0084 - val_categorical_accuracy: 0.9562
Epoch 35/65535
420/420 [==============================] - 408s 972ms/step - loss: 0.0059 - categorical_accuracy: 0.9677 - val_loss: 0.0089 - val_categorical_accuracy: 0.9533
Epoch 36/65535
420/420 [==============================] - 403s 959ms/step - loss: 0.0057 - categorical_accuracy: 0.9690 - val_loss: 0.0083 - val_categorical_accuracy: 0.9551
Epoch 37/65535
420/420 [==============================] - 393s 935ms/step - loss: 0.0057 - categorical_accuracy: 0.9688 - val_loss: 0.0086 - val_categorical_accuracy: 0.9545
Epoch 38/65535
420/420 [==============================] - 401s 954ms/step - loss: 0.0055 - categorical_accuracy: 0.9697 - val_loss: 0.0089 - val_categorical_accuracy: 0.9536

Epoch 00038: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 39/65535
420/420 [==============================] - 393s 935ms/step - loss: 0.0054 - categorical_accuracy: 0.9696 - val_loss: 0.0085 - val_categorical_accuracy: 0.9554
Epoch 40/65535
420/420 [==============================] - 391s 930ms/step - loss: 0.0052 - categorical_accuracy: 0.9710 - val_loss: 0.0082 - val_categorical_accuracy: 0.9572
Epoch 41/65535
420/420 [==============================] - 392s 934ms/step - loss: 0.0053 - categorical_accuracy: 0.9707 - val_loss: 0.0085 - val_categorical_accuracy: 0.9557
Epoch 42/65535
420/420 [==============================] - 397s 946ms/step - loss: 0.0052 - categorical_accuracy: 0.9710 - val_loss: 0.0083 - val_categorical_accuracy: 0.9565
Epoch 43/65535
420/420 [==============================] - 389s 925ms/step - loss: 0.0051 - categorical_accuracy: 0.9723 - val_loss: 0.0084 - val_categorical_accuracy: 0.9564

Epoch 00043: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 44/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0052 - categorical_accuracy: 0.9721 - val_loss: 0.0084 - val_categorical_accuracy: 0.9568
Epoch 45/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0049 - categorical_accuracy: 0.9728 - val_loss: 0.0085 - val_categorical_accuracy: 0.9552
Epoch 46/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0050 - categorical_accuracy: 0.9725 - val_loss: 0.0083 - val_categorical_accuracy: 0.9568
Epoch 47/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0050 - categorical_accuracy: 0.9728 - val_loss: 0.0084 - val_categorical_accuracy: 0.9560
Epoch 48/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0050 - categorical_accuracy: 0.9723 - val_loss: 0.0085 - val_categorical_accuracy: 0.9554

Epoch 00048: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 49/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0049 - categorical_accuracy: 0.9731 - val_loss: 0.0085 - val_categorical_accuracy: 0.9557
Epoch 50/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0050 - categorical_accuracy: 0.9724 - val_loss: 0.0085 - val_categorical_accuracy: 0.9556
Epoch 51/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0049 - categorical_accuracy: 0.9733 - val_loss: 0.0085 - val_categorical_accuracy: 0.9559
Epoch 52/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0052 - categorical_accuracy: 0.9715 - val_loss: 0.0084 - val_categorical_accuracy: 0.9561
Epoch 53/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0051 - categorical_accuracy: 0.9728 - val_loss: 0.0085 - val_categorical_accuracy: 0.9557

Epoch 00053: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 54/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0050 - categorical_accuracy: 0.9732 - val_loss: 0.0083 - val_categorical_accuracy: 0.9570
Epoch 55/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0050 - categorical_accuracy: 0.9724 - val_loss: 0.0084 - val_categorical_accuracy: 0.9562
Epoch 00055: early stopping
========= generating oof predictions 23:49:50 =========
========= generating test set predictions 23:49:57 =========
========= fitting 5 th model 23:51:23 =========
Epoch 1/65535
420/420 [==============================] - 387s 921ms/step - loss: 0.1057 - categorical_accuracy: 0.3575 - val_loss: 0.0827 - val_categorical_accuracy: 0.4930
Epoch 2/65535
420/420 [==============================] - 380s 904ms/step - loss: 0.0481 - categorical_accuracy: 0.7126 - val_loss: 0.0409 - val_categorical_accuracy: 0.7689
Epoch 3/65535
420/420 [==============================] - 382s 908ms/step - loss: 0.0327 - categorical_accuracy: 0.8084 - val_loss: 0.0443 - val_categorical_accuracy: 0.7754
Epoch 4/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0267 - categorical_accuracy: 0.8466 - val_loss: 0.0305 - val_categorical_accuracy: 0.8339
Epoch 5/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0232 - categorical_accuracy: 0.8669 - val_loss: 0.0267 - val_categorical_accuracy: 0.8552
Epoch 6/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0208 - categorical_accuracy: 0.8816 - val_loss: 0.0341 - val_categorical_accuracy: 0.8231
Epoch 7/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0190 - categorical_accuracy: 0.8916 - val_loss: 0.0189 - val_categorical_accuracy: 0.8963
Epoch 8/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0180 - categorical_accuracy: 0.8987 - val_loss: 0.0223 - val_categorical_accuracy: 0.8784
Epoch 9/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0165 - categorical_accuracy: 0.9058 - val_loss: 0.0167 - val_categorical_accuracy: 0.9093
Epoch 10/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0158 - categorical_accuracy: 0.9101 - val_loss: 0.0149 - val_categorical_accuracy: 0.9198
Epoch 11/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0151 - categorical_accuracy: 0.9148 - val_loss: 0.0190 - val_categorical_accuracy: 0.8955
Epoch 12/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0144 - categorical_accuracy: 0.9188 - val_loss: 0.0157 - val_categorical_accuracy: 0.9180
Epoch 13/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0140 - categorical_accuracy: 0.9202 - val_loss: 0.0146 - val_categorical_accuracy: 0.9212
Epoch 14/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0132 - categorical_accuracy: 0.9239 - val_loss: 0.0274 - val_categorical_accuracy: 0.8761
Epoch 15/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0128 - categorical_accuracy: 0.9275 - val_loss: 0.0130 - val_categorical_accuracy: 0.9319
Epoch 16/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0121 - categorical_accuracy: 0.9317 - val_loss: 0.0157 - val_categorical_accuracy: 0.9167
Epoch 17/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0121 - categorical_accuracy: 0.9306 - val_loss: 0.0162 - val_categorical_accuracy: 0.9176
Epoch 18/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0116 - categorical_accuracy: 0.9345 - val_loss: 0.0136 - val_categorical_accuracy: 0.9272
Epoch 19/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0114 - categorical_accuracy: 0.9351 - val_loss: 0.0162 - val_categorical_accuracy: 0.9152
Epoch 20/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0110 - categorical_accuracy: 0.9380 - val_loss: 0.0128 - val_categorical_accuracy: 0.9307
Epoch 21/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0107 - categorical_accuracy: 0.9395 - val_loss: 0.0129 - val_categorical_accuracy: 0.9348
Epoch 22/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0102 - categorical_accuracy: 0.9420 - val_loss: 0.0175 - val_categorical_accuracy: 0.9129
Epoch 23/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0102 - categorical_accuracy: 0.9426 - val_loss: 0.0116 - val_categorical_accuracy: 0.9392
Epoch 24/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0099 - categorical_accuracy: 0.9434 - val_loss: 0.0140 - val_categorical_accuracy: 0.9278
Epoch 25/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0096 - categorical_accuracy: 0.9458 - val_loss: 0.0132 - val_categorical_accuracy: 0.9337
Epoch 26/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0095 - categorical_accuracy: 0.9461 - val_loss: 0.0166 - val_categorical_accuracy: 0.9147
Epoch 27/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0092 - categorical_accuracy: 0.9479 - val_loss: 0.0145 - val_categorical_accuracy: 0.9272
Epoch 28/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0092 - categorical_accuracy: 0.9479 - val_loss: 0.0128 - val_categorical_accuracy: 0.9372
Epoch 29/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0090 - categorical_accuracy: 0.9485 - val_loss: 0.0141 - val_categorical_accuracy: 0.9322

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 30/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0071 - categorical_accuracy: 0.9603 - val_loss: 0.0101 - val_categorical_accuracy: 0.9490
Epoch 31/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0065 - categorical_accuracy: 0.9638 - val_loss: 0.0096 - val_categorical_accuracy: 0.9512
Epoch 32/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0063 - categorical_accuracy: 0.9643 - val_loss: 0.0098 - val_categorical_accuracy: 0.9503
Epoch 33/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0059 - categorical_accuracy: 0.9673 - val_loss: 0.0098 - val_categorical_accuracy: 0.9503
Epoch 34/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0061 - categorical_accuracy: 0.9662 - val_loss: 0.0096 - val_categorical_accuracy: 0.9527
Epoch 35/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0057 - categorical_accuracy: 0.9680 - val_loss: 0.0096 - val_categorical_accuracy: 0.9518
Epoch 36/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0058 - categorical_accuracy: 0.9676 - val_loss: 0.0097 - val_categorical_accuracy: 0.9519
Epoch 37/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0054 - categorical_accuracy: 0.9704 - val_loss: 0.0096 - val_categorical_accuracy: 0.9532

Epoch 00037: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 38/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0052 - categorical_accuracy: 0.9713 - val_loss: 0.0092 - val_categorical_accuracy: 0.9542
Epoch 39/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0052 - categorical_accuracy: 0.9713 - val_loss: 0.0094 - val_categorical_accuracy: 0.9536
Epoch 40/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0051 - categorical_accuracy: 0.9715 - val_loss: 0.0094 - val_categorical_accuracy: 0.9536
Epoch 41/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0051 - categorical_accuracy: 0.9714 - val_loss: 0.0096 - val_categorical_accuracy: 0.9524
Epoch 42/65535
420/420 [==============================] - 382s 908ms/step - loss: 0.0050 - categorical_accuracy: 0.9723 - val_loss: 0.0094 - val_categorical_accuracy: 0.9535
Epoch 43/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0051 - categorical_accuracy: 0.9725 - val_loss: 0.0093 - val_categorical_accuracy: 0.9538
Epoch 44/65535
420/420 [==============================] - 382s 908ms/step - loss: 0.0049 - categorical_accuracy: 0.9723 - val_loss: 0.0094 - val_categorical_accuracy: 0.9538

Epoch 00044: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 45/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0050 - categorical_accuracy: 0.9727 - val_loss: 0.0094 - val_categorical_accuracy: 0.9538
Epoch 46/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0050 - categorical_accuracy: 0.9720 - val_loss: 0.0094 - val_categorical_accuracy: 0.9538
Epoch 47/65535
420/420 [==============================] - 382s 908ms/step - loss: 0.0050 - categorical_accuracy: 0.9719 - val_loss: 0.0093 - val_categorical_accuracy: 0.9541
Epoch 48/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0049 - categorical_accuracy: 0.9728 - val_loss: 0.0094 - val_categorical_accuracy: 0.9539
Epoch 49/65535
420/420 [==============================] - 382s 909ms/step - loss: 0.0049 - categorical_accuracy: 0.9725 - val_loss: 0.0091 - val_categorical_accuracy: 0.9549

Epoch 00049: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 50/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0050 - categorical_accuracy: 0.9719 - val_loss: 0.0094 - val_categorical_accuracy: 0.9542
Epoch 51/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0049 - categorical_accuracy: 0.9724 - val_loss: 0.0092 - val_categorical_accuracy: 0.9546
Epoch 52/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0049 - categorical_accuracy: 0.9732 - val_loss: 0.0093 - val_categorical_accuracy: 0.9541
Epoch 53/65535
420/420 [==============================] - 381s 906ms/step - loss: 0.0049 - categorical_accuracy: 0.9729 - val_loss: 0.0095 - val_categorical_accuracy: 0.9535
Epoch 54/65535
420/420 [==============================] - 380s 905ms/step - loss: 0.0050 - categorical_accuracy: 0.9724 - val_loss: 0.0093 - val_categorical_accuracy: 0.9543

Epoch 00054: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 55/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0050 - categorical_accuracy: 0.9718 - val_loss: 0.0093 - val_categorical_accuracy: 0.9544
Epoch 56/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0049 - categorical_accuracy: 0.9732 - val_loss: 0.0093 - val_categorical_accuracy: 0.9543
Epoch 57/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0048 - categorical_accuracy: 0.9732 - val_loss: 0.0093 - val_categorical_accuracy: 0.9544
Epoch 58/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0050 - categorical_accuracy: 0.9725 - val_loss: 0.0095 - val_categorical_accuracy: 0.9536
Epoch 59/65535
420/420 [==============================] - 380s 906ms/step - loss: 0.0049 - categorical_accuracy: 0.9724 - val_loss: 0.0092 - val_categorical_accuracy: 0.9546

Epoch 00059: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 60/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0049 - categorical_accuracy: 0.9719 - val_loss: 0.0093 - val_categorical_accuracy: 0.9546
Epoch 61/65535
420/420 [==============================] - 381s 907ms/step - loss: 0.0050 - categorical_accuracy: 0.9721 - val_loss: 0.0092 - val_categorical_accuracy: 0.9545
Epoch 62/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0049 - categorical_accuracy: 0.9730 - val_loss: 0.0093 - val_categorical_accuracy: 0.9542
Epoch 63/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0051 - categorical_accuracy: 0.9718 - val_loss: 0.0094 - val_categorical_accuracy: 0.9543
Epoch 64/65535
420/420 [==============================] - 381s 908ms/step - loss: 0.0048 - categorical_accuracy: 0.9734 - val_loss: 0.0095 - val_categorical_accuracy: 0.9532
Epoch 00064: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 00064: early stopping
