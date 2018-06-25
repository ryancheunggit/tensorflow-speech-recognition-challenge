ren (master *+) python $ python train.py logmelspectrogram_40_25_10 join s 128
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
conv2d_1 (Conv2D)               (None, 101, 40, 32)  352         reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 101, 40, 32)  128         conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 101, 40, 32)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 101, 40, 32)  0           activation_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 101, 40, 32)  10272       dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 101, 40, 32)  128         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 101, 40, 32)  0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 101, 40, 32)  0           activation_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 50, 20, 32)   0           dropout_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 50, 20, 64)   20544       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 50, 20, 64)   256         conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 50, 20, 64)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 50, 20, 64)   0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 50, 20, 64)   41024       dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 50, 20, 64)   256         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 50, 20, 64)   0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 50, 20, 64)   0           activation_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 25, 10, 64)   0           dropout_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 25, 10, 128)  82048       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 25, 10, 128)  512         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 25, 10, 128)  0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 25, 10, 128)  0           activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 25, 10, 128)  163968      dropout_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 25, 10, 128)  512         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 25, 10, 128)  0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 25, 10, 128)  0           activation_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 12, 5, 128)   0           dropout_6[0][0]
__________________________________________________________________________________________________
cu_dnngru_1 (CuDNNGRU)          (None, 256)          228864      input_1[0][0]
__________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalM (None, 128)          0           max_pooling2d_3[0][0]
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 128)          0           max_pooling2d_3[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 256)          0           cu_dnngru_1[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 512)          0           global_max_pooling2d_1[0][0]
                                                                 global_average_pooling2d_1[0][0]
                                                                 dropout_7[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 512)          262656      concatenate_1[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 512)          2048        dense_1[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 512)          0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 512)          262656      dropout_8[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 512)          2048        dense_2[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 512)          0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 31)           15903       dropout_9[0][0]
==================================================================================================
Total params: 1,094,175
Trainable params: 1,091,231
Non-trainable params: 2,944
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 00:16:49 =========
Epoch 1/65535
420/420 [==============================] - 40s 94ms/step - loss: 0.1571 - categorical_accuracy: 0.0911 - val_loss: 0.2536 - val_categorical_accuracy: 0.0352
Epoch 2/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.1145 - categorical_accuracy: 0.2750 - val_loss: 0.1725 - val_categorical_accuracy: 0.0706
Epoch 3/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0778 - categorical_accuracy: 0.5120 - val_loss: 0.1182 - val_categorical_accuracy: 0.2523
Epoch 4/65535
420/420 [==============================] - 38s 92ms/step - loss: 0.0563 - categorical_accuracy: 0.6557 - val_loss: 0.0796 - val_categorical_accuracy: 0.5137
Epoch 5/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0451 - categorical_accuracy: 0.7305 - val_loss: 0.0592 - val_categorical_accuracy: 0.6558
Epoch 6/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0381 - categorical_accuracy: 0.7753 - val_loss: 0.0477 - val_categorical_accuracy: 0.7348
Epoch 7/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0337 - categorical_accuracy: 0.8036 - val_loss: 0.0424 - val_categorical_accuracy: 0.7710
Epoch 8/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0300 - categorical_accuracy: 0.8272 - val_loss: 0.0379 - val_categorical_accuracy: 0.8036
Epoch 9/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0272 - categorical_accuracy: 0.8439 - val_loss: 0.0352 - val_categorical_accuracy: 0.8121
Epoch 10/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0251 - categorical_accuracy: 0.8576 - val_loss: 0.0293 - val_categorical_accuracy: 0.8561
Epoch 11/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0234 - categorical_accuracy: 0.8675 - val_loss: 0.0276 - val_categorical_accuracy: 0.8599
Epoch 12/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0219 - categorical_accuracy: 0.8766 - val_loss: 0.0248 - val_categorical_accuracy: 0.8772
Epoch 13/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0207 - categorical_accuracy: 0.8841 - val_loss: 0.0224 - val_categorical_accuracy: 0.8889
Epoch 14/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0198 - categorical_accuracy: 0.8900 - val_loss: 0.0232 - val_categorical_accuracy: 0.8913
Epoch 15/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0188 - categorical_accuracy: 0.8947 - val_loss: 0.0226 - val_categorical_accuracy: 0.8883
Epoch 16/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0181 - categorical_accuracy: 0.8975 - val_loss: 0.0200 - val_categorical_accuracy: 0.9049
Epoch 17/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0172 - categorical_accuracy: 0.9040 - val_loss: 0.0182 - val_categorical_accuracy: 0.9149
Epoch 18/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0168 - categorical_accuracy: 0.9065 - val_loss: 0.0181 - val_categorical_accuracy: 0.9149
Epoch 19/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0162 - categorical_accuracy: 0.9094 - val_loss: 0.0176 - val_categorical_accuracy: 0.9187
Epoch 20/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0155 - categorical_accuracy: 0.9143 - val_loss: 0.0167 - val_categorical_accuracy: 0.9198
Epoch 21/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0149 - categorical_accuracy: 0.9189 - val_loss: 0.0165 - val_categorical_accuracy: 0.9232
Epoch 22/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0146 - categorical_accuracy: 0.9205 - val_loss: 0.0155 - val_categorical_accuracy: 0.9260
Epoch 23/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0143 - categorical_accuracy: 0.9217 - val_loss: 0.0147 - val_categorical_accuracy: 0.9312
Epoch 24/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0137 - categorical_accuracy: 0.9242 - val_loss: 0.0150 - val_categorical_accuracy: 0.9321
Epoch 25/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0134 - categorical_accuracy: 0.9261 - val_loss: 0.0156 - val_categorical_accuracy: 0.9301
Epoch 26/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0132 - categorical_accuracy: 0.9264 - val_loss: 0.0139 - val_categorical_accuracy: 0.9342
Epoch 27/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0131 - categorical_accuracy: 0.9268 - val_loss: 0.0147 - val_categorical_accuracy: 0.9330
Epoch 28/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0126 - categorical_accuracy: 0.9306 - val_loss: 0.0147 - val_categorical_accuracy: 0.9324
Epoch 29/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0124 - categorical_accuracy: 0.9304 - val_loss: 0.0145 - val_categorical_accuracy: 0.9345
Epoch 30/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0123 - categorical_accuracy: 0.9320 - val_loss: 0.0133 - val_categorical_accuracy: 0.9385
Epoch 31/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0120 - categorical_accuracy: 0.9347 - val_loss: 0.0143 - val_categorical_accuracy: 0.9350
Epoch 32/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0117 - categorical_accuracy: 0.9358 - val_loss: 0.0143 - val_categorical_accuracy: 0.9348
Epoch 33/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0113 - categorical_accuracy: 0.9382 - val_loss: 0.0141 - val_categorical_accuracy: 0.9358
Epoch 34/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0113 - categorical_accuracy: 0.9377 - val_loss: 0.0122 - val_categorical_accuracy: 0.9409
Epoch 35/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0110 - categorical_accuracy: 0.9391 - val_loss: 0.0123 - val_categorical_accuracy: 0.9443
Epoch 36/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0110 - categorical_accuracy: 0.9391 - val_loss: 0.0114 - val_categorical_accuracy: 0.9454
Epoch 37/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0107 - categorical_accuracy: 0.9419 - val_loss: 0.0111 - val_categorical_accuracy: 0.9472
Epoch 38/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0106 - categorical_accuracy: 0.9413 - val_loss: 0.0116 - val_categorical_accuracy: 0.9464
Epoch 39/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0103 - categorical_accuracy: 0.9442 - val_loss: 0.0110 - val_categorical_accuracy: 0.9462
Epoch 40/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0102 - categorical_accuracy: 0.9442 - val_loss: 0.0124 - val_categorical_accuracy: 0.9430
Epoch 41/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0099 - categorical_accuracy: 0.9456 - val_loss: 0.0119 - val_categorical_accuracy: 0.9431
Epoch 42/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0100 - categorical_accuracy: 0.9449 - val_loss: 0.0120 - val_categorical_accuracy: 0.9450
Epoch 43/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0097 - categorical_accuracy: 0.9467 - val_loss: 0.0105 - val_categorical_accuracy: 0.9506
Epoch 44/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0097 - categorical_accuracy: 0.9466 - val_loss: 0.0111 - val_categorical_accuracy: 0.9489
Epoch 45/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0096 - categorical_accuracy: 0.9476 - val_loss: 0.0106 - val_categorical_accuracy: 0.9510
Epoch 46/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0093 - categorical_accuracy: 0.9492 - val_loss: 0.0105 - val_categorical_accuracy: 0.9504
Epoch 47/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0094 - categorical_accuracy: 0.9485 - val_loss: 0.0109 - val_categorical_accuracy: 0.9492
Epoch 48/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0091 - categorical_accuracy: 0.9500 - val_loss: 0.0104 - val_categorical_accuracy: 0.9518
Epoch 49/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0089 - categorical_accuracy: 0.9508 - val_loss: 0.0106 - val_categorical_accuracy: 0.9501
Epoch 50/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0090 - categorical_accuracy: 0.9505 - val_loss: 0.0097 - val_categorical_accuracy: 0.9537
Epoch 51/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0089 - categorical_accuracy: 0.9521 - val_loss: 0.0103 - val_categorical_accuracy: 0.9513
Epoch 52/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0087 - categorical_accuracy: 0.9514 - val_loss: 0.0100 - val_categorical_accuracy: 0.9521
Epoch 53/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0085 - categorical_accuracy: 0.9535 - val_loss: 0.0097 - val_categorical_accuracy: 0.9531
Epoch 54/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0086 - categorical_accuracy: 0.9529 - val_loss: 0.0099 - val_categorical_accuracy: 0.9519
Epoch 55/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0083 - categorical_accuracy: 0.9538 - val_loss: 0.0099 - val_categorical_accuracy: 0.9525
Epoch 56/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0085 - categorical_accuracy: 0.9531 - val_loss: 0.0098 - val_categorical_accuracy: 0.9516

Epoch 00056: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 57/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0082 - categorical_accuracy: 0.9550 - val_loss: 0.0094 - val_categorical_accuracy: 0.9548
Epoch 58/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0081 - categorical_accuracy: 0.9553 - val_loss: 0.0095 - val_categorical_accuracy: 0.9542
Epoch 59/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0081 - categorical_accuracy: 0.9557 - val_loss: 0.0092 - val_categorical_accuracy: 0.9554
Epoch 60/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0080 - categorical_accuracy: 0.9558 - val_loss: 0.0094 - val_categorical_accuracy: 0.9546
Epoch 61/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0081 - categorical_accuracy: 0.9550 - val_loss: 0.0092 - val_categorical_accuracy: 0.9558
Epoch 62/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0080 - categorical_accuracy: 0.9552 - val_loss: 0.0095 - val_categorical_accuracy: 0.9536
Epoch 63/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0081 - categorical_accuracy: 0.9561 - val_loss: 0.0090 - val_categorical_accuracy: 0.9565
Epoch 64/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0079 - categorical_accuracy: 0.9558 - val_loss: 0.0091 - val_categorical_accuracy: 0.9561
Epoch 65/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0080 - categorical_accuracy: 0.9554 - val_loss: 0.0091 - val_categorical_accuracy: 0.9570
Epoch 66/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0079 - categorical_accuracy: 0.9565 - val_loss: 0.0091 - val_categorical_accuracy: 0.9565
Epoch 67/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0079 - categorical_accuracy: 0.9553 - val_loss: 0.0090 - val_categorical_accuracy: 0.9570
Epoch 68/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0079 - categorical_accuracy: 0.9560 - val_loss: 0.0092 - val_categorical_accuracy: 0.9560
Epoch 69/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9559 - val_loss: 0.0091 - val_categorical_accuracy: 0.9565

Epoch 00069: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 70/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0079 - categorical_accuracy: 0.9561 - val_loss: 0.0090 - val_categorical_accuracy: 0.9568
Epoch 71/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9568 - val_loss: 0.0090 - val_categorical_accuracy: 0.9568
Epoch 72/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0077 - categorical_accuracy: 0.9575 - val_loss: 0.0089 - val_categorical_accuracy: 0.9572
Epoch 73/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9565 - val_loss: 0.0089 - val_categorical_accuracy: 0.9573
Epoch 74/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9567 - val_loss: 0.0089 - val_categorical_accuracy: 0.9570

Epoch 00074: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 75/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9561 - val_loss: 0.0089 - val_categorical_accuracy: 0.9571
Epoch 76/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0077 - categorical_accuracy: 0.9572 - val_loss: 0.0089 - val_categorical_accuracy: 0.9570
Epoch 77/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0077 - categorical_accuracy: 0.9576 - val_loss: 0.0089 - val_categorical_accuracy: 0.9570
Epoch 78/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9570 - val_loss: 0.0089 - val_categorical_accuracy: 0.9572
Epoch 79/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9565 - val_loss: 0.0089 - val_categorical_accuracy: 0.9569

Epoch 00079: ReduceLROnPlateau reducing learning rate to 0.00015999998431652786.
Epoch 80/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0078 - categorical_accuracy: 0.9569 - val_loss: 0.0089 - val_categorical_accuracy: 0.9571
Epoch 81/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9569 - val_loss: 0.0089 - val_categorical_accuracy: 0.9571
Epoch 82/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0079 - categorical_accuracy: 0.9567 - val_loss: 0.0089 - val_categorical_accuracy: 0.9570
Epoch 83/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9564 - val_loss: 0.0089 - val_categorical_accuracy: 0.9568
Epoch 84/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9569 - val_loss: 0.0089 - val_categorical_accuracy: 0.9570

Epoch 00084: ReduceLROnPlateau reducing learning rate to 0.00010000000149011611.
Epoch 85/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9569 - val_loss: 0.0089 - val_categorical_accuracy: 0.9570
Epoch 86/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9563 - val_loss: 0.0089 - val_categorical_accuracy: 0.9569
Epoch 87/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9577 - val_loss: 0.0089 - val_categorical_accuracy: 0.9571
Epoch 88/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0077 - categorical_accuracy: 0.9577 - val_loss: 0.0089 - val_categorical_accuracy: 0.9569
Epoch 00088: early stopping
========= generating oof predictions 01:12:30 =========
========= generating test set predictions 01:12:33 =========
========= fitting 2 th model 01:13:06 =========
Epoch 1/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.1587 - categorical_accuracy: 0.0827 - val_loss: 0.1785 - val_categorical_accuracy: 0.0358
Epoch 2/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.1202 - categorical_accuracy: 0.2371 - val_loss: 0.1406 - val_categorical_accuracy: 0.1095
Epoch 3/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0816 - categorical_accuracy: 0.4806 - val_loss: 0.0844 - val_categorical_accuracy: 0.5034
Epoch 4/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0583 - categorical_accuracy: 0.6436 - val_loss: 0.0637 - val_categorical_accuracy: 0.6441
Epoch 5/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0461 - categorical_accuracy: 0.7238 - val_loss: 0.0494 - val_categorical_accuracy: 0.7394
Epoch 6/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0381 - categorical_accuracy: 0.7773 - val_loss: 0.0426 - val_categorical_accuracy: 0.7750
Epoch 7/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0333 - categorical_accuracy: 0.8077 - val_loss: 0.0361 - val_categorical_accuracy: 0.8132
Epoch 8/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0294 - categorical_accuracy: 0.8318 - val_loss: 0.0332 - val_categorical_accuracy: 0.8266
Epoch 9/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0269 - categorical_accuracy: 0.8482 - val_loss: 0.0298 - val_categorical_accuracy: 0.8477
Epoch 10/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0249 - categorical_accuracy: 0.8603 - val_loss: 0.0272 - val_categorical_accuracy: 0.8675
Epoch 11/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0231 - categorical_accuracy: 0.8704 - val_loss: 0.0270 - val_categorical_accuracy: 0.8609
Epoch 12/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0215 - categorical_accuracy: 0.8796 - val_loss: 0.0244 - val_categorical_accuracy: 0.8773
Epoch 13/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0207 - categorical_accuracy: 0.8853 - val_loss: 0.0224 - val_categorical_accuracy: 0.8888
Epoch 14/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0193 - categorical_accuracy: 0.8932 - val_loss: 0.0210 - val_categorical_accuracy: 0.8952
Epoch 15/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0183 - categorical_accuracy: 0.9006 - val_loss: 0.0203 - val_categorical_accuracy: 0.8968
Epoch 16/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0175 - categorical_accuracy: 0.9040 - val_loss: 0.0198 - val_categorical_accuracy: 0.9008
Epoch 17/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0168 - categorical_accuracy: 0.9059 - val_loss: 0.0188 - val_categorical_accuracy: 0.9057
Epoch 18/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0162 - categorical_accuracy: 0.9105 - val_loss: 0.0194 - val_categorical_accuracy: 0.9020
Epoch 19/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0157 - categorical_accuracy: 0.9125 - val_loss: 0.0172 - val_categorical_accuracy: 0.9125
Epoch 20/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0151 - categorical_accuracy: 0.9168 - val_loss: 0.0171 - val_categorical_accuracy: 0.9121
Epoch 21/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0147 - categorical_accuracy: 0.9195 - val_loss: 0.0159 - val_categorical_accuracy: 0.9191
Epoch 22/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0141 - categorical_accuracy: 0.9227 - val_loss: 0.0158 - val_categorical_accuracy: 0.9183
Epoch 23/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0140 - categorical_accuracy: 0.9224 - val_loss: 0.0155 - val_categorical_accuracy: 0.9233
Epoch 24/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0134 - categorical_accuracy: 0.9263 - val_loss: 0.0149 - val_categorical_accuracy: 0.9241
Epoch 25/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0132 - categorical_accuracy: 0.9271 - val_loss: 0.0141 - val_categorical_accuracy: 0.9293
Epoch 26/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0130 - categorical_accuracy: 0.9286 - val_loss: 0.0145 - val_categorical_accuracy: 0.9242
Epoch 27/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0126 - categorical_accuracy: 0.9307 - val_loss: 0.0139 - val_categorical_accuracy: 0.9284
Epoch 28/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0123 - categorical_accuracy: 0.9333 - val_loss: 0.0133 - val_categorical_accuracy: 0.9321
Epoch 29/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0120 - categorical_accuracy: 0.9335 - val_loss: 0.0127 - val_categorical_accuracy: 0.9355
Epoch 30/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0117 - categorical_accuracy: 0.9353 - val_loss: 0.0125 - val_categorical_accuracy: 0.9366
Epoch 31/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0115 - categorical_accuracy: 0.9365 - val_loss: 0.0122 - val_categorical_accuracy: 0.9369
Epoch 32/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0114 - categorical_accuracy: 0.9384 - val_loss: 0.0122 - val_categorical_accuracy: 0.9382
Epoch 33/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0111 - categorical_accuracy: 0.9393 - val_loss: 0.0112 - val_categorical_accuracy: 0.9422
Epoch 34/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0109 - categorical_accuracy: 0.9388 - val_loss: 0.0114 - val_categorical_accuracy: 0.9404
Epoch 35/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0106 - categorical_accuracy: 0.9428 - val_loss: 0.0109 - val_categorical_accuracy: 0.9434
Epoch 36/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0103 - categorical_accuracy: 0.9432 - val_loss: 0.0114 - val_categorical_accuracy: 0.9416
Epoch 37/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0104 - categorical_accuracy: 0.9437 - val_loss: 0.0110 - val_categorical_accuracy: 0.9433
Epoch 38/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0101 - categorical_accuracy: 0.9452 - val_loss: 0.0107 - val_categorical_accuracy: 0.9442
Epoch 39/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0100 - categorical_accuracy: 0.9460 - val_loss: 0.0109 - val_categorical_accuracy: 0.9441
Epoch 40/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0099 - categorical_accuracy: 0.9450 - val_loss: 0.0108 - val_categorical_accuracy: 0.9448
Epoch 41/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0097 - categorical_accuracy: 0.9474 - val_loss: 0.0106 - val_categorical_accuracy: 0.9446
Epoch 42/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0097 - categorical_accuracy: 0.9468 - val_loss: 0.0107 - val_categorical_accuracy: 0.9450
Epoch 43/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0096 - categorical_accuracy: 0.9474 - val_loss: 0.0100 - val_categorical_accuracy: 0.9425
Epoch 44/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0095 - categorical_accuracy: 0.9484 - val_loss: 0.0100 - val_categorical_accuracy: 0.9497
Epoch 45/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0093 - categorical_accuracy: 0.9492 - val_loss: 0.0104 - val_categorical_accuracy: 0.9456
Epoch 46/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0091 - categorical_accuracy: 0.9498 - val_loss: 0.0097 - val_categorical_accuracy: 0.9494
Epoch 47/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0090 - categorical_accuracy: 0.9512 - val_loss: 0.0097 - val_categorical_accuracy: 0.9487
Epoch 48/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0090 - categorical_accuracy: 0.9499 - val_loss: 0.0093 - val_categorical_accuracy: 0.9524
Epoch 49/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0088 - categorical_accuracy: 0.9511 - val_loss: 0.0096 - val_categorical_accuracy: 0.9502
Epoch 50/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0087 - categorical_accuracy: 0.9525 - val_loss: 0.0099 - val_categorical_accuracy: 0.9511
Epoch 51/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0086 - categorical_accuracy: 0.9520 - val_loss: 0.0097 - val_categorical_accuracy: 0.9509
Epoch 52/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0086 - categorical_accuracy: 0.9522 - val_loss: 0.0097 - val_categorical_accuracy: 0.9503
Epoch 53/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0084 - categorical_accuracy: 0.9543 - val_loss: 0.0092 - val_categorical_accuracy: 0.9517
Epoch 54/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0083 - categorical_accuracy: 0.9534 - val_loss: 0.0091 - val_categorical_accuracy: 0.9527
Epoch 55/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0082 - categorical_accuracy: 0.9545 - val_loss: 0.0088 - val_categorical_accuracy: 0.9538
Epoch 56/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0082 - categorical_accuracy: 0.9547 - val_loss: 0.0091 - val_categorical_accuracy: 0.9537
Epoch 57/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0081 - categorical_accuracy: 0.9549 - val_loss: 0.0093 - val_categorical_accuracy: 0.9506
Epoch 58/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0080 - categorical_accuracy: 0.9561 - val_loss: 0.0087 - val_categorical_accuracy: 0.9540
Epoch 59/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0079 - categorical_accuracy: 0.9565 - val_loss: 0.0086 - val_categorical_accuracy: 0.9550
Epoch 60/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9567 - val_loss: 0.0088 - val_categorical_accuracy: 0.9564
Epoch 61/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0077 - categorical_accuracy: 0.9567 - val_loss: 0.0087 - val_categorical_accuracy: 0.9560
Epoch 62/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0078 - categorical_accuracy: 0.9565 - val_loss: 0.0086 - val_categorical_accuracy: 0.9553
Epoch 63/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0076 - categorical_accuracy: 0.9580 - val_loss: 0.0085 - val_categorical_accuracy: 0.9550
Epoch 64/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0074 - categorical_accuracy: 0.9601 - val_loss: 0.0084 - val_categorical_accuracy: 0.9560
Epoch 65/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0075 - categorical_accuracy: 0.9591 - val_loss: 0.0084 - val_categorical_accuracy: 0.9560
Epoch 66/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0075 - categorical_accuracy: 0.9587 - val_loss: 0.0085 - val_categorical_accuracy: 0.9554
Epoch 67/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0073 - categorical_accuracy: 0.9596 - val_loss: 0.0084 - val_categorical_accuracy: 0.9552
Epoch 68/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0073 - categorical_accuracy: 0.9604 - val_loss: 0.0086 - val_categorical_accuracy: 0.9555
Epoch 69/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0072 - categorical_accuracy: 0.9602 - val_loss: 0.0083 - val_categorical_accuracy: 0.9568
Epoch 70/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0071 - categorical_accuracy: 0.9610 - val_loss: 0.0082 - val_categorical_accuracy: 0.9592
Epoch 71/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0071 - categorical_accuracy: 0.9606 - val_loss: 0.0080 - val_categorical_accuracy: 0.9583
Epoch 72/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0069 - categorical_accuracy: 0.9621 - val_loss: 0.0081 - val_categorical_accuracy: 0.9576
Epoch 73/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0071 - categorical_accuracy: 0.9605 - val_loss: 0.0077 - val_categorical_accuracy: 0.9596
Epoch 74/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0070 - categorical_accuracy: 0.9612 - val_loss: 0.0081 - val_categorical_accuracy: 0.9591
Epoch 75/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0068 - categorical_accuracy: 0.9628 - val_loss: 0.0080 - val_categorical_accuracy: 0.9578
Epoch 76/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0068 - categorical_accuracy: 0.9625 - val_loss: 0.0082 - val_categorical_accuracy: 0.9573
Epoch 77/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0068 - categorical_accuracy: 0.9623 - val_loss: 0.0081 - val_categorical_accuracy: 0.9564
Epoch 78/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0068 - categorical_accuracy: 0.9626 - val_loss: 0.0079 - val_categorical_accuracy: 0.9588
Epoch 79/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0068 - categorical_accuracy: 0.9626 - val_loss: 0.0078 - val_categorical_accuracy: 0.9583

Epoch 00079: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 80/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0065 - categorical_accuracy: 0.9638 - val_loss: 0.0076 - val_categorical_accuracy: 0.9601
Epoch 81/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0065 - categorical_accuracy: 0.9641 - val_loss: 0.0076 - val_categorical_accuracy: 0.9593
Epoch 82/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0066 - categorical_accuracy: 0.9630 - val_loss: 0.0076 - val_categorical_accuracy: 0.9596
Epoch 83/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0065 - categorical_accuracy: 0.9642 - val_loss: 0.0074 - val_categorical_accuracy: 0.9604
Epoch 84/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0064 - categorical_accuracy: 0.9647 - val_loss: 0.0076 - val_categorical_accuracy: 0.9602
Epoch 85/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0065 - categorical_accuracy: 0.9644 - val_loss: 0.0075 - val_categorical_accuracy: 0.9606
Epoch 86/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0065 - categorical_accuracy: 0.9642 - val_loss: 0.0075 - val_categorical_accuracy: 0.9594
Epoch 87/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0063 - categorical_accuracy: 0.9647 - val_loss: 0.0075 - val_categorical_accuracy: 0.9599
Epoch 88/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0065 - categorical_accuracy: 0.9643 - val_loss: 0.0074 - val_categorical_accuracy: 0.9603
Epoch 89/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0064 - categorical_accuracy: 0.9649 - val_loss: 0.0074 - val_categorical_accuracy: 0.9605

Epoch 00089: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 90/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0063 - categorical_accuracy: 0.9645 - val_loss: 0.0074 - val_categorical_accuracy: 0.9596
Epoch 91/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0063 - categorical_accuracy: 0.9645 - val_loss: 0.0075 - val_categorical_accuracy: 0.9596
Epoch 92/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0063 - categorical_accuracy: 0.9658 - val_loss: 0.0075 - val_categorical_accuracy: 0.9596
Epoch 93/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0062 - categorical_accuracy: 0.9651 - val_loss: 0.0074 - val_categorical_accuracy: 0.9598
Epoch 94/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0062 - categorical_accuracy: 0.9652 - val_loss: 0.0075 - val_categorical_accuracy: 0.9597

Epoch 00094: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 95/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0063 - categorical_accuracy: 0.9653 - val_loss: 0.0074 - val_categorical_accuracy: 0.9597
Epoch 96/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0062 - categorical_accuracy: 0.9656 - val_loss: 0.0075 - val_categorical_accuracy: 0.9594
Epoch 97/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0062 - categorical_accuracy: 0.9651 - val_loss: 0.0074 - val_categorical_accuracy: 0.9595
Epoch 98/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0063 - categorical_accuracy: 0.9653 - val_loss: 0.0074 - val_categorical_accuracy: 0.9598
Epoch 99/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0063 - categorical_accuracy: 0.9648 - val_loss: 0.0074 - val_categorical_accuracy: 0.9596

Epoch 00099: ReduceLROnPlateau reducing learning rate to 0.00015999998431652786.
Epoch 100/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0063 - categorical_accuracy: 0.9656 - val_loss: 0.0075 - val_categorical_accuracy: 0.9594
Epoch 101/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0062 - categorical_accuracy: 0.9655 - val_loss: 0.0074 - val_categorical_accuracy: 0.9596
Epoch 102/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0063 - categorical_accuracy: 0.9646 - val_loss: 0.0074 - val_categorical_accuracy: 0.9595
Epoch 103/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0064 - categorical_accuracy: 0.9646 - val_loss: 0.0075 - val_categorical_accuracy: 0.9596
Epoch 104/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0063 - categorical_accuracy: 0.9650 - val_loss: 0.0074 - val_categorical_accuracy: 0.9595

Epoch 00104: ReduceLROnPlateau reducing learning rate to 0.00010000000149011611.
Epoch 00104: early stopping
========= generating oof predictions 02:18:30 =========
========= generating test set predictions 02:18:32 =========
========= fitting 3 th model 02:19:06 =========
Epoch 1/65535
420/420 [==============================] - 38s 92ms/step - loss: 0.1573 - categorical_accuracy: 0.0908 - val_loss: 0.2581 - val_categorical_accuracy: 0.0394
Epoch 2/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.1158 - categorical_accuracy: 0.2627 - val_loss: 0.1937 - val_categorical_accuracy: 0.0542
Epoch 3/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0790 - categorical_accuracy: 0.4953 - val_loss: 0.1167 - val_categorical_accuracy: 0.2681
Epoch 4/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0576 - categorical_accuracy: 0.6442 - val_loss: 0.0859 - val_categorical_accuracy: 0.4774
Epoch 5/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0462 - categorical_accuracy: 0.7226 - val_loss: 0.0599 - val_categorical_accuracy: 0.6641
Epoch 6/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0387 - categorical_accuracy: 0.7738 - val_loss: 0.0493 - val_categorical_accuracy: 0.7315
Epoch 7/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0334 - categorical_accuracy: 0.8053 - val_loss: 0.0413 - val_categorical_accuracy: 0.7881
Epoch 8/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0298 - categorical_accuracy: 0.8277 - val_loss: 0.0361 - val_categorical_accuracy: 0.8206
Epoch 9/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0268 - categorical_accuracy: 0.8481 - val_loss: 0.0345 - val_categorical_accuracy: 0.8195
Epoch 10/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0246 - categorical_accuracy: 0.8608 - val_loss: 0.0306 - val_categorical_accuracy: 0.8473
Epoch 11/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0227 - categorical_accuracy: 0.8711 - val_loss: 0.0313 - val_categorical_accuracy: 0.8399
Epoch 12/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0212 - categorical_accuracy: 0.8810 - val_loss: 0.0304 - val_categorical_accuracy: 0.8496
Epoch 13/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0199 - categorical_accuracy: 0.8885 - val_loss: 0.0268 - val_categorical_accuracy: 0.8693
Epoch 14/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0191 - categorical_accuracy: 0.8942 - val_loss: 0.0246 - val_categorical_accuracy: 0.8844
Epoch 15/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0180 - categorical_accuracy: 0.8997 - val_loss: 0.0233 - val_categorical_accuracy: 0.8928
Epoch 16/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0172 - categorical_accuracy: 0.9034 - val_loss: 0.0221 - val_categorical_accuracy: 0.8958
Epoch 17/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0166 - categorical_accuracy: 0.9071 - val_loss: 0.0239 - val_categorical_accuracy: 0.8841
Epoch 18/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0158 - categorical_accuracy: 0.9135 - val_loss: 0.0213 - val_categorical_accuracy: 0.9031
Epoch 19/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0152 - categorical_accuracy: 0.9155 - val_loss: 0.0211 - val_categorical_accuracy: 0.9007
Epoch 20/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0147 - categorical_accuracy: 0.9182 - val_loss: 0.0204 - val_categorical_accuracy: 0.9028
Epoch 21/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0142 - categorical_accuracy: 0.9210 - val_loss: 0.0199 - val_categorical_accuracy: 0.9034
Epoch 22/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0139 - categorical_accuracy: 0.9232 - val_loss: 0.0197 - val_categorical_accuracy: 0.9071
Epoch 23/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0135 - categorical_accuracy: 0.9244 - val_loss: 0.0174 - val_categorical_accuracy: 0.9181
Epoch 24/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0131 - categorical_accuracy: 0.9284 - val_loss: 0.0185 - val_categorical_accuracy: 0.9120
Epoch 25/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0128 - categorical_accuracy: 0.9294 - val_loss: 0.0162 - val_categorical_accuracy: 0.9213
Epoch 26/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0125 - categorical_accuracy: 0.9306 - val_loss: 0.0178 - val_categorical_accuracy: 0.9130
Epoch 27/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0123 - categorical_accuracy: 0.9324 - val_loss: 0.0172 - val_categorical_accuracy: 0.9168
Epoch 28/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0119 - categorical_accuracy: 0.9347 - val_loss: 0.0173 - val_categorical_accuracy: 0.9151
Epoch 29/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0115 - categorical_accuracy: 0.9364 - val_loss: 0.0157 - val_categorical_accuracy: 0.9251
Epoch 30/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0115 - categorical_accuracy: 0.9375 - val_loss: 0.0155 - val_categorical_accuracy: 0.9257
Epoch 31/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0113 - categorical_accuracy: 0.9368 - val_loss: 0.0163 - val_categorical_accuracy: 0.9200
Epoch 32/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0110 - categorical_accuracy: 0.9391 - val_loss: 0.0151 - val_categorical_accuracy: 0.9291
Epoch 33/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0107 - categorical_accuracy: 0.9422 - val_loss: 0.0149 - val_categorical_accuracy: 0.9240
Epoch 34/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0107 - categorical_accuracy: 0.9413 - val_loss: 0.0137 - val_categorical_accuracy: 0.9326
Epoch 35/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0105 - categorical_accuracy: 0.9429 - val_loss: 0.0142 - val_categorical_accuracy: 0.9309
Epoch 36/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0103 - categorical_accuracy: 0.9426 - val_loss: 0.0155 - val_categorical_accuracy: 0.9223
Epoch 37/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0101 - categorical_accuracy: 0.9440 - val_loss: 0.0149 - val_categorical_accuracy: 0.9253
Epoch 38/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0100 - categorical_accuracy: 0.9448 - val_loss: 0.0136 - val_categorical_accuracy: 0.9315
Epoch 39/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0099 - categorical_accuracy: 0.9463 - val_loss: 0.0133 - val_categorical_accuracy: 0.9302
Epoch 40/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0097 - categorical_accuracy: 0.9466 - val_loss: 0.0142 - val_categorical_accuracy: 0.9299
Epoch 41/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0096 - categorical_accuracy: 0.9472 - val_loss: 0.0139 - val_categorical_accuracy: 0.9292
Epoch 42/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0094 - categorical_accuracy: 0.9479 - val_loss: 0.0128 - val_categorical_accuracy: 0.9334
Epoch 43/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0094 - categorical_accuracy: 0.9489 - val_loss: 0.0132 - val_categorical_accuracy: 0.9350
Epoch 44/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0092 - categorical_accuracy: 0.9489 - val_loss: 0.0131 - val_categorical_accuracy: 0.9323
Epoch 45/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0092 - categorical_accuracy: 0.9492 - val_loss: 0.0124 - val_categorical_accuracy: 0.9368
Epoch 46/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0090 - categorical_accuracy: 0.9513 - val_loss: 0.0136 - val_categorical_accuracy: 0.9321
Epoch 47/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0088 - categorical_accuracy: 0.9515 - val_loss: 0.0122 - val_categorical_accuracy: 0.9375
Epoch 48/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0088 - categorical_accuracy: 0.9516 - val_loss: 0.0134 - val_categorical_accuracy: 0.9312
Epoch 49/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0086 - categorical_accuracy: 0.9519 - val_loss: 0.0120 - val_categorical_accuracy: 0.9376
Epoch 50/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0086 - categorical_accuracy: 0.9526 - val_loss: 0.0128 - val_categorical_accuracy: 0.9361
Epoch 51/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0086 - categorical_accuracy: 0.9530 - val_loss: 0.0129 - val_categorical_accuracy: 0.9352
Epoch 52/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0085 - categorical_accuracy: 0.9534 - val_loss: 0.0119 - val_categorical_accuracy: 0.9384
Epoch 53/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0084 - categorical_accuracy: 0.9533 - val_loss: 0.0133 - val_categorical_accuracy: 0.9323
Epoch 54/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0082 - categorical_accuracy: 0.9541 - val_loss: 0.0124 - val_categorical_accuracy: 0.9349
Epoch 55/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0081 - categorical_accuracy: 0.9553 - val_loss: 0.0134 - val_categorical_accuracy: 0.9316
Epoch 56/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0080 - categorical_accuracy: 0.9556 - val_loss: 0.0136 - val_categorical_accuracy: 0.9293
Epoch 57/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0081 - categorical_accuracy: 0.9545 - val_loss: 0.0116 - val_categorical_accuracy: 0.9399
Epoch 58/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0078 - categorical_accuracy: 0.9577 - val_loss: 0.0125 - val_categorical_accuracy: 0.9355
Epoch 59/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0079 - categorical_accuracy: 0.9571 - val_loss: 0.0149 - val_categorical_accuracy: 0.9237
Epoch 60/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0077 - categorical_accuracy: 0.9578 - val_loss: 0.0110 - val_categorical_accuracy: 0.9424
Epoch 61/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0077 - categorical_accuracy: 0.9568 - val_loss: 0.0122 - val_categorical_accuracy: 0.9376
Epoch 62/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0076 - categorical_accuracy: 0.9579 - val_loss: 0.0111 - val_categorical_accuracy: 0.9434
Epoch 63/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0076 - categorical_accuracy: 0.9577 - val_loss: 0.0115 - val_categorical_accuracy: 0.9405
Epoch 64/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0074 - categorical_accuracy: 0.9586 - val_loss: 0.0118 - val_categorical_accuracy: 0.9375
Epoch 65/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0075 - categorical_accuracy: 0.9585 - val_loss: 0.0110 - val_categorical_accuracy: 0.9431
Epoch 66/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0073 - categorical_accuracy: 0.9596 - val_loss: 0.0116 - val_categorical_accuracy: 0.9405

Epoch 00066: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 67/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0072 - categorical_accuracy: 0.9600 - val_loss: 0.0111 - val_categorical_accuracy: 0.9417
Epoch 68/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0071 - categorical_accuracy: 0.9607 - val_loss: 0.0110 - val_categorical_accuracy: 0.9414
Epoch 69/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0071 - categorical_accuracy: 0.9605 - val_loss: 0.0111 - val_categorical_accuracy: 0.9420
Epoch 70/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0070 - categorical_accuracy: 0.9618 - val_loss: 0.0107 - val_categorical_accuracy: 0.9437
Epoch 71/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0071 - categorical_accuracy: 0.9611 - val_loss: 0.0115 - val_categorical_accuracy: 0.9407
Epoch 72/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0071 - categorical_accuracy: 0.9611 - val_loss: 0.0113 - val_categorical_accuracy: 0.9405
Epoch 73/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0070 - categorical_accuracy: 0.9617 - val_loss: 0.0111 - val_categorical_accuracy: 0.9414
Epoch 74/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0071 - categorical_accuracy: 0.9600 - val_loss: 0.0109 - val_categorical_accuracy: 0.9424
Epoch 75/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0070 - categorical_accuracy: 0.9612 - val_loss: 0.0108 - val_categorical_accuracy: 0.9433
Epoch 76/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0069 - categorical_accuracy: 0.9612 - val_loss: 0.0107 - val_categorical_accuracy: 0.9432

Epoch 00076: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 77/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0070 - categorical_accuracy: 0.9610 - val_loss: 0.0110 - val_categorical_accuracy: 0.9420
Epoch 78/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0068 - categorical_accuracy: 0.9615 - val_loss: 0.0109 - val_categorical_accuracy: 0.9426
Epoch 79/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0069 - categorical_accuracy: 0.9612 - val_loss: 0.0110 - val_categorical_accuracy: 0.9423
Epoch 80/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0071 - categorical_accuracy: 0.9602 - val_loss: 0.0109 - val_categorical_accuracy: 0.9426
Epoch 81/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0071 - categorical_accuracy: 0.9606 - val_loss: 0.0111 - val_categorical_accuracy: 0.9416

Epoch 00081: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 82/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0069 - categorical_accuracy: 0.9626 - val_loss: 0.0109 - val_categorical_accuracy: 0.9423
Epoch 83/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0069 - categorical_accuracy: 0.9615 - val_loss: 0.0109 - val_categorical_accuracy: 0.9422
Epoch 84/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0069 - categorical_accuracy: 0.9611 - val_loss: 0.0109 - val_categorical_accuracy: 0.9419
Epoch 85/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0069 - categorical_accuracy: 0.9619 - val_loss: 0.0109 - val_categorical_accuracy: 0.9423
Epoch 00085: early stopping
========= generating oof predictions 03:13:04 =========
========= generating test set predictions 03:13:06 =========
========= fitting 4 th model 03:13:40 =========
Epoch 1/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.1513 - categorical_accuracy: 0.1072 - val_loss: 0.1730 - val_categorical_accuracy: 0.0805
Epoch 2/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.1078 - categorical_accuracy: 0.3086 - val_loss: 0.1222 - val_categorical_accuracy: 0.2408
Epoch 3/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0745 - categorical_accuracy: 0.5322 - val_loss: 0.0878 - val_categorical_accuracy: 0.4699
Epoch 4/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0560 - categorical_accuracy: 0.6586 - val_loss: 0.0677 - val_categorical_accuracy: 0.6194
Epoch 5/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0459 - categorical_accuracy: 0.7256 - val_loss: 0.0549 - val_categorical_accuracy: 0.7078
Epoch 6/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0391 - categorical_accuracy: 0.7674 - val_loss: 0.0508 - val_categorical_accuracy: 0.7419
Epoch 7/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0345 - categorical_accuracy: 0.7989 - val_loss: 0.0404 - val_categorical_accuracy: 0.8065
Epoch 8/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0309 - categorical_accuracy: 0.8222 - val_loss: 0.0343 - val_categorical_accuracy: 0.8413
Epoch 9/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0281 - categorical_accuracy: 0.8383 - val_loss: 0.0325 - val_categorical_accuracy: 0.8415
Epoch 10/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0263 - categorical_accuracy: 0.8505 - val_loss: 0.0281 - val_categorical_accuracy: 0.8669
Epoch 11/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0243 - categorical_accuracy: 0.8629 - val_loss: 0.0269 - val_categorical_accuracy: 0.8740
Epoch 12/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0229 - categorical_accuracy: 0.8701 - val_loss: 0.0251 - val_categorical_accuracy: 0.8832
Epoch 13/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0217 - categorical_accuracy: 0.8780 - val_loss: 0.0234 - val_categorical_accuracy: 0.8906
Epoch 14/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0204 - categorical_accuracy: 0.8861 - val_loss: 0.0226 - val_categorical_accuracy: 0.8976
Epoch 15/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0198 - categorical_accuracy: 0.8898 - val_loss: 0.0214 - val_categorical_accuracy: 0.8984
Epoch 16/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0188 - categorical_accuracy: 0.8945 - val_loss: 0.0204 - val_categorical_accuracy: 0.9054
Epoch 17/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0182 - categorical_accuracy: 0.9005 - val_loss: 0.0187 - val_categorical_accuracy: 0.9113
Epoch 18/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0174 - categorical_accuracy: 0.9037 - val_loss: 0.0199 - val_categorical_accuracy: 0.9065
Epoch 19/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0168 - categorical_accuracy: 0.9076 - val_loss: 0.0178 - val_categorical_accuracy: 0.9156
Epoch 20/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0162 - categorical_accuracy: 0.9113 - val_loss: 0.0181 - val_categorical_accuracy: 0.9124
Epoch 21/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0157 - categorical_accuracy: 0.9123 - val_loss: 0.0177 - val_categorical_accuracy: 0.9183
Epoch 22/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0153 - categorical_accuracy: 0.9143 - val_loss: 0.0170 - val_categorical_accuracy: 0.9183
Epoch 23/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0148 - categorical_accuracy: 0.9175 - val_loss: 0.0164 - val_categorical_accuracy: 0.9221
Epoch 24/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0143 - categorical_accuracy: 0.9220 - val_loss: 0.0183 - val_categorical_accuracy: 0.9150
Epoch 25/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0140 - categorical_accuracy: 0.9219 - val_loss: 0.0157 - val_categorical_accuracy: 0.9227
Epoch 26/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0138 - categorical_accuracy: 0.9244 - val_loss: 0.0149 - val_categorical_accuracy: 0.9298
Epoch 27/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0134 - categorical_accuracy: 0.9254 - val_loss: 0.0156 - val_categorical_accuracy: 0.9276
Epoch 28/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0131 - categorical_accuracy: 0.9275 - val_loss: 0.0144 - val_categorical_accuracy: 0.9319
Epoch 29/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0128 - categorical_accuracy: 0.9306 - val_loss: 0.0145 - val_categorical_accuracy: 0.9330
Epoch 30/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0124 - categorical_accuracy: 0.9312 - val_loss: 0.0156 - val_categorical_accuracy: 0.9292
Epoch 31/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0122 - categorical_accuracy: 0.9325 - val_loss: 0.0161 - val_categorical_accuracy: 0.9214
Epoch 32/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0120 - categorical_accuracy: 0.9347 - val_loss: 0.0154 - val_categorical_accuracy: 0.9296
Epoch 33/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0119 - categorical_accuracy: 0.9353 - val_loss: 0.0153 - val_categorical_accuracy: 0.9258
Epoch 34/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0117 - categorical_accuracy: 0.9352 - val_loss: 0.0129 - val_categorical_accuracy: 0.9393
Epoch 35/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0113 - categorical_accuracy: 0.9381 - val_loss: 0.0117 - val_categorical_accuracy: 0.9434
Epoch 36/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0113 - categorical_accuracy: 0.9391 - val_loss: 0.0133 - val_categorical_accuracy: 0.9355
Epoch 37/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0110 - categorical_accuracy: 0.9388 - val_loss: 0.0122 - val_categorical_accuracy: 0.9410
Epoch 38/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0108 - categorical_accuracy: 0.9397 - val_loss: 0.0131 - val_categorical_accuracy: 0.9384
Epoch 39/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0107 - categorical_accuracy: 0.9405 - val_loss: 0.0129 - val_categorical_accuracy: 0.9374
Epoch 40/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0104 - categorical_accuracy: 0.9420 - val_loss: 0.0125 - val_categorical_accuracy: 0.9404
Epoch 41/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0103 - categorical_accuracy: 0.9437 - val_loss: 0.0133 - val_categorical_accuracy: 0.9363

Epoch 00041: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 42/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0102 - categorical_accuracy: 0.9443 - val_loss: 0.0126 - val_categorical_accuracy: 0.9385
Epoch 43/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0099 - categorical_accuracy: 0.9454 - val_loss: 0.0115 - val_categorical_accuracy: 0.9421
Epoch 44/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0099 - categorical_accuracy: 0.9464 - val_loss: 0.0113 - val_categorical_accuracy: 0.9437
Epoch 45/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0097 - categorical_accuracy: 0.9462 - val_loss: 0.0115 - val_categorical_accuracy: 0.9431
Epoch 46/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0098 - categorical_accuracy: 0.9463 - val_loss: 0.0120 - val_categorical_accuracy: 0.9412
Epoch 47/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0098 - categorical_accuracy: 0.9465 - val_loss: 0.0114 - val_categorical_accuracy: 0.9434
Epoch 48/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0098 - categorical_accuracy: 0.9453 - val_loss: 0.0114 - val_categorical_accuracy: 0.9430
Epoch 49/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0097 - categorical_accuracy: 0.9472 - val_loss: 0.0120 - val_categorical_accuracy: 0.9418
Epoch 50/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0096 - categorical_accuracy: 0.9478 - val_loss: 0.0115 - val_categorical_accuracy: 0.9433

Epoch 00050: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 51/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0096 - categorical_accuracy: 0.9470 - val_loss: 0.0114 - val_categorical_accuracy: 0.9434
Epoch 52/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0095 - categorical_accuracy: 0.9476 - val_loss: 0.0114 - val_categorical_accuracy: 0.9436
Epoch 53/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0097 - categorical_accuracy: 0.9461 - val_loss: 0.0113 - val_categorical_accuracy: 0.9442
Epoch 54/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0095 - categorical_accuracy: 0.9479 - val_loss: 0.0113 - val_categorical_accuracy: 0.9442
Epoch 55/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0095 - categorical_accuracy: 0.9471 - val_loss: 0.0113 - val_categorical_accuracy: 0.9437

Epoch 00055: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 56/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0096 - categorical_accuracy: 0.9477 - val_loss: 0.0114 - val_categorical_accuracy: 0.9437
Epoch 57/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0096 - categorical_accuracy: 0.9471 - val_loss: 0.0114 - val_categorical_accuracy: 0.9438
Epoch 58/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0094 - categorical_accuracy: 0.9484 - val_loss: 0.0114 - val_categorical_accuracy: 0.9438
Epoch 59/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0096 - categorical_accuracy: 0.9470 - val_loss: 0.0114 - val_categorical_accuracy: 0.9440
Epoch 60/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0095 - categorical_accuracy: 0.9481 - val_loss: 0.0114 - val_categorical_accuracy: 0.9439

Epoch 00060: ReduceLROnPlateau reducing learning rate to 0.00015999998431652786.
Epoch 61/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0095 - categorical_accuracy: 0.9479 - val_loss: 0.0114 - val_categorical_accuracy: 0.9440
Epoch 62/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0095 - categorical_accuracy: 0.9473 - val_loss: 0.0114 - val_categorical_accuracy: 0.9440
Epoch 63/65535
420/420 [==============================] - 37s 89ms/step - loss: 0.0095 - categorical_accuracy: 0.9474 - val_loss: 0.0114 - val_categorical_accuracy: 0.9438
Epoch 64/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0095 - categorical_accuracy: 0.9468 - val_loss: 0.0114 - val_categorical_accuracy: 0.9440
Epoch 65/65535
420/420 [==============================] - 38s 89ms/step - loss: 0.0095 - categorical_accuracy: 0.9475 - val_loss: 0.0114 - val_categorical_accuracy: 0.9436

Epoch 00065: ReduceLROnPlateau reducing learning rate to 0.00010000000149011611.
Epoch 66/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0096 - categorical_accuracy: 0.9469 - val_loss: 0.0114 - val_categorical_accuracy: 0.9439
Epoch 67/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0096 - categorical_accuracy: 0.9473 - val_loss: 0.0113 - val_categorical_accuracy: 0.9439
Epoch 68/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0095 - categorical_accuracy: 0.9469 - val_loss: 0.0114 - val_categorical_accuracy: 0.9440
Epoch 69/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0095 - categorical_accuracy: 0.9475 - val_loss: 0.0114 - val_categorical_accuracy: 0.9440
Epoch 00069: early stopping
========= generating oof predictions 03:56:53 =========
========= generating test set predictions 03:56:55 =========
========= fitting 5 th model 03:57:29 =========
Epoch 1/65535
420/420 [==============================] - 38s 92ms/step - loss: 0.1580 - categorical_accuracy: 0.0835 - val_loss: 0.2247 - val_categorical_accuracy: 0.0415
Epoch 2/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.1210 - categorical_accuracy: 0.2321 - val_loss: 0.2026 - val_categorical_accuracy: 0.0666
Epoch 3/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0849 - categorical_accuracy: 0.4595 - val_loss: 0.1516 - val_categorical_accuracy: 0.1516
Epoch 4/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0621 - categorical_accuracy: 0.6168 - val_loss: 0.1136 - val_categorical_accuracy: 0.3215
Epoch 5/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0493 - categorical_accuracy: 0.7033 - val_loss: 0.0860 - val_categorical_accuracy: 0.4863
Epoch 6/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0413 - categorical_accuracy: 0.7543 - val_loss: 0.0766 - val_categorical_accuracy: 0.5664
Epoch 7/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0354 - categorical_accuracy: 0.7925 - val_loss: 0.0609 - val_categorical_accuracy: 0.6564
Epoch 8/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0320 - categorical_accuracy: 0.8140 - val_loss: 0.0529 - val_categorical_accuracy: 0.7089
Epoch 9/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0288 - categorical_accuracy: 0.8338 - val_loss: 0.0481 - val_categorical_accuracy: 0.7342
Epoch 10/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0266 - categorical_accuracy: 0.8487 - val_loss: 0.0433 - val_categorical_accuracy: 0.7711
Epoch 11/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0244 - categorical_accuracy: 0.8607 - val_loss: 0.0394 - val_categorical_accuracy: 0.7927
Epoch 12/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0231 - categorical_accuracy: 0.8694 - val_loss: 0.0356 - val_categorical_accuracy: 0.8166
Epoch 13/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0219 - categorical_accuracy: 0.8756 - val_loss: 0.0329 - val_categorical_accuracy: 0.8304
Epoch 14/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0205 - categorical_accuracy: 0.8848 - val_loss: 0.0317 - val_categorical_accuracy: 0.8358
Epoch 15/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0195 - categorical_accuracy: 0.8918 - val_loss: 0.0301 - val_categorical_accuracy: 0.8484
Epoch 16/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0187 - categorical_accuracy: 0.8962 - val_loss: 0.0285 - val_categorical_accuracy: 0.8618
Epoch 17/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0179 - categorical_accuracy: 0.8998 - val_loss: 0.0287 - val_categorical_accuracy: 0.8567
Epoch 18/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0172 - categorical_accuracy: 0.9045 - val_loss: 0.0259 - val_categorical_accuracy: 0.8734
Epoch 19/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0166 - categorical_accuracy: 0.9072 - val_loss: 0.0271 - val_categorical_accuracy: 0.8678
Epoch 20/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0161 - categorical_accuracy: 0.9113 - val_loss: 0.0256 - val_categorical_accuracy: 0.8752
Epoch 21/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0154 - categorical_accuracy: 0.9140 - val_loss: 0.0245 - val_categorical_accuracy: 0.8823
Epoch 22/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0149 - categorical_accuracy: 0.9173 - val_loss: 0.0217 - val_categorical_accuracy: 0.8962
Epoch 23/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0144 - categorical_accuracy: 0.9199 - val_loss: 0.0220 - val_categorical_accuracy: 0.8936
Epoch 24/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0144 - categorical_accuracy: 0.9199 - val_loss: 0.0208 - val_categorical_accuracy: 0.8993
Epoch 25/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0137 - categorical_accuracy: 0.9244 - val_loss: 0.0206 - val_categorical_accuracy: 0.8994
Epoch 26/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0135 - categorical_accuracy: 0.9252 - val_loss: 0.0207 - val_categorical_accuracy: 0.9003
Epoch 27/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0131 - categorical_accuracy: 0.9276 - val_loss: 0.0203 - val_categorical_accuracy: 0.9023
Epoch 28/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0128 - categorical_accuracy: 0.9288 - val_loss: 0.0189 - val_categorical_accuracy: 0.9078
Epoch 29/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0124 - categorical_accuracy: 0.9311 - val_loss: 0.0196 - val_categorical_accuracy: 0.9066
Epoch 30/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0123 - categorical_accuracy: 0.9322 - val_loss: 0.0189 - val_categorical_accuracy: 0.9077
Epoch 31/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0120 - categorical_accuracy: 0.9345 - val_loss: 0.0178 - val_categorical_accuracy: 0.9128
Epoch 32/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0117 - categorical_accuracy: 0.9348 - val_loss: 0.0182 - val_categorical_accuracy: 0.9134
Epoch 33/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0117 - categorical_accuracy: 0.9348 - val_loss: 0.0192 - val_categorical_accuracy: 0.9090
Epoch 34/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0113 - categorical_accuracy: 0.9372 - val_loss: 0.0185 - val_categorical_accuracy: 0.9090
Epoch 35/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0111 - categorical_accuracy: 0.9388 - val_loss: 0.0165 - val_categorical_accuracy: 0.9218
Epoch 36/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0109 - categorical_accuracy: 0.9399 - val_loss: 0.0168 - val_categorical_accuracy: 0.9169
Epoch 37/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0107 - categorical_accuracy: 0.9411 - val_loss: 0.0165 - val_categorical_accuracy: 0.9189
Epoch 38/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0107 - categorical_accuracy: 0.9405 - val_loss: 0.0171 - val_categorical_accuracy: 0.9182
Epoch 39/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0104 - categorical_accuracy: 0.9427 - val_loss: 0.0173 - val_categorical_accuracy: 0.9177
Epoch 40/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0102 - categorical_accuracy: 0.9442 - val_loss: 0.0177 - val_categorical_accuracy: 0.9155
Epoch 41/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0103 - categorical_accuracy: 0.9440 - val_loss: 0.0174 - val_categorical_accuracy: 0.9162

Epoch 00041: ReduceLROnPlateau reducing learning rate to 0.020000000298023225.
Epoch 42/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0100 - categorical_accuracy: 0.9441 - val_loss: 0.0173 - val_categorical_accuracy: 0.9166
Epoch 43/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0097 - categorical_accuracy: 0.9460 - val_loss: 0.0174 - val_categorical_accuracy: 0.9155
Epoch 44/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0097 - categorical_accuracy: 0.9456 - val_loss: 0.0166 - val_categorical_accuracy: 0.9187
Epoch 45/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0097 - categorical_accuracy: 0.9467 - val_loss: 0.0168 - val_categorical_accuracy: 0.9181
Epoch 46/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0096 - categorical_accuracy: 0.9473 - val_loss: 0.0166 - val_categorical_accuracy: 0.9199

Epoch 00046: ReduceLROnPlateau reducing learning rate to 0.003999999910593033.
Epoch 47/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0096 - categorical_accuracy: 0.9468 - val_loss: 0.0167 - val_categorical_accuracy: 0.9193
Epoch 48/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0095 - categorical_accuracy: 0.9474 - val_loss: 0.0168 - val_categorical_accuracy: 0.9187
Epoch 49/65535
420/420 [==============================] - 38s 90ms/step - loss: 0.0095 - categorical_accuracy: 0.9468 - val_loss: 0.0167 - val_categorical_accuracy: 0.9193
Epoch 50/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0095 - categorical_accuracy: 0.9480 - val_loss: 0.0167 - val_categorical_accuracy: 0.9190
Epoch 51/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0096 - categorical_accuracy: 0.9463 - val_loss: 0.0167 - val_categorical_accuracy: 0.9187

Epoch 00051: ReduceLROnPlateau reducing learning rate to 0.0007999999448657036.
Epoch 52/65535
420/420 [==============================] - 38s 91ms/step - loss: 0.0096 - categorical_accuracy: 0.9471 - val_loss: 0.0166 - val_categorical_accuracy: 0.9193
Epoch 00052: early stopping
========= generating oof predictions 04:30:28 =========
========= generating test set predictions 04:30:30 =========
train loss avg 0.008009895158275666 -- std 0.001347927785256733, val loss avg 0.011045032944089195 -- std 0.003105035077618878
train acc avg 0.9558232098921975 -- std 0.0073192588619730705, val acc avg 0.9443904753781144 -- std 0.014292469488286382
mean nb epochs 79.6
dump oof predicted probs
dump test set predicted probs
