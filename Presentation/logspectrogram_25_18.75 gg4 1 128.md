ren (master *+) python $ python train.py logspectrogram_25_18.75 gg4 1 128

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
conv2d_1 (Conv2D)               (None, 55, 201, 16)  272         reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 55, 201, 16)  64          conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 55, 201, 16)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 55, 201, 16)  4112        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 55, 201, 16)  64          conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 55, 201, 16)  0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 27, 100, 16)  0           activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 27, 100, 32)  8224        max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 27, 100, 32)  128         conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 27, 100, 32)  0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 27, 100, 32)  16416       activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 27, 100, 32)  128         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 27, 100, 32)  0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 13, 50, 32)   0           activation_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 13, 50, 64)   32832       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 13, 50, 64)   256         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 13, 50, 64)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 13, 50, 64)   65600       activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 13, 50, 64)   256         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 13, 50, 64)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 6, 25, 64)    0           activation_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 6, 25, 128)   131200      max_pooling2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 6, 25, 128)   512         conv2d_7[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 6, 25, 128)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 6, 25, 128)   262272      activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 6, 25, 128)   512         conv2d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 6, 25, 128)   0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 3, 12, 128)   0           activation_8[0][0]
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
Total params: 664,447
Trainable params: 662,463
Non-trainable params: 1,984
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 18:57:29 =========
Epoch 1/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.1311 - categorical_accuracy: 0.2033 - val_loss: 0.0854 - val_categorical_accuracy: 0.4354
Epoch 2/65535
420/420 [==============================] - 45s 108ms/step - loss: 0.0403 - categorical_accuracy: 0.7595 - val_loss: 0.0331 - val_categorical_accuracy: 0.8022
Epoch 3/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0219 - categorical_accuracy: 0.8793 - val_loss: 0.0185 - val_categorical_accuracy: 0.8927
Epoch 4/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0163 - categorical_accuracy: 0.9128 - val_loss: 0.0113 - val_categorical_accuracy: 0.9364
Epoch 5/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0130 - categorical_accuracy: 0.9307 - val_loss: 0.0114 - val_categorical_accuracy: 0.9326
Epoch 6/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0112 - categorical_accuracy: 0.9391 - val_loss: 0.0125 - val_categorical_accuracy: 0.9296
Epoch 7/65535
420/420 [==============================] - 47s 111ms/step - loss: 0.0099 - categorical_accuracy: 0.9469 - val_loss: 0.0094 - val_categorical_accuracy: 0.9472
Epoch 8/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0087 - categorical_accuracy: 0.9544 - val_loss: 0.0087 - val_categorical_accuracy: 0.9516
Epoch 9/65535
420/420 [==============================] - 47s 111ms/step - loss: 0.0078 - categorical_accuracy: 0.9586 - val_loss: 0.0096 - val_categorical_accuracy: 0.9461
Epoch 10/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0071 - categorical_accuracy: 0.9625 - val_loss: 0.0090 - val_categorical_accuracy: 0.9493
Epoch 11/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0064 - categorical_accuracy: 0.9661 - val_loss: 0.0096 - val_categorical_accuracy: 0.9469
Epoch 12/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0059 - categorical_accuracy: 0.9683 - val_loss: 0.0095 - val_categorical_accuracy: 0.9466
Epoch 13/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0052 - categorical_accuracy: 0.9721 - val_loss: 0.0101 - val_categorical_accuracy: 0.9481
Epoch 14/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0049 - categorical_accuracy: 0.9739 - val_loss: 0.0096 - val_categorical_accuracy: 0.9495

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 15/65535
420/420 [==============================] - 47s 111ms/step - loss: 0.0034 - categorical_accuracy: 0.9817 - val_loss: 0.0071 - val_categorical_accuracy: 0.9634
Epoch 16/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0029 - categorical_accuracy: 0.9854 - val_loss: 0.0072 - val_categorical_accuracy: 0.9636
Epoch 17/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0026 - categorical_accuracy: 0.9870 - val_loss: 0.0071 - val_categorical_accuracy: 0.9652
Epoch 18/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0024 - categorical_accuracy: 0.9884 - val_loss: 0.0074 - val_categorical_accuracy: 0.9648
Epoch 19/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0022 - categorical_accuracy: 0.9893 - val_loss: 0.0074 - val_categorical_accuracy: 0.9653
Epoch 20/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0021 - categorical_accuracy: 0.9899 - val_loss: 0.0072 - val_categorical_accuracy: 0.9652
Epoch 21/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0019 - categorical_accuracy: 0.9916 - val_loss: 0.0073 - val_categorical_accuracy: 0.9641

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 22/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0018 - categorical_accuracy: 0.9922 - val_loss: 0.0072 - val_categorical_accuracy: 0.9652
Epoch 23/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0017 - categorical_accuracy: 0.9926 - val_loss: 0.0072 - val_categorical_accuracy: 0.9660
Epoch 24/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0017 - categorical_accuracy: 0.9928 - val_loss: 0.0072 - val_categorical_accuracy: 0.9665
Epoch 25/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0016 - categorical_accuracy: 0.9930 - val_loss: 0.0072 - val_categorical_accuracy: 0.9661
Epoch 26/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0016 - categorical_accuracy: 0.9930 - val_loss: 0.0072 - val_categorical_accuracy: 0.9662

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 27/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0015 - categorical_accuracy: 0.9938 - val_loss: 0.0072 - val_categorical_accuracy: 0.9659
Epoch 28/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0016 - categorical_accuracy: 0.9933 - val_loss: 0.0072 - val_categorical_accuracy: 0.9658
Epoch 29/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0016 - categorical_accuracy: 0.9929 - val_loss: 0.0072 - val_categorical_accuracy: 0.9660
Epoch 30/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0015 - categorical_accuracy: 0.9940 - val_loss: 0.0072 - val_categorical_accuracy: 0.9656
Epoch 00030: early stopping
========= generating oof predictions 19:20:31 =========
========= generating test set predictions 19:20:34 =========
========= fitting 2 th model 19:21:19 =========
Epoch 1/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.1314 - categorical_accuracy: 0.2039 - val_loss: 0.0866 - val_categorical_accuracy: 0.4011
Epoch 2/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0421 - categorical_accuracy: 0.7495 - val_loss: 0.0343 - val_categorical_accuracy: 0.8023
Epoch 3/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0222 - categorical_accuracy: 0.8782 - val_loss: 0.0184 - val_categorical_accuracy: 0.8971
Epoch 4/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0164 - categorical_accuracy: 0.9116 - val_loss: 0.0189 - val_categorical_accuracy: 0.8927
Epoch 5/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0135 - categorical_accuracy: 0.9274 - val_loss: 0.0181 - val_categorical_accuracy: 0.8967
Epoch 6/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0113 - categorical_accuracy: 0.9396 - val_loss: 0.0118 - val_categorical_accuracy: 0.9343
Epoch 7/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0101 - categorical_accuracy: 0.9458 - val_loss: 0.0112 - val_categorical_accuracy: 0.9384
Epoch 8/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0087 - categorical_accuracy: 0.9535 - val_loss: 0.0105 - val_categorical_accuracy: 0.9429
Epoch 9/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0080 - categorical_accuracy: 0.9573 - val_loss: 0.0111 - val_categorical_accuracy: 0.9404
Epoch 10/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0073 - categorical_accuracy: 0.9618 - val_loss: 0.0119 - val_categorical_accuracy: 0.9360
Epoch 11/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0064 - categorical_accuracy: 0.9658 - val_loss: 0.0101 - val_categorical_accuracy: 0.9477
Epoch 12/65535
420/420 [==============================] - 46s 108ms/step - loss: 0.0058 - categorical_accuracy: 0.9693 - val_loss: 0.0106 - val_categorical_accuracy: 0.9427
Epoch 13/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0053 - categorical_accuracy: 0.9722 - val_loss: 0.0099 - val_categorical_accuracy: 0.9490
Epoch 14/65535
420/420 [==============================] - 46s 110ms/step - loss: 0.0048 - categorical_accuracy: 0.9749 - val_loss: 0.0137 - val_categorical_accuracy: 0.9284
Epoch 15/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0044 - categorical_accuracy: 0.9766 - val_loss: 0.0089 - val_categorical_accuracy: 0.9541
Epoch 16/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0041 - categorical_accuracy: 0.9785 - val_loss: 0.0096 - val_categorical_accuracy: 0.9507
Epoch 17/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0037 - categorical_accuracy: 0.9813 - val_loss: 0.0104 - val_categorical_accuracy: 0.9478
Epoch 18/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0033 - categorical_accuracy: 0.9821 - val_loss: 0.0090 - val_categorical_accuracy: 0.9552
Epoch 19/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0031 - categorical_accuracy: 0.9844 - val_loss: 0.0107 - val_categorical_accuracy: 0.9486
Epoch 20/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0028 - categorical_accuracy: 0.9859 - val_loss: 0.0117 - val_categorical_accuracy: 0.9456
Epoch 21/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0026 - categorical_accuracy: 0.9872 - val_loss: 0.0102 - val_categorical_accuracy: 0.9515

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 22/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0015 - categorical_accuracy: 0.9932 - val_loss: 0.0081 - val_categorical_accuracy: 0.9630
Epoch 23/65535
420/420 [==============================] - 46s 109ms/step - loss: 0.0012 - categorical_accuracy: 0.9951 - val_loss: 0.0081 - val_categorical_accuracy: 0.9629
Epoch 24/65535
420/420 [==============================] - 46s 108ms/step - loss: 0.0011 - categorical_accuracy: 0.9956 - val_loss: 0.0082 - val_categorical_accuracy: 0.9642
Epoch 25/65535
420/420 [==============================] - 46s 110ms/step - loss: 9.3502e-04 - categorical_accuracy: 0.9967 - val_loss: 0.0082 - val_categorical_accuracy: 0.9645
Epoch 26/65535
420/420 [==============================] - 45s 108ms/step - loss: 8.6051e-04 - categorical_accuracy: 0.9968 - val_loss: 0.0082 - val_categorical_accuracy: 0.9641
Epoch 27/65535
420/420 [==============================] - 45s 108ms/step - loss: 7.9491e-04 - categorical_accuracy: 0.9973 - val_loss: 0.0084 - val_categorical_accuracy: 0.9654
Epoch 28/65535
420/420 [==============================] - 45s 108ms/step - loss: 7.5727e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0086 - val_categorical_accuracy: 0.9638

Epoch 00028: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 29/65535
420/420 [==============================] - 45s 108ms/step - loss: 6.5600e-04 - categorical_accuracy: 0.9980 - val_loss: 0.0083 - val_categorical_accuracy: 0.9649
Epoch 30/65535
420/420 [==============================] - 45s 108ms/step - loss: 6.4596e-04 - categorical_accuracy: 0.9981 - val_loss: 0.0083 - val_categorical_accuracy: 0.9647
Epoch 31/65535
420/420 [==============================] - 45s 108ms/step - loss: 6.1479e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0083 - val_categorical_accuracy: 0.9645
Epoch 32/65535
420/420 [==============================] - 46s 108ms/step - loss: 6.1372e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0083 - val_categorical_accuracy: 0.9646
Epoch 33/65535
420/420 [==============================] - 46s 110ms/step - loss: 5.8053e-04 - categorical_accuracy: 0.9987 - val_loss: 0.0084 - val_categorical_accuracy: 0.9646

Epoch 00033: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 34/65535
420/420 [==============================] - 47s 112ms/step - loss: 5.8052e-04 - categorical_accuracy: 0.9983 - val_loss: 0.0083 - val_categorical_accuracy: 0.9646
Epoch 35/65535
420/420 [==============================] - 47s 111ms/step - loss: 5.7403e-04 - categorical_accuracy: 0.9984 - val_loss: 0.0083 - val_categorical_accuracy: 0.9649
Epoch 36/65535
420/420 [==============================] - 47s 112ms/step - loss: 5.8747e-04 - categorical_accuracy: 0.9983 - val_loss: 0.0083 - val_categorical_accuracy: 0.9650
Epoch 37/65535
420/420 [==============================] - 47s 113ms/step - loss: 5.6073e-04 - categorical_accuracy: 0.9985 - val_loss: 0.0083 - val_categorical_accuracy: 0.9647
Epoch 00037: early stopping
========= generating oof predictions 19:49:40 =========
========= generating test set predictions 19:49:43 =========
========= fitting 3 th model 19:50:29 =========
Epoch 1/65535
420/420 [==============================] - 47s 112ms/step - loss: 0.1417 - categorical_accuracy: 0.1334 - val_loss: 0.1160 - val_categorical_accuracy: 0.2013
Epoch 2/65535
420/420 [==============================] - 48s 113ms/step - loss: 0.0566 - categorical_accuracy: 0.6431 - val_loss: 0.0543 - val_categorical_accuracy: 0.6533
Epoch 3/65535
420/420 [==============================] - 47s 112ms/step - loss: 0.0258 - categorical_accuracy: 0.8576 - val_loss: 0.0255 - val_categorical_accuracy: 0.8513
Epoch 4/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0183 - categorical_accuracy: 0.9010 - val_loss: 0.0134 - val_categorical_accuracy: 0.9243
Epoch 5/65535
420/420 [==============================] - 47s 112ms/step - loss: 0.0147 - categorical_accuracy: 0.9217 - val_loss: 0.0118 - val_categorical_accuracy: 0.9354
Epoch 6/65535
420/420 [==============================] - 47s 112ms/step - loss: 0.0121 - categorical_accuracy: 0.9357 - val_loss: 0.0136 - val_categorical_accuracy: 0.9253
Epoch 7/65535
420/420 [==============================] - 47s 112ms/step - loss: 0.0108 - categorical_accuracy: 0.9424 - val_loss: 0.0107 - val_categorical_accuracy: 0.9418
Epoch 8/65535
420/420 [==============================] - 47s 113ms/step - loss: 0.0097 - categorical_accuracy: 0.9498 - val_loss: 0.0105 - val_categorical_accuracy: 0.9448
Epoch 9/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0090 - categorical_accuracy: 0.9526 - val_loss: 0.0113 - val_categorical_accuracy: 0.9368
Epoch 10/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0081 - categorical_accuracy: 0.9570 - val_loss: 0.0097 - val_categorical_accuracy: 0.9472
Epoch 11/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0073 - categorical_accuracy: 0.9617 - val_loss: 0.0107 - val_categorical_accuracy: 0.9447
Epoch 12/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0066 - categorical_accuracy: 0.9644 - val_loss: 0.0099 - val_categorical_accuracy: 0.9462
Epoch 13/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0062 - categorical_accuracy: 0.9665 - val_loss: 0.0103 - val_categorical_accuracy: 0.9458
Epoch 14/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0057 - categorical_accuracy: 0.9698 - val_loss: 0.0093 - val_categorical_accuracy: 0.9529
Epoch 15/65535
420/420 [==============================] - 49s 116ms/step - loss: 0.0053 - categorical_accuracy: 0.9718 - val_loss: 0.0094 - val_categorical_accuracy: 0.9525
Epoch 16/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0046 - categorical_accuracy: 0.9756 - val_loss: 0.0094 - val_categorical_accuracy: 0.9525
Epoch 17/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0044 - categorical_accuracy: 0.9772 - val_loss: 0.0085 - val_categorical_accuracy: 0.9584
Epoch 18/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0040 - categorical_accuracy: 0.9789 - val_loss: 0.0095 - val_categorical_accuracy: 0.9544
Epoch 19/65535
420/420 [==============================] - 48s 113ms/step - loss: 0.0038 - categorical_accuracy: 0.9794 - val_loss: 0.0102 - val_categorical_accuracy: 0.9502
Epoch 20/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0034 - categorical_accuracy: 0.9815 - val_loss: 0.0103 - val_categorical_accuracy: 0.9511
Epoch 21/65535
420/420 [==============================] - 48s 113ms/step - loss: 0.0033 - categorical_accuracy: 0.9828 - val_loss: 0.0121 - val_categorical_accuracy: 0.9469
Epoch 22/65535
420/420 [==============================] - 48s 113ms/step - loss: 0.0031 - categorical_accuracy: 0.9839 - val_loss: 0.0101 - val_categorical_accuracy: 0.9519
Epoch 23/65535
420/420 [==============================] - 47s 113ms/step - loss: 0.0029 - categorical_accuracy: 0.9850 - val_loss: 0.0194 - val_categorical_accuracy: 0.9075

Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 24/65535
420/420 [==============================] - 47s 113ms/step - loss: 0.0020 - categorical_accuracy: 0.9902 - val_loss: 0.0077 - val_categorical_accuracy: 0.9645
Epoch 25/65535
420/420 [==============================] - 47s 112ms/step - loss: 0.0014 - categorical_accuracy: 0.9938 - val_loss: 0.0078 - val_categorical_accuracy: 0.9651
Epoch 26/65535
420/420 [==============================] - 47s 113ms/step - loss: 0.0012 - categorical_accuracy: 0.9949 - val_loss: 0.0078 - val_categorical_accuracy: 0.9654
Epoch 27/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0011 - categorical_accuracy: 0.9955 - val_loss: 0.0079 - val_categorical_accuracy: 0.9660
Epoch 28/65535
420/420 [==============================] - 47s 113ms/step - loss: 0.0011 - categorical_accuracy: 0.9956 - val_loss: 0.0080 - val_categorical_accuracy: 0.9657
Epoch 29/65535
420/420 [==============================] - 47s 113ms/step - loss: 9.7655e-04 - categorical_accuracy: 0.9962 - val_loss: 0.0079 - val_categorical_accuracy: 0.9661
Epoch 30/65535
420/420 [==============================] - 48s 114ms/step - loss: 8.4898e-04 - categorical_accuracy: 0.9971 - val_loss: 0.0080 - val_categorical_accuracy: 0.9657

Epoch 00030: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 31/65535
420/420 [==============================] - 48s 113ms/step - loss: 7.6851e-04 - categorical_accuracy: 0.9974 - val_loss: 0.0079 - val_categorical_accuracy: 0.9657
Epoch 32/65535
420/420 [==============================] - 48s 113ms/step - loss: 7.7313e-04 - categorical_accuracy: 0.9972 - val_loss: 0.0079 - val_categorical_accuracy: 0.9654
Epoch 33/65535
420/420 [==============================] - 48s 114ms/step - loss: 7.2810e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0079 - val_categorical_accuracy: 0.9659
Epoch 34/65535
420/420 [==============================] - 47s 112ms/step - loss: 7.4421e-04 - categorical_accuracy: 0.9974 - val_loss: 0.0079 - val_categorical_accuracy: 0.9659
Epoch 35/65535
420/420 [==============================] - 47s 113ms/step - loss: 6.9463e-04 - categorical_accuracy: 0.9978 - val_loss: 0.0079 - val_categorical_accuracy: 0.9657

Epoch 00035: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 36/65535
420/420 [==============================] - 47s 113ms/step - loss: 7.0347e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0079 - val_categorical_accuracy: 0.9661
Epoch 37/65535
420/420 [==============================] - 47s 113ms/step - loss: 6.9850e-04 - categorical_accuracy: 0.9978 - val_loss: 0.0079 - val_categorical_accuracy: 0.9660
Epoch 38/65535
420/420 [==============================] - 47s 113ms/step - loss: 7.1727e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0079 - val_categorical_accuracy: 0.9658
Epoch 39/65535
420/420 [==============================] - 48s 114ms/step - loss: 6.7260e-04 - categorical_accuracy: 0.9980 - val_loss: 0.0079 - val_categorical_accuracy: 0.9657
Epoch 00039: early stopping
========= generating oof predictions 20:21:35 =========
========= generating test set predictions 20:21:39 =========
========= fitting 4 th model 20:22:26 =========
Epoch 1/65535
420/420 [==============================] - 47s 113ms/step - loss: 0.1073 - categorical_accuracy: 0.3515 - val_loss: 0.0657 - val_categorical_accuracy: 0.5922
Epoch 2/65535
420/420 [==============================] - 49s 115ms/step - loss: 0.0304 - categorical_accuracy: 0.8299 - val_loss: 0.0230 - val_categorical_accuracy: 0.8690
Epoch 3/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0186 - categorical_accuracy: 0.8983 - val_loss: 0.0161 - val_categorical_accuracy: 0.9082
Epoch 4/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0141 - categorical_accuracy: 0.9244 - val_loss: 0.0121 - val_categorical_accuracy: 0.9320
Epoch 5/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0116 - categorical_accuracy: 0.9380 - val_loss: 0.0100 - val_categorical_accuracy: 0.9436
Epoch 6/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0099 - categorical_accuracy: 0.9477 - val_loss: 0.0107 - val_categorical_accuracy: 0.9407
Epoch 7/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0088 - categorical_accuracy: 0.9533 - val_loss: 0.0092 - val_categorical_accuracy: 0.9488
Epoch 8/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0077 - categorical_accuracy: 0.9590 - val_loss: 0.0089 - val_categorical_accuracy: 0.9519
Epoch 9/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0069 - categorical_accuracy: 0.9637 - val_loss: 0.0094 - val_categorical_accuracy: 0.9501
Epoch 10/65535
420/420 [==============================] - 47s 113ms/step - loss: 0.0063 - categorical_accuracy: 0.9663 - val_loss: 0.0088 - val_categorical_accuracy: 0.9507
Epoch 11/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0056 - categorical_accuracy: 0.9704 - val_loss: 0.0081 - val_categorical_accuracy: 0.9553
Epoch 12/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0049 - categorical_accuracy: 0.9739 - val_loss: 0.0082 - val_categorical_accuracy: 0.9559
Epoch 13/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0047 - categorical_accuracy: 0.9751 - val_loss: 0.0091 - val_categorical_accuracy: 0.9525
Epoch 14/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0041 - categorical_accuracy: 0.9788 - val_loss: 0.0085 - val_categorical_accuracy: 0.9581
Epoch 15/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0038 - categorical_accuracy: 0.9797 - val_loss: 0.0083 - val_categorical_accuracy: 0.9570
Epoch 16/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0034 - categorical_accuracy: 0.9828 - val_loss: 0.0089 - val_categorical_accuracy: 0.9570
Epoch 17/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0031 - categorical_accuracy: 0.9834 - val_loss: 0.0100 - val_categorical_accuracy: 0.9539

Epoch 00017: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 18/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0022 - categorical_accuracy: 0.9892 - val_loss: 0.0073 - val_categorical_accuracy: 0.9644
Epoch 19/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0017 - categorical_accuracy: 0.9919 - val_loss: 0.0072 - val_categorical_accuracy: 0.9647
Epoch 20/65535
420/420 [==============================] - 47s 113ms/step - loss: 0.0015 - categorical_accuracy: 0.9933 - val_loss: 0.0072 - val_categorical_accuracy: 0.9645
Epoch 21/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0014 - categorical_accuracy: 0.9939 - val_loss: 0.0074 - val_categorical_accuracy: 0.9648
Epoch 22/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0013 - categorical_accuracy: 0.9939 - val_loss: 0.0073 - val_categorical_accuracy: 0.9652
Epoch 23/65535
420/420 [==============================] - 48s 113ms/step - loss: 0.0012 - categorical_accuracy: 0.9949 - val_loss: 0.0073 - val_categorical_accuracy: 0.9656
Epoch 24/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0011 - categorical_accuracy: 0.9956 - val_loss: 0.0075 - val_categorical_accuracy: 0.9654

Epoch 00024: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 25/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0010 - categorical_accuracy: 0.9962 - val_loss: 0.0073 - val_categorical_accuracy: 0.9660
Epoch 26/65535
420/420 [==============================] - 48s 115ms/step - loss: 9.8826e-04 - categorical_accuracy: 0.9961 - val_loss: 0.0073 - val_categorical_accuracy: 0.9659
Epoch 27/65535
420/420 [==============================] - 48s 115ms/step - loss: 9.8866e-04 - categorical_accuracy: 0.9963 - val_loss: 0.0073 - val_categorical_accuracy: 0.9656
Epoch 28/65535
420/420 [==============================] - 49s 117ms/step - loss: 9.4067e-04 - categorical_accuracy: 0.9965 - val_loss: 0.0073 - val_categorical_accuracy: 0.9655
Epoch 29/65535
420/420 [==============================] - 49s 116ms/step - loss: 9.1331e-04 - categorical_accuracy: 0.9968 - val_loss: 0.0073 - val_categorical_accuracy: 0.9654

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 30/65535
420/420 [==============================] - 49s 116ms/step - loss: 9.2131e-04 - categorical_accuracy: 0.9967 - val_loss: 0.0073 - val_categorical_accuracy: 0.9657
Epoch 31/65535
420/420 [==============================] - 49s 117ms/step - loss: 8.8084e-04 - categorical_accuracy: 0.9970 - val_loss: 0.0073 - val_categorical_accuracy: 0.9655
Epoch 32/65535
420/420 [==============================] - 49s 116ms/step - loss: 8.9573e-04 - categorical_accuracy: 0.9970 - val_loss: 0.0073 - val_categorical_accuracy: 0.9656
Epoch 33/65535
420/420 [==============================] - 48s 115ms/step - loss: 9.0746e-04 - categorical_accuracy: 0.9967 - val_loss: 0.0073 - val_categorical_accuracy: 0.9656
Epoch 34/65535
420/420 [==============================] - 49s 116ms/step - loss: 8.8011e-04 - categorical_accuracy: 0.9967 - val_loss: 0.0073 - val_categorical_accuracy: 0.9658

Epoch 00034: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 00034: early stopping
========= generating oof predictions 20:49:43 =========
========= generating test set predictions 20:49:47 =========
========= fitting 5 th model 20:50:36 =========
Epoch 1/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.1545 - categorical_accuracy: 0.0747 - val_loss: 0.1275 - val_categorical_accuracy: 0.1604
Epoch 2/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0788 - categorical_accuracy: 0.4907 - val_loss: 0.0439 - val_categorical_accuracy: 0.7346
Epoch 3/65535
420/420 [==============================] - 49s 117ms/step - loss: 0.0303 - categorical_accuracy: 0.8295 - val_loss: 0.0231 - val_categorical_accuracy: 0.8621
Epoch 4/65535
420/420 [==============================] - 49s 117ms/step - loss: 0.0197 - categorical_accuracy: 0.8914 - val_loss: 0.0226 - val_categorical_accuracy: 0.8734
Epoch 5/65535
420/420 [==============================] - 49s 117ms/step - loss: 0.0151 - categorical_accuracy: 0.9191 - val_loss: 0.0125 - val_categorical_accuracy: 0.9319
Epoch 6/65535
420/420 [==============================] - 49s 117ms/step - loss: 0.0125 - categorical_accuracy: 0.9327 - val_loss: 0.0145 - val_categorical_accuracy: 0.9201
Epoch 7/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0108 - categorical_accuracy: 0.9431 - val_loss: 0.0122 - val_categorical_accuracy: 0.9329
Epoch 8/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0097 - categorical_accuracy: 0.9482 - val_loss: 0.0104 - val_categorical_accuracy: 0.9419
Epoch 9/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0084 - categorical_accuracy: 0.9554 - val_loss: 0.0136 - val_categorical_accuracy: 0.9284
Epoch 10/65535
420/420 [==============================] - 49s 115ms/step - loss: 0.0078 - categorical_accuracy: 0.9581 - val_loss: 0.0124 - val_categorical_accuracy: 0.9330
Epoch 11/65535
420/420 [==============================] - 49s 116ms/step - loss: 0.0068 - categorical_accuracy: 0.9641 - val_loss: 0.0109 - val_categorical_accuracy: 0.9402
Epoch 12/65535
420/420 [==============================] - 49s 116ms/step - loss: 0.0062 - categorical_accuracy: 0.9672 - val_loss: 0.0091 - val_categorical_accuracy: 0.9509
Epoch 13/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0058 - categorical_accuracy: 0.9696 - val_loss: 0.0093 - val_categorical_accuracy: 0.9518
Epoch 14/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0053 - categorical_accuracy: 0.9720 - val_loss: 0.0089 - val_categorical_accuracy: 0.9543
Epoch 15/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0048 - categorical_accuracy: 0.9750 - val_loss: 0.0088 - val_categorical_accuracy: 0.9532
Epoch 16/65535
420/420 [==============================] - 49s 116ms/step - loss: 0.0044 - categorical_accuracy: 0.9769 - val_loss: 0.0093 - val_categorical_accuracy: 0.9541
Epoch 17/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0041 - categorical_accuracy: 0.9775 - val_loss: 0.0095 - val_categorical_accuracy: 0.9530
Epoch 18/65535
420/420 [==============================] - 49s 116ms/step - loss: 0.0038 - categorical_accuracy: 0.9796 - val_loss: 0.0091 - val_categorical_accuracy: 0.9547
Epoch 19/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0033 - categorical_accuracy: 0.9827 - val_loss: 0.0099 - val_categorical_accuracy: 0.9538
Epoch 20/65535
420/420 [==============================] - 49s 116ms/step - loss: 0.0030 - categorical_accuracy: 0.9845 - val_loss: 0.0098 - val_categorical_accuracy: 0.9518
Epoch 21/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0030 - categorical_accuracy: 0.9844 - val_loss: 0.0094 - val_categorical_accuracy: 0.9537

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 22/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0019 - categorical_accuracy: 0.9907 - val_loss: 0.0076 - val_categorical_accuracy: 0.9641
Epoch 23/65535
420/420 [==============================] - 48s 114ms/step - loss: 0.0015 - categorical_accuracy: 0.9933 - val_loss: 0.0077 - val_categorical_accuracy: 0.9647
Epoch 24/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0014 - categorical_accuracy: 0.9939 - val_loss: 0.0078 - val_categorical_accuracy: 0.9644
Epoch 25/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0012 - categorical_accuracy: 0.9949 - val_loss: 0.0078 - val_categorical_accuracy: 0.9640
Epoch 26/65535
420/420 [==============================] - 48s 115ms/step - loss: 0.0012 - categorical_accuracy: 0.9951 - val_loss: 0.0081 - val_categorical_accuracy: 0.9641
Epoch 27/65535
420/420 [==============================] - 49s 116ms/step - loss: 0.0011 - categorical_accuracy: 0.9956 - val_loss: 0.0079 - val_categorical_accuracy: 0.9641
Epoch 28/65535
420/420 [==============================] - 49s 116ms/step - loss: 9.8472e-04 - categorical_accuracy: 0.9964 - val_loss: 0.0081 - val_categorical_accuracy: 0.9638

Epoch 00028: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 29/65535
420/420 [==============================] - 48s 114ms/step - loss: 9.6266e-04 - categorical_accuracy: 0.9966 - val_loss: 0.0079 - val_categorical_accuracy: 0.9645
Epoch 30/65535
420/420 [==============================] - 49s 116ms/step - loss: 8.6415e-04 - categorical_accuracy: 0.9968 - val_loss: 0.0079 - val_categorical_accuracy: 0.9647
Epoch 31/65535
420/420 [==============================] - 48s 115ms/step - loss: 8.1396e-04 - categorical_accuracy: 0.9974 - val_loss: 0.0079 - val_categorical_accuracy: 0.9645
Epoch 32/65535
420/420 [==============================] - 48s 115ms/step - loss: 7.9938e-04 - categorical_accuracy: 0.9974 - val_loss: 0.0078 - val_categorical_accuracy: 0.9650
Epoch 33/65535
420/420 [==============================] - 49s 117ms/step - loss: 8.3345e-04 - categorical_accuracy: 0.9968 - val_loss: 0.0079 - val_categorical_accuracy: 0.9649

Epoch 00033: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 34/65535
420/420 [==============================] - 48s 115ms/step - loss: 7.9110e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0078 - val_categorical_accuracy: 0.9649
Epoch 35/65535
420/420 [==============================] - 49s 116ms/step - loss: 7.7406e-04 - categorical_accuracy: 0.9975 - val_loss: 0.0079 - val_categorical_accuracy: 0.9650
Epoch 36/65535
420/420 [==============================] - 48s 115ms/step - loss: 7.6794e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0079 - val_categorical_accuracy: 0.9650
Epoch 37/65535
420/420 [==============================] - 49s 115ms/step - loss: 7.9763e-04 - categorical_accuracy: 0.9973 - val_loss: 0.0079 - val_categorical_accuracy: 0.9650
Epoch 00037: early stopping
========= generating oof predictions 21:20:34 =========
========= generating test set predictions 21:20:38 =========
train loss avg 0.0008828004574763085 -- std 0.0003273338821216566, val loss avg 0.007712638949196033 -- std 0.0004210573980819042
train acc avg 0.9968935562457236 -- std 0.0015911663439375653, val acc avg 0.9653759928521632 -- std 0.0004271440710042162
mean nb epochs 35.4
dump oof predicted probs
dump test set predicted probs
