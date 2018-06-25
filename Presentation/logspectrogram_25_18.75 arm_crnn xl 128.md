ren (master *+) python $ python train.py logspectrogram_25_18.75 arm_crnn xl 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

======= processing data =======
========== input shape is : (55, 201) ===========
------------- SUMMARY OF MODEL -------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 55, 201)           0
_________________________________________________________________
reshape_1 (Reshape)          (None, 55, 201, 1)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 23, 198, 168)      6888
_________________________________________________________________
activation_1 (Activation)    (None, 23, 198, 168)      0
_________________________________________________________________
dropout_1 (Dropout)          (None, 23, 198, 168)      0
_________________________________________________________________
permute_1 (Permute)          (None, 23, 168, 198)      0
_________________________________________________________________
reshape_2 (Reshape)          (None, 23, 33264)         0
_________________________________________________________________
cu_dnngru_1 (CuDNNGRU)       (None, 23, 168)           16850736
_________________________________________________________________
cu_dnngru_2 (CuDNNGRU)       (None, 168)               170352
_________________________________________________________________
dense_1 (Dense)              (None, 256)               43264
_________________________________________________________________
activation_2 (Activation)    (None, 256)               0
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 31)                7967
=================================================================
Total params: 17,079,207
Trainable params: 17,079,207
Non-trainable params: 0
_________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 16:42:12 =========
Epoch 1/65535
420/420 [==============================] - 50s 120ms/step - loss: 0.0661 - categorical_accuracy: 0.5804 - val_loss: 0.0320 - val_categorical_accuracy: 0.8131
Epoch 2/65535
420/420 [==============================] - 49s 116ms/step - loss: 0.0244 - categorical_accuracy: 0.8600 - val_loss: 0.0193 - val_categorical_accuracy: 0.8872
Epoch 3/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0173 - categorical_accuracy: 0.9004 - val_loss: 0.0158 - val_categorical_accuracy: 0.9083
Epoch 4/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0139 - categorical_accuracy: 0.9202 - val_loss: 0.0144 - val_categorical_accuracy: 0.9159
Epoch 5/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0121 - categorical_accuracy: 0.9312 - val_loss: 0.0136 - val_categorical_accuracy: 0.9218
Epoch 6/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0104 - categorical_accuracy: 0.9405 - val_loss: 0.0128 - val_categorical_accuracy: 0.9257
Epoch 7/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0094 - categorical_accuracy: 0.9464 - val_loss: 0.0132 - val_categorical_accuracy: 0.9252
Epoch 8/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0081 - categorical_accuracy: 0.9540 - val_loss: 0.0136 - val_categorical_accuracy: 0.9257
Epoch 9/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0077 - categorical_accuracy: 0.9565 - val_loss: 0.0123 - val_categorical_accuracy: 0.9324
Epoch 10/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0073 - categorical_accuracy: 0.9578 - val_loss: 0.0148 - val_categorical_accuracy: 0.9222
Epoch 11/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0062 - categorical_accuracy: 0.9643 - val_loss: 0.0122 - val_categorical_accuracy: 0.9346
Epoch 12/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0057 - categorical_accuracy: 0.9668 - val_loss: 0.0124 - val_categorical_accuracy: 0.9330
Epoch 13/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0062 - categorical_accuracy: 0.9639 - val_loss: 0.0129 - val_categorical_accuracy: 0.9322
Epoch 14/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0054 - categorical_accuracy: 0.9685 - val_loss: 0.0141 - val_categorical_accuracy: 0.9274
Epoch 15/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0048 - categorical_accuracy: 0.9724 - val_loss: 0.0130 - val_categorical_accuracy: 0.9344

Epoch 00015: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 16/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0027 - categorical_accuracy: 0.9845 - val_loss: 0.0105 - val_categorical_accuracy: 0.9492
Epoch 17/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0020 - categorical_accuracy: 0.9889 - val_loss: 0.0107 - val_categorical_accuracy: 0.9494
Epoch 18/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0017 - categorical_accuracy: 0.9907 - val_loss: 0.0110 - val_categorical_accuracy: 0.9494
Epoch 19/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0015 - categorical_accuracy: 0.9922 - val_loss: 0.0113 - val_categorical_accuracy: 0.9492
Epoch 20/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0013 - categorical_accuracy: 0.9934 - val_loss: 0.0114 - val_categorical_accuracy: 0.9500
Epoch 21/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0011 - categorical_accuracy: 0.9945 - val_loss: 0.0122 - val_categorical_accuracy: 0.9483
Epoch 22/65535
420/420 [==============================] - 49s 118ms/step - loss: 9.4239e-04 - categorical_accuracy: 0.9953 - val_loss: 0.0125 - val_categorical_accuracy: 0.9504

Epoch 00022: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 23/65535
420/420 [==============================] - 50s 118ms/step - loss: 6.8996e-04 - categorical_accuracy: 0.9968 - val_loss: 0.0122 - val_categorical_accuracy: 0.9508
Epoch 24/65535
420/420 [==============================] - 49s 118ms/step - loss: 6.0100e-04 - categorical_accuracy: 0.9973 - val_loss: 0.0124 - val_categorical_accuracy: 0.9503
Epoch 25/65535
420/420 [==============================] - 50s 118ms/step - loss: 5.4789e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0127 - val_categorical_accuracy: 0.9496
Epoch 26/65535
420/420 [==============================] - 50s 118ms/step - loss: 5.1545e-04 - categorical_accuracy: 0.9979 - val_loss: 0.0125 - val_categorical_accuracy: 0.9507
Epoch 27/65535
420/420 [==============================] - 50s 118ms/step - loss: 4.5774e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0128 - val_categorical_accuracy: 0.9499

Epoch 00027: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 28/65535
420/420 [==============================] - 50s 118ms/step - loss: 4.1181e-04 - categorical_accuracy: 0.9984 - val_loss: 0.0128 - val_categorical_accuracy: 0.9498
Epoch 29/65535
420/420 [==============================] - 50s 118ms/step - loss: 3.8617e-04 - categorical_accuracy: 0.9986 - val_loss: 0.0129 - val_categorical_accuracy: 0.9497
Epoch 30/65535
420/420 [==============================] - 50s 118ms/step - loss: 3.6850e-04 - categorical_accuracy: 0.9988 - val_loss: 0.0129 - val_categorical_accuracy: 0.9502
Epoch 31/65535
420/420 [==============================] - 50s 118ms/step - loss: 3.4367e-04 - categorical_accuracy: 0.9988 - val_loss: 0.0130 - val_categorical_accuracy: 0.9501
Epoch 00031: early stopping
========= generating oof predictions 17:07:54 =========
========= generating test set predictions 17:07:57 =========
========= fitting 2 th model 17:08:45 =========
Epoch 1/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0684 - categorical_accuracy: 0.5633 - val_loss: 0.0303 - val_categorical_accuracy: 0.8237
Epoch 2/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0258 - categorical_accuracy: 0.8522 - val_loss: 0.0215 - val_categorical_accuracy: 0.8773
Epoch 3/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0185 - categorical_accuracy: 0.8942 - val_loss: 0.0199 - val_categorical_accuracy: 0.8851
Epoch 4/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0146 - categorical_accuracy: 0.9166 - val_loss: 0.0165 - val_categorical_accuracy: 0.9040
Epoch 5/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0125 - categorical_accuracy: 0.9283 - val_loss: 0.0149 - val_categorical_accuracy: 0.9139
Epoch 6/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0109 - categorical_accuracy: 0.9370 - val_loss: 0.0159 - val_categorical_accuracy: 0.9084
Epoch 7/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0093 - categorical_accuracy: 0.9466 - val_loss: 0.0134 - val_categorical_accuracy: 0.9238
Epoch 8/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0089 - categorical_accuracy: 0.9487 - val_loss: 0.0144 - val_categorical_accuracy: 0.9187
Epoch 9/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0082 - categorical_accuracy: 0.9535 - val_loss: 0.0131 - val_categorical_accuracy: 0.9272
Epoch 10/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0070 - categorical_accuracy: 0.9602 - val_loss: 0.0135 - val_categorical_accuracy: 0.9247
Epoch 11/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0074 - categorical_accuracy: 0.9571 - val_loss: 0.0149 - val_categorical_accuracy: 0.9177
Epoch 12/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0068 - categorical_accuracy: 0.9610 - val_loss: 0.0132 - val_categorical_accuracy: 0.9311
Epoch 13/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0061 - categorical_accuracy: 0.9650 - val_loss: 0.0144 - val_categorical_accuracy: 0.9231
Epoch 14/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0057 - categorical_accuracy: 0.9674 - val_loss: 0.0137 - val_categorical_accuracy: 0.9285
Epoch 15/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0048 - categorical_accuracy: 0.9728 - val_loss: 0.0140 - val_categorical_accuracy: 0.9291

Epoch 00015: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 16/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0030 - categorical_accuracy: 0.9832 - val_loss: 0.0114 - val_categorical_accuracy: 0.9427
Epoch 17/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0022 - categorical_accuracy: 0.9880 - val_loss: 0.0113 - val_categorical_accuracy: 0.9448
Epoch 18/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0019 - categorical_accuracy: 0.9899 - val_loss: 0.0114 - val_categorical_accuracy: 0.9454
Epoch 19/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0016 - categorical_accuracy: 0.9920 - val_loss: 0.0116 - val_categorical_accuracy: 0.9454
Epoch 20/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0014 - categorical_accuracy: 0.9926 - val_loss: 0.0122 - val_categorical_accuracy: 0.9455
Epoch 21/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0012 - categorical_accuracy: 0.9940 - val_loss: 0.0129 - val_categorical_accuracy: 0.9444
Epoch 22/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0011 - categorical_accuracy: 0.9942 - val_loss: 0.0133 - val_categorical_accuracy: 0.9440

Epoch 00022: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 23/65535
420/420 [==============================] - 50s 118ms/step - loss: 8.9037e-04 - categorical_accuracy: 0.9956 - val_loss: 0.0129 - val_categorical_accuracy: 0.9463
Epoch 24/65535
420/420 [==============================] - 50s 118ms/step - loss: 7.6356e-04 - categorical_accuracy: 0.9965 - val_loss: 0.0130 - val_categorical_accuracy: 0.9454
Epoch 25/65535
420/420 [==============================] - 50s 119ms/step - loss: 6.8285e-04 - categorical_accuracy: 0.9970 - val_loss: 0.0130 - val_categorical_accuracy: 0.9454
Epoch 26/65535
420/420 [==============================] - 50s 119ms/step - loss: 6.3186e-04 - categorical_accuracy: 0.9972 - val_loss: 0.0131 - val_categorical_accuracy: 0.9451
Epoch 27/65535
420/420 [==============================] - 50s 119ms/step - loss: 6.0263e-04 - categorical_accuracy: 0.9973 - val_loss: 0.0133 - val_categorical_accuracy: 0.9459

Epoch 00027: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 28/65535
420/420 [==============================] - 50s 119ms/step - loss: 5.5343e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0133 - val_categorical_accuracy: 0.9457
Epoch 29/65535
420/420 [==============================] - 50s 119ms/step - loss: 5.1804e-04 - categorical_accuracy: 0.9978 - val_loss: 0.0133 - val_categorical_accuracy: 0.9455
Epoch 30/65535
420/420 [==============================] - 50s 119ms/step - loss: 5.0554e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0134 - val_categorical_accuracy: 0.9452
Epoch 31/65535
420/420 [==============================] - 50s 119ms/step - loss: 4.7780e-04 - categorical_accuracy: 0.9980 - val_loss: 0.0135 - val_categorical_accuracy: 0.9453
Epoch 32/65535
420/420 [==============================] - 50s 118ms/step - loss: 4.5713e-04 - categorical_accuracy: 0.9980 - val_loss: 0.0136 - val_categorical_accuracy: 0.9448
Epoch 00032: early stopping
========= generating oof predictions 17:35:21 =========
========= generating test set predictions 17:35:25 =========
========= fitting 3 th model 17:36:14 =========
Epoch 1/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0656 - categorical_accuracy: 0.5827 - val_loss: 0.0280 - val_categorical_accuracy: 0.8327
Epoch 2/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0239 - categorical_accuracy: 0.8622 - val_loss: 0.0192 - val_categorical_accuracy: 0.8871
Epoch 3/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0178 - categorical_accuracy: 0.8986 - val_loss: 0.0182 - val_categorical_accuracy: 0.8950
Epoch 4/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0142 - categorical_accuracy: 0.9186 - val_loss: 0.0159 - val_categorical_accuracy: 0.9078
Epoch 5/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0118 - categorical_accuracy: 0.9323 - val_loss: 0.0151 - val_categorical_accuracy: 0.9140
Epoch 6/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0101 - categorical_accuracy: 0.9421 - val_loss: 0.0143 - val_categorical_accuracy: 0.9177
Epoch 7/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0092 - categorical_accuracy: 0.9472 - val_loss: 0.0154 - val_categorical_accuracy: 0.9164
Epoch 8/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0083 - categorical_accuracy: 0.9526 - val_loss: 0.0142 - val_categorical_accuracy: 0.9204
Epoch 9/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0078 - categorical_accuracy: 0.9544 - val_loss: 0.0150 - val_categorical_accuracy: 0.9177
Epoch 10/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0069 - categorical_accuracy: 0.9610 - val_loss: 0.0142 - val_categorical_accuracy: 0.9251
Epoch 11/65535
420/420 [==============================] - 51s 122ms/step - loss: 0.0066 - categorical_accuracy: 0.9618 - val_loss: 0.0156 - val_categorical_accuracy: 0.9163
Epoch 12/65535
420/420 [==============================] - 52s 123ms/step - loss: 0.0064 - categorical_accuracy: 0.9633 - val_loss: 0.0152 - val_categorical_accuracy: 0.9220

Epoch 00012: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 13/65535
420/420 [==============================] - 52s 124ms/step - loss: 0.0034 - categorical_accuracy: 0.9808 - val_loss: 0.0110 - val_categorical_accuracy: 0.9458
Epoch 14/65535
420/420 [==============================] - 51s 121ms/step - loss: 0.0025 - categorical_accuracy: 0.9862 - val_loss: 0.0112 - val_categorical_accuracy: 0.9462
Epoch 15/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0021 - categorical_accuracy: 0.9888 - val_loss: 0.0114 - val_categorical_accuracy: 0.9464
Epoch 16/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0018 - categorical_accuracy: 0.9911 - val_loss: 0.0118 - val_categorical_accuracy: 0.9463
Epoch 17/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0015 - categorical_accuracy: 0.9927 - val_loss: 0.0121 - val_categorical_accuracy: 0.9470
Epoch 18/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0013 - categorical_accuracy: 0.9936 - val_loss: 0.0126 - val_categorical_accuracy: 0.9446
Epoch 19/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0011 - categorical_accuracy: 0.9946 - val_loss: 0.0129 - val_categorical_accuracy: 0.9461

Epoch 00019: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 20/65535
420/420 [==============================] - 50s 118ms/step - loss: 8.0000e-04 - categorical_accuracy: 0.9964 - val_loss: 0.0128 - val_categorical_accuracy: 0.9456
Epoch 21/65535
420/420 [==============================] - 50s 118ms/step - loss: 6.9280e-04 - categorical_accuracy: 0.9970 - val_loss: 0.0130 - val_categorical_accuracy: 0.9462
Epoch 22/65535
420/420 [==============================] - 50s 118ms/step - loss: 6.2827e-04 - categorical_accuracy: 0.9973 - val_loss: 0.0131 - val_categorical_accuracy: 0.9468
Epoch 23/65535
420/420 [==============================] - 50s 118ms/step - loss: 5.9176e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0133 - val_categorical_accuracy: 0.9466
Epoch 24/65535
420/420 [==============================] - 50s 118ms/step - loss: 5.2249e-04 - categorical_accuracy: 0.9980 - val_loss: 0.0134 - val_categorical_accuracy: 0.9464

Epoch 00024: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 25/65535
420/420 [==============================] - 50s 118ms/step - loss: 4.8320e-04 - categorical_accuracy: 0.9983 - val_loss: 0.0136 - val_categorical_accuracy: 0.9467
Epoch 26/65535
420/420 [==============================] - 49s 118ms/step - loss: 4.6093e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0137 - val_categorical_accuracy: 0.9463
Epoch 27/65535
420/420 [==============================] - 50s 118ms/step - loss: 4.2289e-04 - categorical_accuracy: 0.9985 - val_loss: 0.0138 - val_categorical_accuracy: 0.9467
Epoch 28/65535
420/420 [==============================] - 50s 118ms/step - loss: 3.9679e-04 - categorical_accuracy: 0.9986 - val_loss: 0.0138 - val_categorical_accuracy: 0.9466
Epoch 00028: early stopping
========= generating oof predictions 17:59:34 =========
========= generating test set predictions 17:59:38 =========
========= fitting 4 th model 18:00:26 =========
Epoch 1/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0663 - categorical_accuracy: 0.5756 - val_loss: 0.0291 - val_categorical_accuracy: 0.8313
Epoch 2/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0247 - categorical_accuracy: 0.8581 - val_loss: 0.0217 - val_categorical_accuracy: 0.8772
Epoch 3/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0180 - categorical_accuracy: 0.8971 - val_loss: 0.0162 - val_categorical_accuracy: 0.9081
Epoch 4/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0148 - categorical_accuracy: 0.9145 - val_loss: 0.0158 - val_categorical_accuracy: 0.9112
Epoch 5/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0127 - categorical_accuracy: 0.9271 - val_loss: 0.0146 - val_categorical_accuracy: 0.9175
Epoch 6/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0112 - categorical_accuracy: 0.9351 - val_loss: 0.0136 - val_categorical_accuracy: 0.9234
Epoch 7/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0099 - categorical_accuracy: 0.9435 - val_loss: 0.0129 - val_categorical_accuracy: 0.9293
Epoch 8/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0086 - categorical_accuracy: 0.9502 - val_loss: 0.0147 - val_categorical_accuracy: 0.9170
Epoch 9/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0085 - categorical_accuracy: 0.9515 - val_loss: 0.0136 - val_categorical_accuracy: 0.9258
Epoch 10/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0074 - categorical_accuracy: 0.9572 - val_loss: 0.0158 - val_categorical_accuracy: 0.9144
Epoch 11/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0071 - categorical_accuracy: 0.9588 - val_loss: 0.0129 - val_categorical_accuracy: 0.9315
Epoch 12/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0067 - categorical_accuracy: 0.9620 - val_loss: 0.0135 - val_categorical_accuracy: 0.9293
Epoch 13/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0064 - categorical_accuracy: 0.9625 - val_loss: 0.0135 - val_categorical_accuracy: 0.9283

Epoch 00013: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 14/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0036 - categorical_accuracy: 0.9800 - val_loss: 0.0109 - val_categorical_accuracy: 0.9457
Epoch 15/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0027 - categorical_accuracy: 0.9850 - val_loss: 0.0109 - val_categorical_accuracy: 0.9465
Epoch 16/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0023 - categorical_accuracy: 0.9878 - val_loss: 0.0112 - val_categorical_accuracy: 0.9478
Epoch 17/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0020 - categorical_accuracy: 0.9890 - val_loss: 0.0116 - val_categorical_accuracy: 0.9472
Epoch 18/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0018 - categorical_accuracy: 0.9902 - val_loss: 0.0116 - val_categorical_accuracy: 0.9457
Epoch 19/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0016 - categorical_accuracy: 0.9918 - val_loss: 0.0121 - val_categorical_accuracy: 0.9475
Epoch 20/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0014 - categorical_accuracy: 0.9926 - val_loss: 0.0123 - val_categorical_accuracy: 0.9480

Epoch 00020: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 21/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0011 - categorical_accuracy: 0.9946 - val_loss: 0.0123 - val_categorical_accuracy: 0.9482
Epoch 22/65535
420/420 [==============================] - 50s 120ms/step - loss: 9.9447e-04 - categorical_accuracy: 0.9952 - val_loss: 0.0124 - val_categorical_accuracy: 0.9487
Epoch 23/65535
420/420 [==============================] - 51s 121ms/step - loss: 9.2590e-04 - categorical_accuracy: 0.9954 - val_loss: 0.0125 - val_categorical_accuracy: 0.9488
Epoch 24/65535
420/420 [==============================] - 50s 118ms/step - loss: 8.7620e-04 - categorical_accuracy: 0.9957 - val_loss: 0.0126 - val_categorical_accuracy: 0.9481
Epoch 25/65535
420/420 [==============================] - 50s 118ms/step - loss: 8.1522e-04 - categorical_accuracy: 0.9961 - val_loss: 0.0127 - val_categorical_accuracy: 0.9481

Epoch 00025: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 26/65535
420/420 [==============================] - 50s 118ms/step - loss: 7.2995e-04 - categorical_accuracy: 0.9967 - val_loss: 0.0128 - val_categorical_accuracy: 0.9483
Epoch 27/65535
420/420 [==============================] - 49s 118ms/step - loss: 6.9901e-04 - categorical_accuracy: 0.9968 - val_loss: 0.0130 - val_categorical_accuracy: 0.9483
Epoch 28/65535
420/420 [==============================] - 50s 118ms/step - loss: 6.8626e-04 - categorical_accuracy: 0.9968 - val_loss: 0.0130 - val_categorical_accuracy: 0.9485
Epoch 29/65535
420/420 [==============================] - 50s 118ms/step - loss: 6.5484e-04 - categorical_accuracy: 0.9971 - val_loss: 0.0131 - val_categorical_accuracy: 0.9489
Epoch 30/65535
420/420 [==============================] - 50s 118ms/step - loss: 6.1072e-04 - categorical_accuracy: 0.9972 - val_loss: 0.0132 - val_categorical_accuracy: 0.9484
Epoch 00030: early stopping
========= generating oof predictions 18:25:19 =========
========= generating test set predictions 18:25:22 =========
========= fitting 5 th model 18:26:11 =========
Epoch 1/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0674 - categorical_accuracy: 0.5704 - val_loss: 0.0306 - val_categorical_accuracy: 0.8198
Epoch 2/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0248 - categorical_accuracy: 0.8569 - val_loss: 0.0195 - val_categorical_accuracy: 0.8893
Epoch 3/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0175 - categorical_accuracy: 0.8991 - val_loss: 0.0176 - val_categorical_accuracy: 0.8982
Epoch 4/65535
420/420 [==============================] - 50s 119ms/step - loss: 0.0144 - categorical_accuracy: 0.9183 - val_loss: 0.0162 - val_categorical_accuracy: 0.9093
Epoch 5/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0123 - categorical_accuracy: 0.9304 - val_loss: 0.0148 - val_categorical_accuracy: 0.9145
Epoch 6/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0110 - categorical_accuracy: 0.9362 - val_loss: 0.0150 - val_categorical_accuracy: 0.9161
Epoch 7/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0094 - categorical_accuracy: 0.9463 - val_loss: 0.0134 - val_categorical_accuracy: 0.9217
Epoch 8/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0083 - categorical_accuracy: 0.9527 - val_loss: 0.0139 - val_categorical_accuracy: 0.9234
Epoch 9/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0078 - categorical_accuracy: 0.9551 - val_loss: 0.0147 - val_categorical_accuracy: 0.9205
Epoch 10/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0070 - categorical_accuracy: 0.9594 - val_loss: 0.0133 - val_categorical_accuracy: 0.9280
Epoch 11/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0067 - categorical_accuracy: 0.9615 - val_loss: 0.0131 - val_categorical_accuracy: 0.9315
Epoch 12/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0062 - categorical_accuracy: 0.9651 - val_loss: 0.0149 - val_categorical_accuracy: 0.9204
Epoch 13/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0063 - categorical_accuracy: 0.9640 - val_loss: 0.0155 - val_categorical_accuracy: 0.9216
Epoch 14/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0059 - categorical_accuracy: 0.9660 - val_loss: 0.0151 - val_categorical_accuracy: 0.9222
Epoch 15/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0056 - categorical_accuracy: 0.9677 - val_loss: 0.0141 - val_categorical_accuracy: 0.9295
Epoch 16/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0049 - categorical_accuracy: 0.9715 - val_loss: 0.0144 - val_categorical_accuracy: 0.9280
Epoch 17/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0050 - categorical_accuracy: 0.9713 - val_loss: 0.0150 - val_categorical_accuracy: 0.9247

Epoch 00017: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 18/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0027 - categorical_accuracy: 0.9847 - val_loss: 0.0115 - val_categorical_accuracy: 0.9428
Epoch 19/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0019 - categorical_accuracy: 0.9893 - val_loss: 0.0119 - val_categorical_accuracy: 0.9444
Epoch 20/65535
420/420 [==============================] - 49s 117ms/step - loss: 0.0016 - categorical_accuracy: 0.9911 - val_loss: 0.0121 - val_categorical_accuracy: 0.9451
Epoch 21/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0014 - categorical_accuracy: 0.9926 - val_loss: 0.0124 - val_categorical_accuracy: 0.9461
Epoch 22/65535
420/420 [==============================] - 50s 118ms/step - loss: 0.0012 - categorical_accuracy: 0.9937 - val_loss: 0.0128 - val_categorical_accuracy: 0.9448
Epoch 23/65535
420/420 [==============================] - 49s 118ms/step - loss: 0.0010 - categorical_accuracy: 0.9947 - val_loss: 0.0134 - val_categorical_accuracy: 0.9451
Epoch 24/65535
420/420 [==============================] - 49s 118ms/step - loss: 8.7665e-04 - categorical_accuracy: 0.9954 - val_loss: 0.0136 - val_categorical_accuracy: 0.9459

Epoch 00024: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 25/65535
420/420 [==============================] - 49s 118ms/step - loss: 6.8465e-04 - categorical_accuracy: 0.9969 - val_loss: 0.0136 - val_categorical_accuracy: 0.9458
Epoch 26/65535
420/420 [==============================] - 49s 118ms/step - loss: 5.8416e-04 - categorical_accuracy: 0.9975 - val_loss: 0.0137 - val_categorical_accuracy: 0.9465
Epoch 27/65535
420/420 [==============================] - 50s 118ms/step - loss: 5.4471e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0139 - val_categorical_accuracy: 0.9458
Epoch 28/65535
420/420 [==============================] - 50s 118ms/step - loss: 4.9655e-04 - categorical_accuracy: 0.9980 - val_loss: 0.0140 - val_categorical_accuracy: 0.9461
Epoch 29/65535
420/420 [==============================] - 49s 118ms/step - loss: 4.5771e-04 - categorical_accuracy: 0.9982 - val_loss: 0.0142 - val_categorical_accuracy: 0.9447

Epoch 00029: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 30/65535
420/420 [==============================] - 50s 118ms/step - loss: 4.1971e-04 - categorical_accuracy: 0.9983 - val_loss: 0.0142 - val_categorical_accuracy: 0.9459
Epoch 31/65535
420/420 [==============================] - 50s 118ms/step - loss: 3.9135e-04 - categorical_accuracy: 0.9984 - val_loss: 0.0142 - val_categorical_accuracy: 0.9452
Epoch 32/65535
420/420 [==============================] - 50s 118ms/step - loss: 3.6684e-04 - categorical_accuracy: 0.9985 - val_loss: 0.0143 - val_categorical_accuracy: 0.9451
Epoch 33/65535
420/420 [==============================] - 50s 118ms/step - loss: 3.6041e-04 - categorical_accuracy: 0.9985 - val_loss: 0.0145 - val_categorical_accuracy: 0.9450
Epoch 00033: early stopping
========= generating oof predictions 18:53:29 =========
========= generating test set predictions 18:53:33 =========
train loss avg 0.00043422925089753013 -- std 9.674552686189426e-05, val loss avg 0.013620210252551012 -- std 0.0005130295323779133
train acc avg 0.9982159403611466 -- std 0.0005774159406490149, val acc avg 0.9469758254774611 -- std 0.0020181128463325772
mean nb epochs 30.8
dump oof predicted probs
dump test set predicted probs
