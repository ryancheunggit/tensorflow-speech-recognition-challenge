ren (master *+) python $ python train.py logmelspectrogram_40_25_10 arm_crnn l 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (101, 40) ===========
------------- SUMMARY OF MODEL -------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 101, 40)           0
_________________________________________________________________
reshape_1 (Reshape)          (None, 101, 40, 1)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 46, 37, 100)       4100
_________________________________________________________________
activation_1 (Activation)    (None, 46, 37, 100)       0
_________________________________________________________________
dropout_1 (Dropout)          (None, 46, 37, 100)       0
_________________________________________________________________
permute_1 (Permute)          (None, 46, 100, 37)       0
_________________________________________________________________
reshape_2 (Reshape)          (None, 46, 3700)          0
_________________________________________________________________
cu_dnngru_1 (CuDNNGRU)       (None, 46, 136)           1565904
_________________________________________________________________
cu_dnngru_2 (CuDNNGRU)       (None, 136)               111792
_________________________________________________________________
dense_1 (Dense)              (None, 188)               25756
_________________________________________________________________
activation_2 (Activation)    (None, 188)               0
_________________________________________________________________
dropout_2 (Dropout)          (None, 188)               0
_________________________________________________________________
dense_2 (Dense)              (None, 31)                5859
=================================================================
Total params: 1,713,411
Trainable params: 1,713,411
Non-trainable params: 0
_________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 16:53:02 =========
Epoch 1/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0945 - categorical_accuracy: 0.3774 - val_loss: 0.0359 - val_categorical_accuracy: 0.7960
Epoch 2/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0287 - categorical_accuracy: 0.8400 - val_loss: 0.0210 - val_categorical_accuracy: 0.8849
Epoch 3/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0186 - categorical_accuracy: 0.8983 - val_loss: 0.0172 - val_categorical_accuracy: 0.9031
Epoch 4/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0151 - categorical_accuracy: 0.9169 - val_loss: 0.0140 - val_categorical_accuracy: 0.9200
Epoch 5/65535
420/420 [==============================] - 13s 30ms/step - loss: 0.0127 - categorical_accuracy: 0.9298 - val_loss: 0.0121 - val_categorical_accuracy: 0.9299
Epoch 6/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0111 - categorical_accuracy: 0.9389 - val_loss: 0.0127 - val_categorical_accuracy: 0.9275
Epoch 7/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0102 - categorical_accuracy: 0.9422 - val_loss: 0.0114 - val_categorical_accuracy: 0.9370
Epoch 8/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0093 - categorical_accuracy: 0.9476 - val_loss: 0.0113 - val_categorical_accuracy: 0.9378
Epoch 9/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0088 - categorical_accuracy: 0.9496 - val_loss: 0.0107 - val_categorical_accuracy: 0.9431
Epoch 10/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0083 - categorical_accuracy: 0.9537 - val_loss: 0.0105 - val_categorical_accuracy: 0.9435
Epoch 11/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0078 - categorical_accuracy: 0.9564 - val_loss: 0.0112 - val_categorical_accuracy: 0.9388
Epoch 12/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0073 - categorical_accuracy: 0.9581 - val_loss: 0.0103 - val_categorical_accuracy: 0.9428
Epoch 13/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0068 - categorical_accuracy: 0.9611 - val_loss: 0.0114 - val_categorical_accuracy: 0.9371
Epoch 14/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0068 - categorical_accuracy: 0.9615 - val_loss: 0.0097 - val_categorical_accuracy: 0.9481
Epoch 15/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0062 - categorical_accuracy: 0.9643 - val_loss: 0.0094 - val_categorical_accuracy: 0.9502
Epoch 16/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0065 - categorical_accuracy: 0.9629 - val_loss: 0.0095 - val_categorical_accuracy: 0.9483
Epoch 17/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0061 - categorical_accuracy: 0.9651 - val_loss: 0.0097 - val_categorical_accuracy: 0.9480
Epoch 18/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0059 - categorical_accuracy: 0.9662 - val_loss: 0.0100 - val_categorical_accuracy: 0.9461
Epoch 19/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0058 - categorical_accuracy: 0.9657 - val_loss: 0.0094 - val_categorical_accuracy: 0.9481
Epoch 20/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0051 - categorical_accuracy: 0.9700 - val_loss: 0.0096 - val_categorical_accuracy: 0.9492
Epoch 21/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0054 - categorical_accuracy: 0.9693 - val_loss: 0.0106 - val_categorical_accuracy: 0.9464

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 22/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0039 - categorical_accuracy: 0.9769 - val_loss: 0.0080 - val_categorical_accuracy: 0.9580
Epoch 23/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0033 - categorical_accuracy: 0.9805 - val_loss: 0.0082 - val_categorical_accuracy: 0.9585
Epoch 24/65535
420/420 [==============================] - 14s 33ms/step - loss: 0.0031 - categorical_accuracy: 0.9822 - val_loss: 0.0080 - val_categorical_accuracy: 0.9595
Epoch 25/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0030 - categorical_accuracy: 0.9824 - val_loss: 0.0081 - val_categorical_accuracy: 0.9594
Epoch 26/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0029 - categorical_accuracy: 0.9833 - val_loss: 0.0082 - val_categorical_accuracy: 0.9590
Epoch 27/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0027 - categorical_accuracy: 0.9838 - val_loss: 0.0083 - val_categorical_accuracy: 0.9603
Epoch 28/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0027 - categorical_accuracy: 0.9842 - val_loss: 0.0085 - val_categorical_accuracy: 0.9597

Epoch 00028: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 29/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0025 - categorical_accuracy: 0.9857 - val_loss: 0.0083 - val_categorical_accuracy: 0.9609
Epoch 30/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0024 - categorical_accuracy: 0.9861 - val_loss: 0.0083 - val_categorical_accuracy: 0.9606
Epoch 31/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0024 - categorical_accuracy: 0.9862 - val_loss: 0.0083 - val_categorical_accuracy: 0.9609
Epoch 32/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0023 - categorical_accuracy: 0.9862 - val_loss: 0.0083 - val_categorical_accuracy: 0.9606
Epoch 33/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0023 - categorical_accuracy: 0.9864 - val_loss: 0.0083 - val_categorical_accuracy: 0.9606

Epoch 00033: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 34/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0023 - categorical_accuracy: 0.9866 - val_loss: 0.0083 - val_categorical_accuracy: 0.9606
Epoch 35/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0022 - categorical_accuracy: 0.9867 - val_loss: 0.0083 - val_categorical_accuracy: 0.9611
Epoch 36/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0022 - categorical_accuracy: 0.9865 - val_loss: 0.0083 - val_categorical_accuracy: 0.9612
Epoch 37/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0022 - categorical_accuracy: 0.9873 - val_loss: 0.0083 - val_categorical_accuracy: 0.9612
Epoch 00037: early stopping
========= generating oof predictions 17:01:13 =========
========= generating test set predictions 17:01:14 =========
========= fitting 2 th model 17:01:31 =========
Epoch 1/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0951 - categorical_accuracy: 0.3715 - val_loss: 0.0360 - val_categorical_accuracy: 0.8000
Epoch 2/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0298 - categorical_accuracy: 0.8350 - val_loss: 0.0222 - val_categorical_accuracy: 0.8767
Epoch 3/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0193 - categorical_accuracy: 0.8951 - val_loss: 0.0163 - val_categorical_accuracy: 0.9090
Epoch 4/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0150 - categorical_accuracy: 0.9185 - val_loss: 0.0141 - val_categorical_accuracy: 0.9219
Epoch 5/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0127 - categorical_accuracy: 0.9306 - val_loss: 0.0131 - val_categorical_accuracy: 0.9242
Epoch 6/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0112 - categorical_accuracy: 0.9374 - val_loss: 0.0119 - val_categorical_accuracy: 0.9323
Epoch 7/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0099 - categorical_accuracy: 0.9462 - val_loss: 0.0108 - val_categorical_accuracy: 0.9397
Epoch 8/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0091 - categorical_accuracy: 0.9495 - val_loss: 0.0108 - val_categorical_accuracy: 0.9383
Epoch 9/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0085 - categorical_accuracy: 0.9530 - val_loss: 0.0111 - val_categorical_accuracy: 0.9401
Epoch 10/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0079 - categorical_accuracy: 0.9565 - val_loss: 0.0101 - val_categorical_accuracy: 0.9443
Epoch 11/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0076 - categorical_accuracy: 0.9570 - val_loss: 0.0093 - val_categorical_accuracy: 0.9485
Epoch 12/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0071 - categorical_accuracy: 0.9599 - val_loss: 0.0120 - val_categorical_accuracy: 0.9346
Epoch 13/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0063 - categorical_accuracy: 0.9644 - val_loss: 0.0096 - val_categorical_accuracy: 0.9492
Epoch 14/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0062 - categorical_accuracy: 0.9652 - val_loss: 0.0095 - val_categorical_accuracy: 0.9469
Epoch 15/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0060 - categorical_accuracy: 0.9665 - val_loss: 0.0092 - val_categorical_accuracy: 0.9489
Epoch 16/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0057 - categorical_accuracy: 0.9674 - val_loss: 0.0097 - val_categorical_accuracy: 0.9488
Epoch 17/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0056 - categorical_accuracy: 0.9683 - val_loss: 0.0096 - val_categorical_accuracy: 0.9497
Epoch 18/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0056 - categorical_accuracy: 0.9686 - val_loss: 0.0099 - val_categorical_accuracy: 0.9479
Epoch 19/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0051 - categorical_accuracy: 0.9709 - val_loss: 0.0098 - val_categorical_accuracy: 0.9505
Epoch 20/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0052 - categorical_accuracy: 0.9705 - val_loss: 0.0093 - val_categorical_accuracy: 0.9507
Epoch 21/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0047 - categorical_accuracy: 0.9728 - val_loss: 0.0101 - val_categorical_accuracy: 0.9482

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 22/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0036 - categorical_accuracy: 0.9797 - val_loss: 0.0082 - val_categorical_accuracy: 0.9585
Epoch 23/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0031 - categorical_accuracy: 0.9818 - val_loss: 0.0080 - val_categorical_accuracy: 0.9590
Epoch 24/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0029 - categorical_accuracy: 0.9834 - val_loss: 0.0078 - val_categorical_accuracy: 0.9604
Epoch 25/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0027 - categorical_accuracy: 0.9842 - val_loss: 0.0080 - val_categorical_accuracy: 0.9605
Epoch 26/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0027 - categorical_accuracy: 0.9846 - val_loss: 0.0079 - val_categorical_accuracy: 0.9610
Epoch 27/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0026 - categorical_accuracy: 0.9851 - val_loss: 0.0081 - val_categorical_accuracy: 0.9610
Epoch 28/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0025 - categorical_accuracy: 0.9857 - val_loss: 0.0083 - val_categorical_accuracy: 0.9599
Epoch 29/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0024 - categorical_accuracy: 0.9858 - val_loss: 0.0084 - val_categorical_accuracy: 0.9607
Epoch 30/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0024 - categorical_accuracy: 0.9861 - val_loss: 0.0087 - val_categorical_accuracy: 0.9601

Epoch 00030: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 31/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0022 - categorical_accuracy: 0.9869 - val_loss: 0.0082 - val_categorical_accuracy: 0.9622
Epoch 32/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0021 - categorical_accuracy: 0.9874 - val_loss: 0.0081 - val_categorical_accuracy: 0.9620
Epoch 33/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0021 - categorical_accuracy: 0.9879 - val_loss: 0.0081 - val_categorical_accuracy: 0.9626
Epoch 34/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0021 - categorical_accuracy: 0.9877 - val_loss: 0.0082 - val_categorical_accuracy: 0.9618
Epoch 35/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0021 - categorical_accuracy: 0.9875 - val_loss: 0.0082 - val_categorical_accuracy: 0.9620

Epoch 00035: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 36/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0020 - categorical_accuracy: 0.9882 - val_loss: 0.0082 - val_categorical_accuracy: 0.9623
Epoch 37/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0020 - categorical_accuracy: 0.9882 - val_loss: 0.0081 - val_categorical_accuracy: 0.9625
Epoch 38/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0020 - categorical_accuracy: 0.9880 - val_loss: 0.0081 - val_categorical_accuracy: 0.9626
Epoch 39/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0020 - categorical_accuracy: 0.9883 - val_loss: 0.0081 - val_categorical_accuracy: 0.9624
Epoch 00039: early stopping
========= generating oof predictions 17:10:08 =========
========= generating test set predictions 17:10:09 =========
========= fitting 3 th model 17:10:26 =========
Epoch 1/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0845 - categorical_accuracy: 0.4540 - val_loss: 0.0317 - val_categorical_accuracy: 0.8190
Epoch 2/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0265 - categorical_accuracy: 0.8547 - val_loss: 0.0192 - val_categorical_accuracy: 0.8934
Epoch 3/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0180 - categorical_accuracy: 0.9017 - val_loss: 0.0155 - val_categorical_accuracy: 0.9161
Epoch 4/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0145 - categorical_accuracy: 0.9215 - val_loss: 0.0141 - val_categorical_accuracy: 0.9220
Epoch 5/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0119 - categorical_accuracy: 0.9346 - val_loss: 0.0117 - val_categorical_accuracy: 0.9356
Epoch 6/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0104 - categorical_accuracy: 0.9426 - val_loss: 0.0106 - val_categorical_accuracy: 0.9423
Epoch 7/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0094 - categorical_accuracy: 0.9479 - val_loss: 0.0106 - val_categorical_accuracy: 0.9411
Epoch 8/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0085 - categorical_accuracy: 0.9526 - val_loss: 0.0110 - val_categorical_accuracy: 0.9432
Epoch 9/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0081 - categorical_accuracy: 0.9546 - val_loss: 0.0109 - val_categorical_accuracy: 0.9408
Epoch 10/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0075 - categorical_accuracy: 0.9581 - val_loss: 0.0103 - val_categorical_accuracy: 0.9449
Epoch 11/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0072 - categorical_accuracy: 0.9589 - val_loss: 0.0098 - val_categorical_accuracy: 0.9487
Epoch 12/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0069 - categorical_accuracy: 0.9608 - val_loss: 0.0107 - val_categorical_accuracy: 0.9432
Epoch 13/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0066 - categorical_accuracy: 0.9624 - val_loss: 0.0092 - val_categorical_accuracy: 0.9492
Epoch 14/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0062 - categorical_accuracy: 0.9646 - val_loss: 0.0113 - val_categorical_accuracy: 0.9406
Epoch 15/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0059 - categorical_accuracy: 0.9667 - val_loss: 0.0099 - val_categorical_accuracy: 0.9490
Epoch 16/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0055 - categorical_accuracy: 0.9682 - val_loss: 0.0094 - val_categorical_accuracy: 0.9516
Epoch 17/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0057 - categorical_accuracy: 0.9674 - val_loss: 0.0092 - val_categorical_accuracy: 0.9519
Epoch 18/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0051 - categorical_accuracy: 0.9703 - val_loss: 0.0104 - val_categorical_accuracy: 0.9459
Epoch 19/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0052 - categorical_accuracy: 0.9697 - val_loss: 0.0124 - val_categorical_accuracy: 0.9414

Epoch 00019: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 20/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0037 - categorical_accuracy: 0.9782 - val_loss: 0.0077 - val_categorical_accuracy: 0.9591
Epoch 21/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0031 - categorical_accuracy: 0.9820 - val_loss: 0.0079 - val_categorical_accuracy: 0.9582
Epoch 22/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0029 - categorical_accuracy: 0.9831 - val_loss: 0.0079 - val_categorical_accuracy: 0.9604
Epoch 23/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0028 - categorical_accuracy: 0.9835 - val_loss: 0.0080 - val_categorical_accuracy: 0.9598
Epoch 24/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0027 - categorical_accuracy: 0.9842 - val_loss: 0.0081 - val_categorical_accuracy: 0.9592
Epoch 25/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0026 - categorical_accuracy: 0.9848 - val_loss: 0.0082 - val_categorical_accuracy: 0.9590
Epoch 26/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0025 - categorical_accuracy: 0.9850 - val_loss: 0.0083 - val_categorical_accuracy: 0.9595

Epoch 00026: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 27/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0023 - categorical_accuracy: 0.9866 - val_loss: 0.0081 - val_categorical_accuracy: 0.9615
Epoch 28/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0022 - categorical_accuracy: 0.9867 - val_loss: 0.0080 - val_categorical_accuracy: 0.9616
Epoch 29/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0022 - categorical_accuracy: 0.9868 - val_loss: 0.0079 - val_categorical_accuracy: 0.9624
Epoch 30/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0021 - categorical_accuracy: 0.9874 - val_loss: 0.0082 - val_categorical_accuracy: 0.9622
Epoch 31/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0021 - categorical_accuracy: 0.9871 - val_loss: 0.0081 - val_categorical_accuracy: 0.9627

Epoch 00031: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 32/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0021 - categorical_accuracy: 0.9872 - val_loss: 0.0081 - val_categorical_accuracy: 0.9631
Epoch 33/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0021 - categorical_accuracy: 0.9877 - val_loss: 0.0081 - val_categorical_accuracy: 0.9631
Epoch 34/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0021 - categorical_accuracy: 0.9879 - val_loss: 0.0081 - val_categorical_accuracy: 0.9627
Epoch 35/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0021 - categorical_accuracy: 0.9876 - val_loss: 0.0082 - val_categorical_accuracy: 0.9629
Epoch 00035: early stopping
========= generating oof predictions 17:18:09 =========
========= generating test set predictions 17:18:10 =========
========= fitting 4 th model 17:18:27 =========
Epoch 1/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0983 - categorical_accuracy: 0.3449 - val_loss: 0.0391 - val_categorical_accuracy: 0.7807
Epoch 2/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0290 - categorical_accuracy: 0.8391 - val_loss: 0.0189 - val_categorical_accuracy: 0.8962
Epoch 3/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0187 - categorical_accuracy: 0.8978 - val_loss: 0.0153 - val_categorical_accuracy: 0.9159
Epoch 4/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0149 - categorical_accuracy: 0.9177 - val_loss: 0.0133 - val_categorical_accuracy: 0.9275
Epoch 5/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0130 - categorical_accuracy: 0.9283 - val_loss: 0.0126 - val_categorical_accuracy: 0.9308
Epoch 6/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0110 - categorical_accuracy: 0.9383 - val_loss: 0.0111 - val_categorical_accuracy: 0.9415
Epoch 7/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0102 - categorical_accuracy: 0.9430 - val_loss: 0.0121 - val_categorical_accuracy: 0.9329
Epoch 8/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0090 - categorical_accuracy: 0.9495 - val_loss: 0.0103 - val_categorical_accuracy: 0.9440
Epoch 9/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0084 - categorical_accuracy: 0.9530 - val_loss: 0.0106 - val_categorical_accuracy: 0.9408
Epoch 10/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0076 - categorical_accuracy: 0.9571 - val_loss: 0.0101 - val_categorical_accuracy: 0.9453
Epoch 11/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0074 - categorical_accuracy: 0.9589 - val_loss: 0.0096 - val_categorical_accuracy: 0.9483
Epoch 12/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0069 - categorical_accuracy: 0.9614 - val_loss: 0.0098 - val_categorical_accuracy: 0.9466
Epoch 13/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0064 - categorical_accuracy: 0.9647 - val_loss: 0.0095 - val_categorical_accuracy: 0.9479
Epoch 14/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0063 - categorical_accuracy: 0.9643 - val_loss: 0.0093 - val_categorical_accuracy: 0.9515
Epoch 15/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0058 - categorical_accuracy: 0.9666 - val_loss: 0.0104 - val_categorical_accuracy: 0.9440
Epoch 16/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0056 - categorical_accuracy: 0.9679 - val_loss: 0.0108 - val_categorical_accuracy: 0.9446
Epoch 17/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0056 - categorical_accuracy: 0.9678 - val_loss: 0.0095 - val_categorical_accuracy: 0.9503
Epoch 18/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0055 - categorical_accuracy: 0.9690 - val_loss: 0.0085 - val_categorical_accuracy: 0.9533
Epoch 19/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0050 - categorical_accuracy: 0.9711 - val_loss: 0.0091 - val_categorical_accuracy: 0.9527
Epoch 20/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0050 - categorical_accuracy: 0.9716 - val_loss: 0.0099 - val_categorical_accuracy: 0.9469
Epoch 21/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0048 - categorical_accuracy: 0.9724 - val_loss: 0.0096 - val_categorical_accuracy: 0.9526
Epoch 22/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0048 - categorical_accuracy: 0.9720 - val_loss: 0.0088 - val_categorical_accuracy: 0.9536
Epoch 23/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0047 - categorical_accuracy: 0.9719 - val_loss: 0.0090 - val_categorical_accuracy: 0.9558
Epoch 24/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0047 - categorical_accuracy: 0.9729 - val_loss: 0.0093 - val_categorical_accuracy: 0.9541

Epoch 00024: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 25/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0033 - categorical_accuracy: 0.9806 - val_loss: 0.0077 - val_categorical_accuracy: 0.9612
Epoch 26/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0028 - categorical_accuracy: 0.9836 - val_loss: 0.0076 - val_categorical_accuracy: 0.9621
Epoch 27/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0027 - categorical_accuracy: 0.9842 - val_loss: 0.0077 - val_categorical_accuracy: 0.9624
Epoch 28/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0025 - categorical_accuracy: 0.9850 - val_loss: 0.0077 - val_categorical_accuracy: 0.9623
Epoch 29/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0025 - categorical_accuracy: 0.9854 - val_loss: 0.0079 - val_categorical_accuracy: 0.9608
Epoch 30/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0024 - categorical_accuracy: 0.9858 - val_loss: 0.0082 - val_categorical_accuracy: 0.9609
Epoch 31/65535
420/420 [==============================] - 14s 33ms/step - loss: 0.0023 - categorical_accuracy: 0.9860 - val_loss: 0.0079 - val_categorical_accuracy: 0.9611
Epoch 32/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0023 - categorical_accuracy: 0.9862 - val_loss: 0.0080 - val_categorical_accuracy: 0.9627

Epoch 00032: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 33/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0021 - categorical_accuracy: 0.9875 - val_loss: 0.0077 - val_categorical_accuracy: 0.9630
Epoch 34/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0021 - categorical_accuracy: 0.9874 - val_loss: 0.0078 - val_categorical_accuracy: 0.9626
Epoch 35/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0020 - categorical_accuracy: 0.9877 - val_loss: 0.0078 - val_categorical_accuracy: 0.9626
Epoch 36/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0020 - categorical_accuracy: 0.9879 - val_loss: 0.0077 - val_categorical_accuracy: 0.9627
Epoch 37/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0020 - categorical_accuracy: 0.9882 - val_loss: 0.0079 - val_categorical_accuracy: 0.9629

Epoch 00037: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 38/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0019 - categorical_accuracy: 0.9882 - val_loss: 0.0078 - val_categorical_accuracy: 0.9633
Epoch 39/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0019 - categorical_accuracy: 0.9883 - val_loss: 0.0078 - val_categorical_accuracy: 0.9635
Epoch 40/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0019 - categorical_accuracy: 0.9884 - val_loss: 0.0078 - val_categorical_accuracy: 0.9632
Epoch 41/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0019 - categorical_accuracy: 0.9888 - val_loss: 0.0079 - val_categorical_accuracy: 0.9634
Epoch 00041: early stopping
========= generating oof predictions 17:27:34 =========
========= generating test set predictions 17:27:35 =========
========= fitting 5 th model 17:27:52 =========
Epoch 1/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.1051 - categorical_accuracy: 0.2970 - val_loss: 0.0427 - val_categorical_accuracy: 0.7570
Epoch 2/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0299 - categorical_accuracy: 0.8344 - val_loss: 0.0202 - val_categorical_accuracy: 0.8884
Epoch 3/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0188 - categorical_accuracy: 0.8982 - val_loss: 0.0162 - val_categorical_accuracy: 0.9105
Epoch 4/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0145 - categorical_accuracy: 0.9214 - val_loss: 0.0138 - val_categorical_accuracy: 0.9239
Epoch 5/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0125 - categorical_accuracy: 0.9328 - val_loss: 0.0127 - val_categorical_accuracy: 0.9300
Epoch 6/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0108 - categorical_accuracy: 0.9402 - val_loss: 0.0128 - val_categorical_accuracy: 0.9291
Epoch 7/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0095 - categorical_accuracy: 0.9484 - val_loss: 0.0119 - val_categorical_accuracy: 0.9332
Epoch 8/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0088 - categorical_accuracy: 0.9513 - val_loss: 0.0109 - val_categorical_accuracy: 0.9411
Epoch 9/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0080 - categorical_accuracy: 0.9549 - val_loss: 0.0105 - val_categorical_accuracy: 0.9425
Epoch 10/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0075 - categorical_accuracy: 0.9582 - val_loss: 0.0101 - val_categorical_accuracy: 0.9452
Epoch 11/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0070 - categorical_accuracy: 0.9605 - val_loss: 0.0105 - val_categorical_accuracy: 0.9413
Epoch 12/65535
420/420 [==============================] - 14s 33ms/step - loss: 0.0068 - categorical_accuracy: 0.9619 - val_loss: 0.0100 - val_categorical_accuracy: 0.9457
Epoch 13/65535
420/420 [==============================] - 14s 33ms/step - loss: 0.0063 - categorical_accuracy: 0.9639 - val_loss: 0.0096 - val_categorical_accuracy: 0.9473
Epoch 14/65535
420/420 [==============================] - 14s 33ms/step - loss: 0.0061 - categorical_accuracy: 0.9654 - val_loss: 0.0099 - val_categorical_accuracy: 0.9469
Epoch 15/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0060 - categorical_accuracy: 0.9662 - val_loss: 0.0103 - val_categorical_accuracy: 0.9455
Epoch 16/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0058 - categorical_accuracy: 0.9670 - val_loss: 0.0097 - val_categorical_accuracy: 0.9480
Epoch 17/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0052 - categorical_accuracy: 0.9707 - val_loss: 0.0093 - val_categorical_accuracy: 0.9526
Epoch 18/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0053 - categorical_accuracy: 0.9690 - val_loss: 0.0103 - val_categorical_accuracy: 0.9473
Epoch 19/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0051 - categorical_accuracy: 0.9703 - val_loss: 0.0097 - val_categorical_accuracy: 0.9499
Epoch 20/65535
420/420 [==============================] - 14s 33ms/step - loss: 0.0049 - categorical_accuracy: 0.9720 - val_loss: 0.0098 - val_categorical_accuracy: 0.9500
Epoch 21/65535
420/420 [==============================] - 14s 32ms/step - loss: 0.0050 - categorical_accuracy: 0.9718 - val_loss: 0.0099 - val_categorical_accuracy: 0.9487
Epoch 22/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0048 - categorical_accuracy: 0.9723 - val_loss: 0.0111 - val_categorical_accuracy: 0.9443
Epoch 23/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0047 - categorical_accuracy: 0.9730 - val_loss: 0.0094 - val_categorical_accuracy: 0.9532

Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 24/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0033 - categorical_accuracy: 0.9810 - val_loss: 0.0083 - val_categorical_accuracy: 0.9597
Epoch 25/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0028 - categorical_accuracy: 0.9832 - val_loss: 0.0082 - val_categorical_accuracy: 0.9615
Epoch 26/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0026 - categorical_accuracy: 0.9845 - val_loss: 0.0082 - val_categorical_accuracy: 0.9609
Epoch 27/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0025 - categorical_accuracy: 0.9850 - val_loss: 0.0082 - val_categorical_accuracy: 0.9618
Epoch 28/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0025 - categorical_accuracy: 0.9855 - val_loss: 0.0084 - val_categorical_accuracy: 0.9611
Epoch 29/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0023 - categorical_accuracy: 0.9857 - val_loss: 0.0086 - val_categorical_accuracy: 0.9609
Epoch 30/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0023 - categorical_accuracy: 0.9863 - val_loss: 0.0088 - val_categorical_accuracy: 0.9596
Epoch 31/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0023 - categorical_accuracy: 0.9866 - val_loss: 0.0087 - val_categorical_accuracy: 0.9623

Epoch 00031: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 32/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0021 - categorical_accuracy: 0.9876 - val_loss: 0.0084 - val_categorical_accuracy: 0.9629
Epoch 33/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0020 - categorical_accuracy: 0.9877 - val_loss: 0.0086 - val_categorical_accuracy: 0.9628
Epoch 34/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0020 - categorical_accuracy: 0.9880 - val_loss: 0.0085 - val_categorical_accuracy: 0.9625
Epoch 35/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0020 - categorical_accuracy: 0.9883 - val_loss: 0.0086 - val_categorical_accuracy: 0.9625
Epoch 36/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0019 - categorical_accuracy: 0.9883 - val_loss: 0.0086 - val_categorical_accuracy: 0.9625

Epoch 00036: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 37/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0019 - categorical_accuracy: 0.9881 - val_loss: 0.0086 - val_categorical_accuracy: 0.9627
Epoch 38/65535
420/420 [==============================] - 13s 32ms/step - loss: 0.0019 - categorical_accuracy: 0.9885 - val_loss: 0.0086 - val_categorical_accuracy: 0.9626
Epoch 39/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0019 - categorical_accuracy: 0.9887 - val_loss: 0.0086 - val_categorical_accuracy: 0.9629
Epoch 40/65535
420/420 [==============================] - 13s 31ms/step - loss: 0.0019 - categorical_accuracy: 0.9887 - val_loss: 0.0086 - val_categorical_accuracy: 0.9622
Epoch 00040: early stopping
========= generating oof predictions 17:36:48 =========
========= generating test set predictions 17:36:49 =========
train loss avg 0.002007319077793154 -- std 0.00012003499227714494, val loss avg 0.00823542155965848 -- std 0.00024093997030450118
train acc avg 0.9881370198840214 -- std 0.0006092937921917877, val acc avg 0.9624263805777111 -- std 0.0007306733342651067
mean nb epochs 38.4
dump oof predicted probs
dump test set predicted probs
