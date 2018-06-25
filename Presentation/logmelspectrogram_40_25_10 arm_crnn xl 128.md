ren (master *+) python $ python train.py logmelspectrogram_40_25_10 arm_crnn xl 128
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
conv2d_1 (Conv2D)            (None, 46, 37, 168)       6888
_________________________________________________________________
activation_1 (Activation)    (None, 46, 37, 168)       0
_________________________________________________________________
dropout_1 (Dropout)          (None, 46, 37, 168)       0
_________________________________________________________________
permute_1 (Permute)          (None, 46, 168, 37)       0
_________________________________________________________________
reshape_2 (Reshape)          (None, 46, 6216)          0
_________________________________________________________________
cu_dnngru_1 (CuDNNGRU)       (None, 46, 168)           3218544
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
Total params: 3,447,015
Trainable params: 3,447,015
Non-trainable params: 0
_________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 19:03:06 =========
Epoch 1/65535
420/420 [==============================] - 20s 49ms/step - loss: 0.0970 - categorical_accuracy: 0.3516 - val_loss: 0.0419 - val_categorical_accuracy: 0.7663
Epoch 2/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0304 - categorical_accuracy: 0.8298 - val_loss: 0.0205 - val_categorical_accuracy: 0.8852
Epoch 3/65535
420/420 [==============================] - 19s 46ms/step - loss: 0.0192 - categorical_accuracy: 0.8943 - val_loss: 0.0173 - val_categorical_accuracy: 0.9056
Epoch 4/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0150 - categorical_accuracy: 0.9180 - val_loss: 0.0141 - val_categorical_accuracy: 0.9199
Epoch 5/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0127 - categorical_accuracy: 0.9301 - val_loss: 0.0130 - val_categorical_accuracy: 0.9292
Epoch 6/65535
420/420 [==============================] - 19s 46ms/step - loss: 0.0110 - categorical_accuracy: 0.9388 - val_loss: 0.0120 - val_categorical_accuracy: 0.9329
Epoch 7/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0100 - categorical_accuracy: 0.9453 - val_loss: 0.0114 - val_categorical_accuracy: 0.9363
Epoch 8/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0092 - categorical_accuracy: 0.9490 - val_loss: 0.0111 - val_categorical_accuracy: 0.9385
Epoch 9/65535
420/420 [==============================] - 20s 49ms/step - loss: 0.0084 - categorical_accuracy: 0.9539 - val_loss: 0.0107 - val_categorical_accuracy: 0.9390
Epoch 10/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0078 - categorical_accuracy: 0.9553 - val_loss: 0.0098 - val_categorical_accuracy: 0.9469
Epoch 11/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0074 - categorical_accuracy: 0.9585 - val_loss: 0.0101 - val_categorical_accuracy: 0.9442
Epoch 12/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0075 - categorical_accuracy: 0.9571 - val_loss: 0.0104 - val_categorical_accuracy: 0.9427
Epoch 13/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0069 - categorical_accuracy: 0.9612 - val_loss: 0.0098 - val_categorical_accuracy: 0.9475
Epoch 14/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0062 - categorical_accuracy: 0.9647 - val_loss: 0.0092 - val_categorical_accuracy: 0.9494
Epoch 15/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0060 - categorical_accuracy: 0.9659 - val_loss: 0.0097 - val_categorical_accuracy: 0.9478
Epoch 16/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0057 - categorical_accuracy: 0.9678 - val_loss: 0.0094 - val_categorical_accuracy: 0.9511
Epoch 17/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0055 - categorical_accuracy: 0.9689 - val_loss: 0.0097 - val_categorical_accuracy: 0.9489
Epoch 18/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0058 - categorical_accuracy: 0.9667 - val_loss: 0.0092 - val_categorical_accuracy: 0.9507
Epoch 19/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0051 - categorical_accuracy: 0.9712 - val_loss: 0.0094 - val_categorical_accuracy: 0.9514
Epoch 20/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0051 - categorical_accuracy: 0.9705 - val_loss: 0.0087 - val_categorical_accuracy: 0.9547
Epoch 21/65535
420/420 [==============================] - 19s 46ms/step - loss: 0.0052 - categorical_accuracy: 0.9704 - val_loss: 0.0088 - val_categorical_accuracy: 0.9519
Epoch 22/65535
420/420 [==============================] - 20s 46ms/step - loss: 0.0047 - categorical_accuracy: 0.9726 - val_loss: 0.0090 - val_categorical_accuracy: 0.9542
Epoch 23/65535
420/420 [==============================] - 19s 46ms/step - loss: 0.0048 - categorical_accuracy: 0.9725 - val_loss: 0.0092 - val_categorical_accuracy: 0.9530
Epoch 24/65535
420/420 [==============================] - 19s 46ms/step - loss: 0.0041 - categorical_accuracy: 0.9755 - val_loss: 0.0090 - val_categorical_accuracy: 0.9517
Epoch 25/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0043 - categorical_accuracy: 0.9752 - val_loss: 0.0092 - val_categorical_accuracy: 0.9547
Epoch 26/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0047 - categorical_accuracy: 0.9722 - val_loss: 0.0088 - val_categorical_accuracy: 0.9554

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 27/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0032 - categorical_accuracy: 0.9811 - val_loss: 0.0075 - val_categorical_accuracy: 0.9623
Epoch 28/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0027 - categorical_accuracy: 0.9842 - val_loss: 0.0074 - val_categorical_accuracy: 0.9633
Epoch 29/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0024 - categorical_accuracy: 0.9852 - val_loss: 0.0077 - val_categorical_accuracy: 0.9633
Epoch 30/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0023 - categorical_accuracy: 0.9862 - val_loss: 0.0076 - val_categorical_accuracy: 0.9626
Epoch 31/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0022 - categorical_accuracy: 0.9869 - val_loss: 0.0077 - val_categorical_accuracy: 0.9629
Epoch 32/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0022 - categorical_accuracy: 0.9872 - val_loss: 0.0079 - val_categorical_accuracy: 0.9633
Epoch 33/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0021 - categorical_accuracy: 0.9877 - val_loss: 0.0080 - val_categorical_accuracy: 0.9613

Epoch 00033: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 34/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9887 - val_loss: 0.0077 - val_categorical_accuracy: 0.9640
Epoch 35/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9886 - val_loss: 0.0078 - val_categorical_accuracy: 0.9642
Epoch 36/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0018 - categorical_accuracy: 0.9893 - val_loss: 0.0078 - val_categorical_accuracy: 0.9641
Epoch 37/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0018 - categorical_accuracy: 0.9894 - val_loss: 0.0078 - val_categorical_accuracy: 0.9638
Epoch 38/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0018 - categorical_accuracy: 0.9896 - val_loss: 0.0079 - val_categorical_accuracy: 0.9640

Epoch 00038: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 39/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0018 - categorical_accuracy: 0.9895 - val_loss: 0.0078 - val_categorical_accuracy: 0.9639
Epoch 40/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0017 - categorical_accuracy: 0.9897 - val_loss: 0.0078 - val_categorical_accuracy: 0.9641
Epoch 41/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0017 - categorical_accuracy: 0.9896 - val_loss: 0.0079 - val_categorical_accuracy: 0.9644
Epoch 42/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0017 - categorical_accuracy: 0.9896 - val_loss: 0.0079 - val_categorical_accuracy: 0.9641
Epoch 43/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0017 - categorical_accuracy: 0.9897 - val_loss: 0.0079 - val_categorical_accuracy: 0.9641
Epoch 00043: early stopping
========= generating oof predictions 19:17:17 =========
========= generating test set predictions 19:17:18 =========
========= fitting 2 th model 19:17:41 =========
Epoch 1/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0984 - categorical_accuracy: 0.3385 - val_loss: 0.0430 - val_categorical_accuracy: 0.7548
Epoch 2/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0316 - categorical_accuracy: 0.8216 - val_loss: 0.0207 - val_categorical_accuracy: 0.8866
Epoch 3/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0191 - categorical_accuracy: 0.8942 - val_loss: 0.0161 - val_categorical_accuracy: 0.9116
Epoch 4/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0151 - categorical_accuracy: 0.9170 - val_loss: 0.0148 - val_categorical_accuracy: 0.9181
Epoch 5/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0128 - categorical_accuracy: 0.9296 - val_loss: 0.0129 - val_categorical_accuracy: 0.9314
Epoch 6/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0112 - categorical_accuracy: 0.9377 - val_loss: 0.0112 - val_categorical_accuracy: 0.9369
Epoch 7/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0099 - categorical_accuracy: 0.9451 - val_loss: 0.0121 - val_categorical_accuracy: 0.9329
Epoch 8/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0093 - categorical_accuracy: 0.9479 - val_loss: 0.0108 - val_categorical_accuracy: 0.9419
Epoch 9/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0084 - categorical_accuracy: 0.9533 - val_loss: 0.0104 - val_categorical_accuracy: 0.9427
Epoch 10/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0075 - categorical_accuracy: 0.9579 - val_loss: 0.0120 - val_categorical_accuracy: 0.9328
Epoch 11/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0071 - categorical_accuracy: 0.9605 - val_loss: 0.0105 - val_categorical_accuracy: 0.9422
Epoch 12/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0070 - categorical_accuracy: 0.9610 - val_loss: 0.0100 - val_categorical_accuracy: 0.9450
Epoch 13/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0067 - categorical_accuracy: 0.9622 - val_loss: 0.0096 - val_categorical_accuracy: 0.9472
Epoch 14/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0061 - categorical_accuracy: 0.9657 - val_loss: 0.0094 - val_categorical_accuracy: 0.9477
Epoch 15/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0062 - categorical_accuracy: 0.9652 - val_loss: 0.0099 - val_categorical_accuracy: 0.9465
Epoch 16/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0061 - categorical_accuracy: 0.9651 - val_loss: 0.0098 - val_categorical_accuracy: 0.9457
Epoch 17/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0057 - categorical_accuracy: 0.9674 - val_loss: 0.0103 - val_categorical_accuracy: 0.9464
Epoch 18/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0052 - categorical_accuracy: 0.9709 - val_loss: 0.0095 - val_categorical_accuracy: 0.9467
Epoch 19/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0054 - categorical_accuracy: 0.9693 - val_loss: 0.0090 - val_categorical_accuracy: 0.9522
Epoch 20/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0053 - categorical_accuracy: 0.9699 - val_loss: 0.0100 - val_categorical_accuracy: 0.9492
Epoch 21/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0052 - categorical_accuracy: 0.9706 - val_loss: 0.0096 - val_categorical_accuracy: 0.9518
Epoch 22/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0048 - categorical_accuracy: 0.9718 - val_loss: 0.0100 - val_categorical_accuracy: 0.9482
Epoch 23/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0049 - categorical_accuracy: 0.9724 - val_loss: 0.0095 - val_categorical_accuracy: 0.9512
Epoch 24/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0047 - categorical_accuracy: 0.9728 - val_loss: 0.0099 - val_categorical_accuracy: 0.9483
Epoch 25/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0047 - categorical_accuracy: 0.9727 - val_loss: 0.0098 - val_categorical_accuracy: 0.9494

Epoch 00025: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 26/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0033 - categorical_accuracy: 0.9807 - val_loss: 0.0084 - val_categorical_accuracy: 0.9575
Epoch 27/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0028 - categorical_accuracy: 0.9838 - val_loss: 0.0083 - val_categorical_accuracy: 0.9579
Epoch 28/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0026 - categorical_accuracy: 0.9848 - val_loss: 0.0085 - val_categorical_accuracy: 0.9576
Epoch 29/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0025 - categorical_accuracy: 0.9853 - val_loss: 0.0084 - val_categorical_accuracy: 0.9599
Epoch 30/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0024 - categorical_accuracy: 0.9857 - val_loss: 0.0088 - val_categorical_accuracy: 0.9591
Epoch 31/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0023 - categorical_accuracy: 0.9866 - val_loss: 0.0087 - val_categorical_accuracy: 0.9594
Epoch 32/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0022 - categorical_accuracy: 0.9865 - val_loss: 0.0093 - val_categorical_accuracy: 0.9582

Epoch 00032: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 33/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0021 - categorical_accuracy: 0.9878 - val_loss: 0.0087 - val_categorical_accuracy: 0.9591
Epoch 34/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0020 - categorical_accuracy: 0.9879 - val_loss: 0.0086 - val_categorical_accuracy: 0.9600
Epoch 35/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0019 - categorical_accuracy: 0.9885 - val_loss: 0.0086 - val_categorical_accuracy: 0.9604
Epoch 36/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0019 - categorical_accuracy: 0.9885 - val_loss: 0.0087 - val_categorical_accuracy: 0.9606
Epoch 37/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9886 - val_loss: 0.0088 - val_categorical_accuracy: 0.9603

Epoch 00037: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 38/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9891 - val_loss: 0.0087 - val_categorical_accuracy: 0.9604
Epoch 39/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9890 - val_loss: 0.0087 - val_categorical_accuracy: 0.9611
Epoch 40/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0018 - categorical_accuracy: 0.9889 - val_loss: 0.0087 - val_categorical_accuracy: 0.9607
Epoch 41/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0018 - categorical_accuracy: 0.9890 - val_loss: 0.0087 - val_categorical_accuracy: 0.9601
Epoch 42/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0018 - categorical_accuracy: 0.9893 - val_loss: 0.0088 - val_categorical_accuracy: 0.9609
Epoch 00042: early stopping
========= generating oof predictions 19:31:37 =========
========= generating test set predictions 19:31:39 =========
========= fitting 3 th model 19:32:02 =========
Epoch 1/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.1024 - categorical_accuracy: 0.3082 - val_loss: 0.0506 - val_categorical_accuracy: 0.6982
Epoch 2/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0310 - categorical_accuracy: 0.8258 - val_loss: 0.0215 - val_categorical_accuracy: 0.8817
Epoch 3/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0187 - categorical_accuracy: 0.8973 - val_loss: 0.0174 - val_categorical_accuracy: 0.9031
Epoch 4/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0146 - categorical_accuracy: 0.9196 - val_loss: 0.0139 - val_categorical_accuracy: 0.9225
Epoch 5/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0122 - categorical_accuracy: 0.9331 - val_loss: 0.0132 - val_categorical_accuracy: 0.9288
Epoch 6/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0109 - categorical_accuracy: 0.9398 - val_loss: 0.0113 - val_categorical_accuracy: 0.9379
Epoch 7/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0097 - categorical_accuracy: 0.9468 - val_loss: 0.0114 - val_categorical_accuracy: 0.9394
Epoch 8/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0088 - categorical_accuracy: 0.9510 - val_loss: 0.0104 - val_categorical_accuracy: 0.9430
Epoch 9/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0083 - categorical_accuracy: 0.9541 - val_loss: 0.0103 - val_categorical_accuracy: 0.9444
Epoch 10/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0077 - categorical_accuracy: 0.9573 - val_loss: 0.0103 - val_categorical_accuracy: 0.9457
Epoch 11/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0072 - categorical_accuracy: 0.9592 - val_loss: 0.0101 - val_categorical_accuracy: 0.9469
Epoch 12/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0067 - categorical_accuracy: 0.9616 - val_loss: 0.0100 - val_categorical_accuracy: 0.9496
Epoch 13/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0065 - categorical_accuracy: 0.9628 - val_loss: 0.0097 - val_categorical_accuracy: 0.9466
Epoch 14/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0064 - categorical_accuracy: 0.9637 - val_loss: 0.0098 - val_categorical_accuracy: 0.9479
Epoch 15/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0062 - categorical_accuracy: 0.9650 - val_loss: 0.0094 - val_categorical_accuracy: 0.9505
Epoch 16/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0059 - categorical_accuracy: 0.9665 - val_loss: 0.0095 - val_categorical_accuracy: 0.9497
Epoch 17/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0056 - categorical_accuracy: 0.9682 - val_loss: 0.0100 - val_categorical_accuracy: 0.9480
Epoch 18/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0052 - categorical_accuracy: 0.9700 - val_loss: 0.0109 - val_categorical_accuracy: 0.9443
Epoch 19/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0051 - categorical_accuracy: 0.9712 - val_loss: 0.0104 - val_categorical_accuracy: 0.9472
Epoch 20/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0049 - categorical_accuracy: 0.9717 - val_loss: 0.0099 - val_categorical_accuracy: 0.9488
Epoch 21/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0053 - categorical_accuracy: 0.9690 - val_loss: 0.0096 - val_categorical_accuracy: 0.9505

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 22/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0036 - categorical_accuracy: 0.9786 - val_loss: 0.0082 - val_categorical_accuracy: 0.9601
Epoch 23/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0030 - categorical_accuracy: 0.9823 - val_loss: 0.0082 - val_categorical_accuracy: 0.9601
Epoch 24/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0028 - categorical_accuracy: 0.9834 - val_loss: 0.0081 - val_categorical_accuracy: 0.9599
Epoch 25/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0026 - categorical_accuracy: 0.9850 - val_loss: 0.0081 - val_categorical_accuracy: 0.9615
Epoch 26/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0025 - categorical_accuracy: 0.9852 - val_loss: 0.0081 - val_categorical_accuracy: 0.9620
Epoch 27/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0023 - categorical_accuracy: 0.9863 - val_loss: 0.0083 - val_categorical_accuracy: 0.9619
Epoch 28/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0023 - categorical_accuracy: 0.9861 - val_loss: 0.0090 - val_categorical_accuracy: 0.9589
Epoch 29/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0023 - categorical_accuracy: 0.9860 - val_loss: 0.0084 - val_categorical_accuracy: 0.9607
Epoch 30/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0022 - categorical_accuracy: 0.9864 - val_loss: 0.0084 - val_categorical_accuracy: 0.9616
Epoch 31/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0021 - categorical_accuracy: 0.9874 - val_loss: 0.0088 - val_categorical_accuracy: 0.9601

Epoch 00031: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 32/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0020 - categorical_accuracy: 0.9880 - val_loss: 0.0081 - val_categorical_accuracy: 0.9628
Epoch 33/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9886 - val_loss: 0.0081 - val_categorical_accuracy: 0.9630
Epoch 34/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9890 - val_loss: 0.0082 - val_categorical_accuracy: 0.9631
Epoch 35/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9887 - val_loss: 0.0083 - val_categorical_accuracy: 0.9624
Epoch 36/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0018 - categorical_accuracy: 0.9888 - val_loss: 0.0082 - val_categorical_accuracy: 0.9630

Epoch 00036: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 37/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0018 - categorical_accuracy: 0.9893 - val_loss: 0.0082 - val_categorical_accuracy: 0.9636
Epoch 38/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0018 - categorical_accuracy: 0.9891 - val_loss: 0.0083 - val_categorical_accuracy: 0.9636
Epoch 39/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0018 - categorical_accuracy: 0.9896 - val_loss: 0.0083 - val_categorical_accuracy: 0.9633
Epoch 40/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0018 - categorical_accuracy: 0.9893 - val_loss: 0.0083 - val_categorical_accuracy: 0.9632
Epoch 41/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0017 - categorical_accuracy: 0.9897 - val_loss: 0.0083 - val_categorical_accuracy: 0.9633
Epoch 00041: early stopping
========= generating oof predictions 19:45:38 =========
========= generating test set predictions 19:45:40 =========
========= fitting 4 th model 19:46:03 =========
Epoch 1/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.1015 - categorical_accuracy: 0.3202 - val_loss: 0.0427 - val_categorical_accuracy: 0.7567
Epoch 2/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0315 - categorical_accuracy: 0.8214 - val_loss: 0.0225 - val_categorical_accuracy: 0.8776
Epoch 3/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0191 - categorical_accuracy: 0.8936 - val_loss: 0.0147 - val_categorical_accuracy: 0.9193
Epoch 4/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0149 - categorical_accuracy: 0.9182 - val_loss: 0.0122 - val_categorical_accuracy: 0.9328
Epoch 5/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0125 - categorical_accuracy: 0.9314 - val_loss: 0.0122 - val_categorical_accuracy: 0.9362
Epoch 6/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0111 - categorical_accuracy: 0.9372 - val_loss: 0.0111 - val_categorical_accuracy: 0.9373
Epoch 7/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0098 - categorical_accuracy: 0.9450 - val_loss: 0.0111 - val_categorical_accuracy: 0.9398
Epoch 8/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0090 - categorical_accuracy: 0.9497 - val_loss: 0.0099 - val_categorical_accuracy: 0.9461
Epoch 9/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0084 - categorical_accuracy: 0.9532 - val_loss: 0.0102 - val_categorical_accuracy: 0.9419
Epoch 10/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0079 - categorical_accuracy: 0.9546 - val_loss: 0.0099 - val_categorical_accuracy: 0.9480
Epoch 11/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0074 - categorical_accuracy: 0.9579 - val_loss: 0.0100 - val_categorical_accuracy: 0.9454
Epoch 12/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0071 - categorical_accuracy: 0.9601 - val_loss: 0.0095 - val_categorical_accuracy: 0.9486
Epoch 13/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0068 - categorical_accuracy: 0.9609 - val_loss: 0.0098 - val_categorical_accuracy: 0.9452
Epoch 14/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0067 - categorical_accuracy: 0.9622 - val_loss: 0.0102 - val_categorical_accuracy: 0.9462
Epoch 15/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0061 - categorical_accuracy: 0.9651 - val_loss: 0.0089 - val_categorical_accuracy: 0.9521
Epoch 16/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0060 - categorical_accuracy: 0.9648 - val_loss: 0.0089 - val_categorical_accuracy: 0.9520
Epoch 17/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0059 - categorical_accuracy: 0.9660 - val_loss: 0.0084 - val_categorical_accuracy: 0.9560
Epoch 18/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0054 - categorical_accuracy: 0.9686 - val_loss: 0.0088 - val_categorical_accuracy: 0.9518
Epoch 19/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0055 - categorical_accuracy: 0.9683 - val_loss: 0.0094 - val_categorical_accuracy: 0.9497
Epoch 20/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0054 - categorical_accuracy: 0.9686 - val_loss: 0.0094 - val_categorical_accuracy: 0.9498
Epoch 21/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0053 - categorical_accuracy: 0.9692 - val_loss: 0.0087 - val_categorical_accuracy: 0.9533
Epoch 22/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0050 - categorical_accuracy: 0.9712 - val_loss: 0.0097 - val_categorical_accuracy: 0.9519
Epoch 23/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0048 - categorical_accuracy: 0.9727 - val_loss: 0.0089 - val_categorical_accuracy: 0.9535

Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 24/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0035 - categorical_accuracy: 0.9798 - val_loss: 0.0075 - val_categorical_accuracy: 0.9607
Epoch 25/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0029 - categorical_accuracy: 0.9829 - val_loss: 0.0073 - val_categorical_accuracy: 0.9622
Epoch 26/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0028 - categorical_accuracy: 0.9834 - val_loss: 0.0073 - val_categorical_accuracy: 0.9620
Epoch 27/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0026 - categorical_accuracy: 0.9844 - val_loss: 0.0074 - val_categorical_accuracy: 0.9636
Epoch 28/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0025 - categorical_accuracy: 0.9851 - val_loss: 0.0073 - val_categorical_accuracy: 0.9627
Epoch 29/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0025 - categorical_accuracy: 0.9855 - val_loss: 0.0078 - val_categorical_accuracy: 0.9621
Epoch 30/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0024 - categorical_accuracy: 0.9859 - val_loss: 0.0077 - val_categorical_accuracy: 0.9616
Epoch 31/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0023 - categorical_accuracy: 0.9867 - val_loss: 0.0078 - val_categorical_accuracy: 0.9626

Epoch 00031: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 32/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0022 - categorical_accuracy: 0.9875 - val_loss: 0.0076 - val_categorical_accuracy: 0.9633
Epoch 33/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0021 - categorical_accuracy: 0.9877 - val_loss: 0.0076 - val_categorical_accuracy: 0.9630
Epoch 34/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0020 - categorical_accuracy: 0.9879 - val_loss: 0.0076 - val_categorical_accuracy: 0.9630
Epoch 35/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0020 - categorical_accuracy: 0.9880 - val_loss: 0.0076 - val_categorical_accuracy: 0.9638
Epoch 36/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0020 - categorical_accuracy: 0.9881 - val_loss: 0.0076 - val_categorical_accuracy: 0.9636

Epoch 00036: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 37/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0020 - categorical_accuracy: 0.9884 - val_loss: 0.0076 - val_categorical_accuracy: 0.9632
Epoch 38/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9887 - val_loss: 0.0076 - val_categorical_accuracy: 0.9635
Epoch 39/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0019 - categorical_accuracy: 0.9885 - val_loss: 0.0075 - val_categorical_accuracy: 0.9641
Epoch 40/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9888 - val_loss: 0.0077 - val_categorical_accuracy: 0.9639
Epoch 41/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0019 - categorical_accuracy: 0.9888 - val_loss: 0.0076 - val_categorical_accuracy: 0.9635
Epoch 00041: early stopping
========= generating oof predictions 19:59:39 =========
========= generating test set predictions 19:59:41 =========
========= fitting 5 th model 20:00:03 =========
Epoch 1/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.1041 - categorical_accuracy: 0.2999 - val_loss: 0.0521 - val_categorical_accuracy: 0.6881
Epoch 2/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0335 - categorical_accuracy: 0.8107 - val_loss: 0.0234 - val_categorical_accuracy: 0.8675
Epoch 3/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0202 - categorical_accuracy: 0.8879 - val_loss: 0.0180 - val_categorical_accuracy: 0.9002
Epoch 4/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0152 - categorical_accuracy: 0.9161 - val_loss: 0.0177 - val_categorical_accuracy: 0.9028
Epoch 5/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0130 - categorical_accuracy: 0.9290 - val_loss: 0.0133 - val_categorical_accuracy: 0.9257
Epoch 6/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0115 - categorical_accuracy: 0.9363 - val_loss: 0.0121 - val_categorical_accuracy: 0.9356
Epoch 7/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0102 - categorical_accuracy: 0.9436 - val_loss: 0.0111 - val_categorical_accuracy: 0.9390
Epoch 8/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0093 - categorical_accuracy: 0.9476 - val_loss: 0.0098 - val_categorical_accuracy: 0.9451
Epoch 9/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0084 - categorical_accuracy: 0.9535 - val_loss: 0.0108 - val_categorical_accuracy: 0.9427
Epoch 10/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0081 - categorical_accuracy: 0.9535 - val_loss: 0.0100 - val_categorical_accuracy: 0.9453
Epoch 11/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0076 - categorical_accuracy: 0.9566 - val_loss: 0.0100 - val_categorical_accuracy: 0.9465
Epoch 12/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0071 - categorical_accuracy: 0.9590 - val_loss: 0.0103 - val_categorical_accuracy: 0.9451
Epoch 13/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0067 - categorical_accuracy: 0.9607 - val_loss: 0.0110 - val_categorical_accuracy: 0.9423
Epoch 14/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0067 - categorical_accuracy: 0.9615 - val_loss: 0.0113 - val_categorical_accuracy: 0.9389

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 15/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0045 - categorical_accuracy: 0.9742 - val_loss: 0.0079 - val_categorical_accuracy: 0.9590
Epoch 16/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0039 - categorical_accuracy: 0.9778 - val_loss: 0.0079 - val_categorical_accuracy: 0.9582
Epoch 17/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0036 - categorical_accuracy: 0.9788 - val_loss: 0.0079 - val_categorical_accuracy: 0.9593
Epoch 18/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0035 - categorical_accuracy: 0.9798 - val_loss: 0.0081 - val_categorical_accuracy: 0.9574
Epoch 19/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0033 - categorical_accuracy: 0.9808 - val_loss: 0.0081 - val_categorical_accuracy: 0.9589
Epoch 20/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0031 - categorical_accuracy: 0.9814 - val_loss: 0.0083 - val_categorical_accuracy: 0.9584
Epoch 21/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0030 - categorical_accuracy: 0.9823 - val_loss: 0.0083 - val_categorical_accuracy: 0.9594

Epoch 00021: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 22/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0028 - categorical_accuracy: 0.9836 - val_loss: 0.0081 - val_categorical_accuracy: 0.9597
Epoch 23/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0027 - categorical_accuracy: 0.9846 - val_loss: 0.0081 - val_categorical_accuracy: 0.9601
Epoch 24/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0026 - categorical_accuracy: 0.9848 - val_loss: 0.0081 - val_categorical_accuracy: 0.9597
Epoch 25/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0026 - categorical_accuracy: 0.9848 - val_loss: 0.0082 - val_categorical_accuracy: 0.9598
Epoch 26/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0026 - categorical_accuracy: 0.9848 - val_loss: 0.0082 - val_categorical_accuracy: 0.9594

Epoch 00026: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 27/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0025 - categorical_accuracy: 0.9853 - val_loss: 0.0082 - val_categorical_accuracy: 0.9601
Epoch 28/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0025 - categorical_accuracy: 0.9852 - val_loss: 0.0083 - val_categorical_accuracy: 0.9601
Epoch 29/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0025 - categorical_accuracy: 0.9853 - val_loss: 0.0082 - val_categorical_accuracy: 0.9603
Epoch 30/65535
420/420 [==============================] - 20s 48ms/step - loss: 0.0025 - categorical_accuracy: 0.9856 - val_loss: 0.0082 - val_categorical_accuracy: 0.9598
Epoch 31/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0024 - categorical_accuracy: 0.9857 - val_loss: 0.0082 - val_categorical_accuracy: 0.9612
Epoch 32/65535
420/420 [==============================] - 20s 47ms/step - loss: 0.0024 - categorical_accuracy: 0.9859 - val_loss: 0.0083 - val_categorical_accuracy: 0.9608
Epoch 00032: early stopping
========= generating oof predictions 20:10:42 =========
========= generating test set predictions 20:10:44 =========
train loss avg 0.0019199653924672675 -- std 0.00025317227758996975, val loss avg 0.008155929594513282 -- std 0.0003796233015319914
train acc avg 0.988669885936028 -- std 0.0014129525890084842, val acc avg 0.9625151747509968 -- std 0.001390030177049149
mean nb epochs 39.8
dump oof predicted probs
dump test set predicted probs
