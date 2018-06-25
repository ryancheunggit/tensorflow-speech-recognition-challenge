ren (master *+) python $ python train.py mfcc_10_40_20 arm_dscnn xl 128

/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (51, 10) ===========
^[[C^[[C^[[C^[[C^[[C^[[C^[[C------------- SUMMARY OF MODEL -------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 51, 10)            0
_________________________________________________________________
reshape_1 (Reshape)          (None, 51, 10, 1)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 26, 10, 384)       15744
_________________________________________________________________
batch_normalization_1 (Batch (None, 26, 10, 384)       1536
_________________________________________________________________
activation_1 (Activation)    (None, 26, 10, 384)       0
_________________________________________________________________
dropout_1 (Dropout)          (None, 26, 10, 384)       0
_________________________________________________________________
separable_conv2d_1 (Separabl (None, 26, 10, 384)       151296
_________________________________________________________________
batch_normalization_2 (Batch (None, 26, 10, 384)       1536
_________________________________________________________________
activation_2 (Activation)    (None, 26, 10, 384)       0
_________________________________________________________________
dropout_2 (Dropout)          (None, 26, 10, 384)       0
_________________________________________________________________
separable_conv2d_2 (Separabl (None, 26, 10, 384)       151296
_________________________________________________________________
batch_normalization_3 (Batch (None, 26, 10, 384)       1536
_________________________________________________________________
activation_3 (Activation)    (None, 26, 10, 384)       0
_________________________________________________________________
dropout_3 (Dropout)          (None, 26, 10, 384)       0
_________________________________________________________________
separable_conv2d_3 (Separabl (None, 26, 10, 384)       151296
_________________________________________________________________
batch_normalization_4 (Batch (None, 26, 10, 384)       1536
_________________________________________________________________
activation_4 (Activation)    (None, 26, 10, 384)       0
_________________________________________________________________
dropout_4 (Dropout)          (None, 26, 10, 384)       0
_________________________________________________________________
separable_conv2d_4 (Separabl (None, 26, 10, 384)       151296
_________________________________________________________________
batch_normalization_5 (Batch (None, 26, 10, 384)       1536
_________________________________________________________________
activation_5 (Activation)    (None, 26, 10, 384)       0
_________________________________________________________________
dropout_5 (Dropout)          (None, 26, 10, 384)       0
_________________________________________________________________
global_average_pooling2d_1 ( (None, 384)               0
_________________________________________________________________
dense_1 (Dense)              (None, 31)                11935
=================================================================
Total params: 640,543
Trainable params: 636,703
Non-trainable params: 3,840
_________________________________________________________________
None
^[[C--------------------------------------------
========= fitting 1 th model 12:33:30 =========
Epoch 1/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0619 - categorical_accuracy: 0.6626 - val_loss: 0.0330 - val_categorical_accuracy: 0.8120
Epoch 2/65535
420/420 [==============================] - 33s 78ms/step - loss: 0.0209 - categorical_accuracy: 0.8930 - val_loss: 0.0230 - val_categorical_accuracy: 0.8680
Epoch 3/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0159 - categorical_accuracy: 0.9153 - val_loss: 0.0156 - val_categorical_accuracy: 0.9118
Epoch 4/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0134 - categorical_accuracy: 0.9281 - val_loss: 0.0177 - val_categorical_accuracy: 0.9044
Epoch 5/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0119 - categorical_accuracy: 0.9353 - val_loss: 0.0197 - val_categorical_accuracy: 0.8895
Epoch 6/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0108 - categorical_accuracy: 0.9414 - val_loss: 0.0141 - val_categorical_accuracy: 0.9214
Epoch 7/65535
420/420 [==============================] - 34s 82ms/step - loss: 0.0098 - categorical_accuracy: 0.9459 - val_loss: 0.0147 - val_categorical_accuracy: 0.9216
Epoch 8/65535
420/420 [==============================] - 34s 82ms/step - loss: 0.0090 - categorical_accuracy: 0.9492 - val_loss: 0.0127 - val_categorical_accuracy: 0.9316
Epoch 9/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0086 - categorical_accuracy: 0.9521 - val_loss: 0.0118 - val_categorical_accuracy: 0.9357
Epoch 10/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0079 - categorical_accuracy: 0.9562 - val_loss: 0.0135 - val_categorical_accuracy: 0.9272
Epoch 11/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0075 - categorical_accuracy: 0.9588 - val_loss: 0.0128 - val_categorical_accuracy: 0.9323
Epoch 12/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0070 - categorical_accuracy: 0.9611 - val_loss: 0.0154 - val_categorical_accuracy: 0.9201
Epoch 13/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0068 - categorical_accuracy: 0.9617 - val_loss: 0.0106 - val_categorical_accuracy: 0.9429
Epoch 14/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0064 - categorical_accuracy: 0.9649 - val_loss: 0.0108 - val_categorical_accuracy: 0.9420
Epoch 15/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0061 - categorical_accuracy: 0.9659 - val_loss: 0.0111 - val_categorical_accuracy: 0.9381
Epoch 16/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0058 - categorical_accuracy: 0.9675 - val_loss: 0.0099 - val_categorical_accuracy: 0.9451
Epoch 17/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0056 - categorical_accuracy: 0.9685 - val_loss: 0.0099 - val_categorical_accuracy: 0.9468
Epoch 18/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0053 - categorical_accuracy: 0.9697 - val_loss: 0.0095 - val_categorical_accuracy: 0.9478
Epoch 19/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0051 - categorical_accuracy: 0.9713 - val_loss: 0.0090 - val_categorical_accuracy: 0.9518
Epoch 20/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0048 - categorical_accuracy: 0.9730 - val_loss: 0.0102 - val_categorical_accuracy: 0.9435
Epoch 21/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0047 - categorical_accuracy: 0.9742 - val_loss: 0.0112 - val_categorical_accuracy: 0.9405
Epoch 22/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0044 - categorical_accuracy: 0.9754 - val_loss: 0.0111 - val_categorical_accuracy: 0.9446
Epoch 23/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0043 - categorical_accuracy: 0.9759 - val_loss: 0.0121 - val_categorical_accuracy: 0.9395
Epoch 24/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0042 - categorical_accuracy: 0.9766 - val_loss: 0.0103 - val_categorical_accuracy: 0.9472
Epoch 25/65535
420/420 [==============================] - 33s 79ms/step - loss: 0.0040 - categorical_accuracy: 0.9772 - val_loss: 0.0112 - val_categorical_accuracy: 0.9435

Epoch 00025: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 26/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0030 - categorical_accuracy: 0.9837 - val_loss: 0.0071 - val_categorical_accuracy: 0.9630
Epoch 27/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0026 - categorical_accuracy: 0.9859 - val_loss: 0.0072 - val_categorical_accuracy: 0.9631
Epoch 28/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0025 - categorical_accuracy: 0.9872 - val_loss: 0.0072 - val_categorical_accuracy: 0.9634
Epoch 29/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0023 - categorical_accuracy: 0.9879 - val_loss: 0.0075 - val_categorical_accuracy: 0.9613
Epoch 30/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0023 - categorical_accuracy: 0.9881 - val_loss: 0.0073 - val_categorical_accuracy: 0.9621
Epoch 31/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0022 - categorical_accuracy: 0.9887 - val_loss: 0.0077 - val_categorical_accuracy: 0.9606
Epoch 32/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0021 - categorical_accuracy: 0.9896 - val_loss: 0.0074 - val_categorical_accuracy: 0.9617

Epoch 00032: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 33/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0019 - categorical_accuracy: 0.9900 - val_loss: 0.0072 - val_categorical_accuracy: 0.9644
Epoch 34/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0018 - categorical_accuracy: 0.9912 - val_loss: 0.0072 - val_categorical_accuracy: 0.9641
Epoch 35/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0018 - categorical_accuracy: 0.9912 - val_loss: 0.0072 - val_categorical_accuracy: 0.9646
Epoch 36/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0018 - categorical_accuracy: 0.9914 - val_loss: 0.0072 - val_categorical_accuracy: 0.9637
Epoch 37/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0018 - categorical_accuracy: 0.9915 - val_loss: 0.0073 - val_categorical_accuracy: 0.9636

Epoch 00037: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 38/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0018 - categorical_accuracy: 0.9912 - val_loss: 0.0072 - val_categorical_accuracy: 0.9641
Epoch 39/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0017 - categorical_accuracy: 0.9914 - val_loss: 0.0072 - val_categorical_accuracy: 0.9650
Epoch 40/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0017 - categorical_accuracy: 0.9916 - val_loss: 0.0072 - val_categorical_accuracy: 0.9646
Epoch 41/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0017 - categorical_accuracy: 0.9919 - val_loss: 0.0072 - val_categorical_accuracy: 0.9644
Epoch 00041: early stopping
========= generating oof predictions 12:56:36 =========
========= generating test set predictions 12:56:38 =========
========= fitting 2 th model 12:57:08 =========
Epoch 1/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0620 - categorical_accuracy: 0.6600 - val_loss: 0.0303 - val_categorical_accuracy: 0.8350
Epoch 2/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0210 - categorical_accuracy: 0.8921 - val_loss: 0.0188 - val_categorical_accuracy: 0.8958
Epoch 3/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0159 - categorical_accuracy: 0.9157 - val_loss: 0.0195 - val_categorical_accuracy: 0.8876
Epoch 4/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0135 - categorical_accuracy: 0.9272 - val_loss: 0.0145 - val_categorical_accuracy: 0.9173
Epoch 5/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0119 - categorical_accuracy: 0.9354 - val_loss: 0.0150 - val_categorical_accuracy: 0.9151
Epoch 6/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0106 - categorical_accuracy: 0.9422 - val_loss: 0.0130 - val_categorical_accuracy: 0.9274
Epoch 7/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0097 - categorical_accuracy: 0.9473 - val_loss: 0.0160 - val_categorical_accuracy: 0.9092
Epoch 8/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0090 - categorical_accuracy: 0.9503 - val_loss: 0.0131 - val_categorical_accuracy: 0.9238
Epoch 9/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0085 - categorical_accuracy: 0.9532 - val_loss: 0.0128 - val_categorical_accuracy: 0.9271
Epoch 10/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0079 - categorical_accuracy: 0.9559 - val_loss: 0.0127 - val_categorical_accuracy: 0.9304
Epoch 11/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0075 - categorical_accuracy: 0.9586 - val_loss: 0.0147 - val_categorical_accuracy: 0.9183
Epoch 12/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0070 - categorical_accuracy: 0.9621 - val_loss: 0.0132 - val_categorical_accuracy: 0.9260
Epoch 13/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0066 - categorical_accuracy: 0.9631 - val_loss: 0.0134 - val_categorical_accuracy: 0.9249
Epoch 14/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0063 - categorical_accuracy: 0.9654 - val_loss: 0.0115 - val_categorical_accuracy: 0.9355
Epoch 15/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0061 - categorical_accuracy: 0.9660 - val_loss: 0.0131 - val_categorical_accuracy: 0.9306
Epoch 16/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0057 - categorical_accuracy: 0.9684 - val_loss: 0.0128 - val_categorical_accuracy: 0.9289
Epoch 17/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0054 - categorical_accuracy: 0.9696 - val_loss: 0.0116 - val_categorical_accuracy: 0.9389
Epoch 18/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0053 - categorical_accuracy: 0.9703 - val_loss: 0.0143 - val_categorical_accuracy: 0.9262
Epoch 19/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0050 - categorical_accuracy: 0.9717 - val_loss: 0.0113 - val_categorical_accuracy: 0.9395
Epoch 20/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0048 - categorical_accuracy: 0.9737 - val_loss: 0.0109 - val_categorical_accuracy: 0.9424
Epoch 21/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0046 - categorical_accuracy: 0.9742 - val_loss: 0.0122 - val_categorical_accuracy: 0.9381
Epoch 22/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0043 - categorical_accuracy: 0.9751 - val_loss: 0.0111 - val_categorical_accuracy: 0.9393
Epoch 23/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0043 - categorical_accuracy: 0.9763 - val_loss: 0.0120 - val_categorical_accuracy: 0.9371
Epoch 24/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0041 - categorical_accuracy: 0.9767 - val_loss: 0.0125 - val_categorical_accuracy: 0.9373
Epoch 25/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0039 - categorical_accuracy: 0.9779 - val_loss: 0.0114 - val_categorical_accuracy: 0.9408
Epoch 26/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0037 - categorical_accuracy: 0.9793 - val_loss: 0.0110 - val_categorical_accuracy: 0.9456

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 27/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0029 - categorical_accuracy: 0.9843 - val_loss: 0.0076 - val_categorical_accuracy: 0.9591
Epoch 28/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9866 - val_loss: 0.0076 - val_categorical_accuracy: 0.9603
Epoch 29/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0075 - val_categorical_accuracy: 0.9597
Epoch 30/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9885 - val_loss: 0.0077 - val_categorical_accuracy: 0.9580
Epoch 31/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0021 - categorical_accuracy: 0.9895 - val_loss: 0.0079 - val_categorical_accuracy: 0.9590
Epoch 32/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0021 - categorical_accuracy: 0.9894 - val_loss: 0.0080 - val_categorical_accuracy: 0.9572
Epoch 33/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0020 - categorical_accuracy: 0.9898 - val_loss: 0.0083 - val_categorical_accuracy: 0.9569
Epoch 34/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0020 - categorical_accuracy: 0.9898 - val_loss: 0.0083 - val_categorical_accuracy: 0.9579
Epoch 35/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0019 - categorical_accuracy: 0.9904 - val_loss: 0.0081 - val_categorical_accuracy: 0.9592

Epoch 00035: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 36/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0018 - categorical_accuracy: 0.9913 - val_loss: 0.0075 - val_categorical_accuracy: 0.9614
Epoch 37/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0016 - categorical_accuracy: 0.9922 - val_loss: 0.0075 - val_categorical_accuracy: 0.9624
Epoch 38/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0016 - categorical_accuracy: 0.9920 - val_loss: 0.0075 - val_categorical_accuracy: 0.9620
Epoch 39/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0016 - categorical_accuracy: 0.9924 - val_loss: 0.0076 - val_categorical_accuracy: 0.9620
Epoch 40/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0016 - categorical_accuracy: 0.9923 - val_loss: 0.0076 - val_categorical_accuracy: 0.9616

Epoch 00040: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 41/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0015 - categorical_accuracy: 0.9925 - val_loss: 0.0076 - val_categorical_accuracy: 0.9617
Epoch 42/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0015 - categorical_accuracy: 0.9925 - val_loss: 0.0076 - val_categorical_accuracy: 0.9617
Epoch 43/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0015 - categorical_accuracy: 0.9924 - val_loss: 0.0076 - val_categorical_accuracy: 0.9613
Epoch 44/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0015 - categorical_accuracy: 0.9928 - val_loss: 0.0076 - val_categorical_accuracy: 0.9612
Epoch 45/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0015 - categorical_accuracy: 0.9928 - val_loss: 0.0076 - val_categorical_accuracy: 0.9617
Epoch 46/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0015 - categorical_accuracy: 0.9928 - val_loss: 0.0076 - val_categorical_accuracy: 0.9616
Epoch 47/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0015 - categorical_accuracy: 0.9929 - val_loss: 0.0076 - val_categorical_accuracy: 0.9622
Epoch 48/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0015 - categorical_accuracy: 0.9930 - val_loss: 0.0076 - val_categorical_accuracy: 0.9620
Epoch 49/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0015 - categorical_accuracy: 0.9929 - val_loss: 0.0076 - val_categorical_accuracy: 0.9617
Epoch 50/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0015 - categorical_accuracy: 0.9930 - val_loss: 0.0076 - val_categorical_accuracy: 0.9623
Epoch 51/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0015 - categorical_accuracy: 0.9928 - val_loss: 0.0076 - val_categorical_accuracy: 0.9621
Epoch 00051: early stopping
========= generating oof predictions 13:25:54 =========
========= generating test set predictions 13:25:57 =========
========= fitting 3 th model 13:26:27 =========
Epoch 1/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0633 - categorical_accuracy: 0.6486 - val_loss: 0.0259 - val_categorical_accuracy: 0.8584
Epoch 2/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0208 - categorical_accuracy: 0.8923 - val_loss: 0.0185 - val_categorical_accuracy: 0.8970
Epoch 3/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0158 - categorical_accuracy: 0.9154 - val_loss: 0.0169 - val_categorical_accuracy: 0.9043
Epoch 4/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0134 - categorical_accuracy: 0.9275 - val_loss: 0.0176 - val_categorical_accuracy: 0.8979
Epoch 5/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0118 - categorical_accuracy: 0.9356 - val_loss: 0.0136 - val_categorical_accuracy: 0.9229
Epoch 6/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0107 - categorical_accuracy: 0.9415 - val_loss: 0.0147 - val_categorical_accuracy: 0.9179
Epoch 7/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0097 - categorical_accuracy: 0.9470 - val_loss: 0.0174 - val_categorical_accuracy: 0.9047
Epoch 8/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0090 - categorical_accuracy: 0.9505 - val_loss: 0.0122 - val_categorical_accuracy: 0.9322
Epoch 9/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0085 - categorical_accuracy: 0.9534 - val_loss: 0.0132 - val_categorical_accuracy: 0.9271
Epoch 10/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0079 - categorical_accuracy: 0.9565 - val_loss: 0.0123 - val_categorical_accuracy: 0.9318
Epoch 11/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0075 - categorical_accuracy: 0.9592 - val_loss: 0.0157 - val_categorical_accuracy: 0.9154
Epoch 12/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0069 - categorical_accuracy: 0.9619 - val_loss: 0.0115 - val_categorical_accuracy: 0.9361
Epoch 13/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0067 - categorical_accuracy: 0.9630 - val_loss: 0.0121 - val_categorical_accuracy: 0.9344
Epoch 14/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0063 - categorical_accuracy: 0.9646 - val_loss: 0.0121 - val_categorical_accuracy: 0.9341
Epoch 15/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0060 - categorical_accuracy: 0.9669 - val_loss: 0.0116 - val_categorical_accuracy: 0.9378
Epoch 16/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0058 - categorical_accuracy: 0.9672 - val_loss: 0.0140 - val_categorical_accuracy: 0.9261
Epoch 17/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0055 - categorical_accuracy: 0.9698 - val_loss: 0.0125 - val_categorical_accuracy: 0.9320
Epoch 18/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0052 - categorical_accuracy: 0.9710 - val_loss: 0.0119 - val_categorical_accuracy: 0.9360

Epoch 00018: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 19/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0041 - categorical_accuracy: 0.9772 - val_loss: 0.0074 - val_categorical_accuracy: 0.9604
Epoch 20/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0036 - categorical_accuracy: 0.9806 - val_loss: 0.0074 - val_categorical_accuracy: 0.9601
Epoch 21/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0034 - categorical_accuracy: 0.9815 - val_loss: 0.0077 - val_categorical_accuracy: 0.9595
Epoch 22/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0033 - categorical_accuracy: 0.9826 - val_loss: 0.0075 - val_categorical_accuracy: 0.9594
Epoch 23/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0032 - categorical_accuracy: 0.9835 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 24/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0030 - categorical_accuracy: 0.9841 - val_loss: 0.0078 - val_categorical_accuracy: 0.9592
Epoch 25/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0030 - categorical_accuracy: 0.9847 - val_loss: 0.0077 - val_categorical_accuracy: 0.9601

Epoch 00025: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 26/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0028 - categorical_accuracy: 0.9856 - val_loss: 0.0071 - val_categorical_accuracy: 0.9632
Epoch 27/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0027 - categorical_accuracy: 0.9858 - val_loss: 0.0071 - val_categorical_accuracy: 0.9627
Epoch 28/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0026 - categorical_accuracy: 0.9864 - val_loss: 0.0071 - val_categorical_accuracy: 0.9618
Epoch 29/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0026 - categorical_accuracy: 0.9870 - val_loss: 0.0071 - val_categorical_accuracy: 0.9632
Epoch 30/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0026 - categorical_accuracy: 0.9871 - val_loss: 0.0071 - val_categorical_accuracy: 0.9625
Epoch 31/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0025 - categorical_accuracy: 0.9870 - val_loss: 0.0071 - val_categorical_accuracy: 0.9626
Epoch 32/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0025 - categorical_accuracy: 0.9875 - val_loss: 0.0072 - val_categorical_accuracy: 0.9626

Epoch 00032: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 33/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0071 - val_categorical_accuracy: 0.9630
Epoch 34/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9877 - val_loss: 0.0071 - val_categorical_accuracy: 0.9628
Epoch 35/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9877 - val_loss: 0.0071 - val_categorical_accuracy: 0.9626
Epoch 36/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9875 - val_loss: 0.0071 - val_categorical_accuracy: 0.9627
Epoch 37/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0071 - val_categorical_accuracy: 0.9623
Epoch 38/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0071 - val_categorical_accuracy: 0.9624
Epoch 39/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0024 - categorical_accuracy: 0.9880 - val_loss: 0.0071 - val_categorical_accuracy: 0.9625
Epoch 40/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9881 - val_loss: 0.0072 - val_categorical_accuracy: 0.9623
Epoch 41/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0071 - val_categorical_accuracy: 0.9624
Epoch 42/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0023 - categorical_accuracy: 0.9885 - val_loss: 0.0072 - val_categorical_accuracy: 0.9625
Epoch 00042: early stopping
========= generating oof predictions 13:50:09 =========
========= generating test set predictions 13:50:11 =========
========= fitting 4 th model 13:50:42 =========
Epoch 1/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0619 - categorical_accuracy: 0.6587 - val_loss: 0.0285 - val_categorical_accuracy: 0.8446
Epoch 2/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0208 - categorical_accuracy: 0.8926 - val_loss: 0.0183 - val_categorical_accuracy: 0.8945
Epoch 3/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0158 - categorical_accuracy: 0.9164 - val_loss: 0.0163 - val_categorical_accuracy: 0.9082
Epoch 4/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0134 - categorical_accuracy: 0.9264 - val_loss: 0.0198 - val_categorical_accuracy: 0.8898
Epoch 5/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0118 - categorical_accuracy: 0.9354 - val_loss: 0.0184 - val_categorical_accuracy: 0.8962
Epoch 6/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0107 - categorical_accuracy: 0.9421 - val_loss: 0.0186 - val_categorical_accuracy: 0.8971
Epoch 7/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0095 - categorical_accuracy: 0.9477 - val_loss: 0.0172 - val_categorical_accuracy: 0.9068
Epoch 8/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0090 - categorical_accuracy: 0.9501 - val_loss: 0.0160 - val_categorical_accuracy: 0.9122
Epoch 9/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0083 - categorical_accuracy: 0.9541 - val_loss: 0.0172 - val_categorical_accuracy: 0.9081
Epoch 10/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0079 - categorical_accuracy: 0.9558 - val_loss: 0.0151 - val_categorical_accuracy: 0.9154
Epoch 11/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0073 - categorical_accuracy: 0.9598 - val_loss: 0.0120 - val_categorical_accuracy: 0.9328
Epoch 12/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0068 - categorical_accuracy: 0.9621 - val_loss: 0.0115 - val_categorical_accuracy: 0.9369
Epoch 13/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0067 - categorical_accuracy: 0.9624 - val_loss: 0.0105 - val_categorical_accuracy: 0.9430
Epoch 14/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0063 - categorical_accuracy: 0.9649 - val_loss: 0.0131 - val_categorical_accuracy: 0.9274
Epoch 15/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0060 - categorical_accuracy: 0.9664 - val_loss: 0.0146 - val_categorical_accuracy: 0.9254
Epoch 16/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0057 - categorical_accuracy: 0.9684 - val_loss: 0.0119 - val_categorical_accuracy: 0.9352
Epoch 17/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0056 - categorical_accuracy: 0.9678 - val_loss: 0.0110 - val_categorical_accuracy: 0.9388
Epoch 18/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0053 - categorical_accuracy: 0.9707 - val_loss: 0.0116 - val_categorical_accuracy: 0.9387
Epoch 19/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0050 - categorical_accuracy: 0.9719 - val_loss: 0.0128 - val_categorical_accuracy: 0.9352

Epoch 00019: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 20/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0039 - categorical_accuracy: 0.9778 - val_loss: 0.0077 - val_categorical_accuracy: 0.9578
Epoch 21/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0035 - categorical_accuracy: 0.9809 - val_loss: 0.0074 - val_categorical_accuracy: 0.9583
Epoch 22/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0033 - categorical_accuracy: 0.9826 - val_loss: 0.0078 - val_categorical_accuracy: 0.9559
Epoch 23/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0032 - categorical_accuracy: 0.9837 - val_loss: 0.0076 - val_categorical_accuracy: 0.9575
Epoch 24/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0030 - categorical_accuracy: 0.9835 - val_loss: 0.0076 - val_categorical_accuracy: 0.9579
Epoch 25/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0030 - categorical_accuracy: 0.9847 - val_loss: 0.0076 - val_categorical_accuracy: 0.9598
Epoch 26/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0028 - categorical_accuracy: 0.9854 - val_loss: 0.0074 - val_categorical_accuracy: 0.9609
Epoch 27/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0028 - categorical_accuracy: 0.9851 - val_loss: 0.0076 - val_categorical_accuracy: 0.9585

Epoch 00027: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 28/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0026 - categorical_accuracy: 0.9863 - val_loss: 0.0070 - val_categorical_accuracy: 0.9617
Epoch 29/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0025 - categorical_accuracy: 0.9872 - val_loss: 0.0070 - val_categorical_accuracy: 0.9618
Epoch 30/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0025 - categorical_accuracy: 0.9875 - val_loss: 0.0070 - val_categorical_accuracy: 0.9623
Epoch 31/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9873 - val_loss: 0.0070 - val_categorical_accuracy: 0.9622
Epoch 32/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9880 - val_loss: 0.0070 - val_categorical_accuracy: 0.9618
Epoch 33/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0024 - categorical_accuracy: 0.9881 - val_loss: 0.0070 - val_categorical_accuracy: 0.9623
Epoch 34/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9880 - val_loss: 0.0071 - val_categorical_accuracy: 0.9621

Epoch 00034: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 35/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0023 - categorical_accuracy: 0.9886 - val_loss: 0.0070 - val_categorical_accuracy: 0.9627
Epoch 36/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0023 - categorical_accuracy: 0.9887 - val_loss: 0.0070 - val_categorical_accuracy: 0.9629
Epoch 37/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0023 - categorical_accuracy: 0.9888 - val_loss: 0.0070 - val_categorical_accuracy: 0.9624
Epoch 38/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0023 - categorical_accuracy: 0.9887 - val_loss: 0.0070 - val_categorical_accuracy: 0.9623
Epoch 39/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0022 - categorical_accuracy: 0.9889 - val_loss: 0.0070 - val_categorical_accuracy: 0.9624
Epoch 40/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0022 - categorical_accuracy: 0.9887 - val_loss: 0.0070 - val_categorical_accuracy: 0.9625
Epoch 41/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0023 - categorical_accuracy: 0.9884 - val_loss: 0.0070 - val_categorical_accuracy: 0.9624
Epoch 42/65535
420/420 [==============================] - 33s 80ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0070 - val_categorical_accuracy: 0.9621
Epoch 43/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9887 - val_loss: 0.0070 - val_categorical_accuracy: 0.9622
Epoch 44/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0022 - categorical_accuracy: 0.9891 - val_loss: 0.0070 - val_categorical_accuracy: 0.9623
Epoch 45/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9893 - val_loss: 0.0070 - val_categorical_accuracy: 0.9627
Epoch 00045: early stopping
========= generating oof predictions 14:16:03 =========
========= generating test set predictions 14:16:05 =========
========= fitting 5 th model 14:16:36 =========
Epoch 1/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0634 - categorical_accuracy: 0.6514 - val_loss: 0.0345 - val_categorical_accuracy: 0.8035
Epoch 2/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0213 - categorical_accuracy: 0.8909 - val_loss: 0.0310 - val_categorical_accuracy: 0.8303
Epoch 3/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0159 - categorical_accuracy: 0.9158 - val_loss: 0.0161 - val_categorical_accuracy: 0.9084
Epoch 4/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0134 - categorical_accuracy: 0.9279 - val_loss: 0.0165 - val_categorical_accuracy: 0.9076
Epoch 5/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0117 - categorical_accuracy: 0.9366 - val_loss: 0.0168 - val_categorical_accuracy: 0.9084
Epoch 6/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0107 - categorical_accuracy: 0.9419 - val_loss: 0.0221 - val_categorical_accuracy: 0.8847
Epoch 7/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0098 - categorical_accuracy: 0.9462 - val_loss: 0.0138 - val_categorical_accuracy: 0.9238
Epoch 8/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0091 - categorical_accuracy: 0.9503 - val_loss: 0.0117 - val_categorical_accuracy: 0.9342
Epoch 9/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0083 - categorical_accuracy: 0.9539 - val_loss: 0.0125 - val_categorical_accuracy: 0.9297
Epoch 10/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0078 - categorical_accuracy: 0.9569 - val_loss: 0.0167 - val_categorical_accuracy: 0.9104
Epoch 11/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0074 - categorical_accuracy: 0.9595 - val_loss: 0.0111 - val_categorical_accuracy: 0.9389
Epoch 12/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0068 - categorical_accuracy: 0.9624 - val_loss: 0.0127 - val_categorical_accuracy: 0.9313
Epoch 13/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0066 - categorical_accuracy: 0.9632 - val_loss: 0.0109 - val_categorical_accuracy: 0.9415
Epoch 14/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0062 - categorical_accuracy: 0.9655 - val_loss: 0.0163 - val_categorical_accuracy: 0.9212
Epoch 15/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0059 - categorical_accuracy: 0.9671 - val_loss: 0.0140 - val_categorical_accuracy: 0.9275
Epoch 16/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0057 - categorical_accuracy: 0.9680 - val_loss: 0.0142 - val_categorical_accuracy: 0.9299
Epoch 17/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0054 - categorical_accuracy: 0.9701 - val_loss: 0.0121 - val_categorical_accuracy: 0.9379
Epoch 18/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0051 - categorical_accuracy: 0.9714 - val_loss: 0.0119 - val_categorical_accuracy: 0.9341
Epoch 19/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0050 - categorical_accuracy: 0.9711 - val_loss: 0.0127 - val_categorical_accuracy: 0.9337

Epoch 00019: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 20/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0038 - categorical_accuracy: 0.9787 - val_loss: 0.0076 - val_categorical_accuracy: 0.9594
Epoch 21/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0034 - categorical_accuracy: 0.9819 - val_loss: 0.0075 - val_categorical_accuracy: 0.9595
Epoch 22/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0032 - categorical_accuracy: 0.9830 - val_loss: 0.0078 - val_categorical_accuracy: 0.9577
Epoch 23/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0030 - categorical_accuracy: 0.9841 - val_loss: 0.0076 - val_categorical_accuracy: 0.9586
Epoch 24/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0029 - categorical_accuracy: 0.9848 - val_loss: 0.0075 - val_categorical_accuracy: 0.9599
Epoch 25/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0029 - categorical_accuracy: 0.9846 - val_loss: 0.0078 - val_categorical_accuracy: 0.9580
Epoch 26/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0028 - categorical_accuracy: 0.9855 - val_loss: 0.0077 - val_categorical_accuracy: 0.9581

Epoch 00026: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 27/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0025 - categorical_accuracy: 0.9873 - val_loss: 0.0072 - val_categorical_accuracy: 0.9632
Epoch 28/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0071 - val_categorical_accuracy: 0.9631
Epoch 29/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0072 - val_categorical_accuracy: 0.9626
Epoch 30/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9878 - val_loss: 0.0072 - val_categorical_accuracy: 0.9632
Epoch 31/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0024 - categorical_accuracy: 0.9878 - val_loss: 0.0071 - val_categorical_accuracy: 0.9630
Epoch 32/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0023 - categorical_accuracy: 0.9881 - val_loss: 0.0071 - val_categorical_accuracy: 0.9637
Epoch 33/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0023 - categorical_accuracy: 0.9882 - val_loss: 0.0072 - val_categorical_accuracy: 0.9632

Epoch 00033: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 34/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0071 - val_categorical_accuracy: 0.9641
Epoch 35/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9886 - val_loss: 0.0072 - val_categorical_accuracy: 0.9638
Epoch 36/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0071 - val_categorical_accuracy: 0.9640
Epoch 37/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0072 - val_categorical_accuracy: 0.9639
Epoch 38/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9885 - val_loss: 0.0072 - val_categorical_accuracy: 0.9638
Epoch 39/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9893 - val_loss: 0.0072 - val_categorical_accuracy: 0.9641
Epoch 40/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0022 - categorical_accuracy: 0.9891 - val_loss: 0.0072 - val_categorical_accuracy: 0.9640
Epoch 41/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0022 - categorical_accuracy: 0.9891 - val_loss: 0.0071 - val_categorical_accuracy: 0.9640
Epoch 42/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0022 - categorical_accuracy: 0.9892 - val_loss: 0.0072 - val_categorical_accuracy: 0.9641
Epoch 43/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0022 - categorical_accuracy: 0.9891 - val_loss: 0.0072 - val_categorical_accuracy: 0.9641
Epoch 44/65535
420/420 [==============================] - 34s 80ms/step - loss: 0.0022 - categorical_accuracy: 0.9895 - val_loss: 0.0072 - val_categorical_accuracy: 0.9632
Epoch 45/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0021 - categorical_accuracy: 0.9897 - val_loss: 0.0072 - val_categorical_accuracy: 0.9638
Epoch 46/65535
420/420 [==============================] - 34s 81ms/step - loss: 0.0021 - categorical_accuracy: 0.9896 - val_loss: 0.0072 - val_categorical_accuracy: 0.9642
Epoch 00046: early stopping
========= generating oof predictions 14:42:36 =========
========= generating test set predictions 14:42:38 =========
train loss avg 0.001953512288755209 -- std 0.0003220816483422892, val loss avg 0.0072448304545297505 -- std 0.0001986173989676116
train acc avg 0.9904242268178287 -- std 0.0016302597558725295, val acc avg 0.9631858926225831 -- std 0.0009193923594028998
mean nb epochs 45.0
dump oof predicted probs
dump test set predicted probs
