ren (master *+) python $ python train.py mfcc_10_40_20 arm_gru xl 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= processing data =======
========== input shape is : (51, 10) ===========
------------- SUMMARY OF MODEL -------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 51, 10)            0
_________________________________________________________________
cu_dnngru_1 (CuDNNGRU)       (None, 616)               1160544
_________________________________________________________________
dropout_1 (Dropout)          (None, 616)               0
_________________________________________________________________
dense_1 (Dense)              (None, 31)                19127
=================================================================
Total params: 1,179,671
Trainable params: 1,179,671
Non-trainable params: 0
_________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 08:46:06 =========
Epoch 1/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0774 - categorical_accuracy: 0.5340 - val_loss: 0.0361 - val_categorical_accuracy: 0.8088
Epoch 2/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0308 - categorical_accuracy: 0.8283 - val_loss: 0.0232 - val_categorical_accuracy: 0.8749
Epoch 3/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0225 - categorical_accuracy: 0.8764 - val_loss: 0.0201 - val_categorical_accuracy: 0.8884
Epoch 4/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0186 - categorical_accuracy: 0.8970 - val_loss: 0.0175 - val_categorical_accuracy: 0.9003
Epoch 5/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0163 - categorical_accuracy: 0.9116 - val_loss: 0.0159 - val_categorical_accuracy: 0.9112
Epoch 6/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0144 - categorical_accuracy: 0.9207 - val_loss: 0.0160 - val_categorical_accuracy: 0.9105
Epoch 7/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0131 - categorical_accuracy: 0.9273 - val_loss: 0.0145 - val_categorical_accuracy: 0.9179
Epoch 8/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0123 - categorical_accuracy: 0.9323 - val_loss: 0.0149 - val_categorical_accuracy: 0.9172
Epoch 9/65535
420/420 [==============================] - 5s 13ms/step - loss: 0.0113 - categorical_accuracy: 0.9374 - val_loss: 0.0139 - val_categorical_accuracy: 0.9210
Epoch 10/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0105 - categorical_accuracy: 0.9426 - val_loss: 0.0141 - val_categorical_accuracy: 0.9221
Epoch 11/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0099 - categorical_accuracy: 0.9456 - val_loss: 0.0124 - val_categorical_accuracy: 0.9318
Epoch 12/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0093 - categorical_accuracy: 0.9485 - val_loss: 0.0129 - val_categorical_accuracy: 0.9290
Epoch 13/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0089 - categorical_accuracy: 0.9508 - val_loss: 0.0131 - val_categorical_accuracy: 0.9264
Epoch 14/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0085 - categorical_accuracy: 0.9529 - val_loss: 0.0121 - val_categorical_accuracy: 0.9364
Epoch 15/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0082 - categorical_accuracy: 0.9551 - val_loss: 0.0130 - val_categorical_accuracy: 0.9291
Epoch 16/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0079 - categorical_accuracy: 0.9564 - val_loss: 0.0120 - val_categorical_accuracy: 0.9335
Epoch 17/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0075 - categorical_accuracy: 0.9581 - val_loss: 0.0121 - val_categorical_accuracy: 0.9341
Epoch 18/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0074 - categorical_accuracy: 0.9589 - val_loss: 0.0122 - val_categorical_accuracy: 0.9331
Epoch 19/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0072 - categorical_accuracy: 0.9603 - val_loss: 0.0121 - val_categorical_accuracy: 0.9356
Epoch 20/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0070 - categorical_accuracy: 0.9606 - val_loss: 0.0126 - val_categorical_accuracy: 0.9338
Epoch 00020: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 21/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0051 - categorical_accuracy: 0.9715 - val_loss: 0.0106 - val_categorical_accuracy: 0.9443
Epoch 22/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0041 - categorical_accuracy: 0.9780 - val_loss: 0.0105 - val_categorical_accuracy: 0.9470
Epoch 23/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0038 - categorical_accuracy: 0.9801 - val_loss: 0.0104 - val_categorical_accuracy: 0.9470
Epoch 24/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0036 - categorical_accuracy: 0.9812 - val_loss: 0.0104 - val_categorical_accuracy: 0.9460
Epoch 25/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0034 - categorical_accuracy: 0.9823 - val_loss: 0.0104 - val_categorical_accuracy: 0.9466
Epoch 26/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0033 - categorical_accuracy: 0.9831 - val_loss: 0.0105 - val_categorical_accuracy: 0.9469
Epoch 27/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0032 - categorical_accuracy: 0.9841 - val_loss: 0.0104 - val_categorical_accuracy: 0.9466
Epoch 28/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9847 - val_loss: 0.0106 - val_categorical_accuracy: 0.9456
Epoch 29/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0029 - categorical_accuracy: 0.9852 - val_loss: 0.0105 - val_categorical_accuracy: 0.9458
Epoch 00029: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 30/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0026 - categorical_accuracy: 0.9866 - val_loss: 0.0104 - val_categorical_accuracy: 0.9467
Epoch 31/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9869 - val_loss: 0.0104 - val_categorical_accuracy: 0.9465
Epoch 32/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9875 - val_loss: 0.0104 - val_categorical_accuracy: 0.9468
Epoch 33/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9874 - val_loss: 0.0104 - val_categorical_accuracy: 0.9463
Epoch 34/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9874 - val_loss: 0.0104 - val_categorical_accuracy: 0.9475
Epoch 00034: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 35/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9878 - val_loss: 0.0104 - val_categorical_accuracy: 0.9469
Epoch 36/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9883 - val_loss: 0.0104 - val_categorical_accuracy: 0.9469
Epoch 37/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9880 - val_loss: 0.0104 - val_categorical_accuracy: 0.9473
Epoch 38/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9885 - val_loss: 0.0104 - val_categorical_accuracy: 0.9473
Epoch 39/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9883 - val_loss: 0.0104 - val_categorical_accuracy: 0.9477
Epoch 40/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9884 - val_loss: 0.0104 - val_categorical_accuracy: 0.9469
Epoch 00040: early stopping
========= generating oof predictions 08:49:55 =========
========= generating test set predictions 08:49:56 =========
========= fitting 2 th model 08:50:06 =========
Epoch 1/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0777 - categorical_accuracy: 0.5294 - val_loss: 0.0369 - val_categorical_accuracy: 0.8011
Epoch 2/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0309 - categorical_accuracy: 0.8309 - val_loss: 0.0233 - val_categorical_accuracy: 0.8719
Epoch 3/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0221 - categorical_accuracy: 0.8774 - val_loss: 0.0189 - val_categorical_accuracy: 0.8944
Epoch 4/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0184 - categorical_accuracy: 0.8984 - val_loss: 0.0170 - val_categorical_accuracy: 0.9049
Epoch 5/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0159 - categorical_accuracy: 0.9128 - val_loss: 0.0160 - val_categorical_accuracy: 0.9087
Epoch 6/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0144 - categorical_accuracy: 0.9205 - val_loss: 0.0150 - val_categorical_accuracy: 0.9165
Epoch 7/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0132 - categorical_accuracy: 0.9268 - val_loss: 0.0151 - val_categorical_accuracy: 0.9153
Epoch 8/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0122 - categorical_accuracy: 0.9333 - val_loss: 0.0138 - val_categorical_accuracy: 0.9232
Epoch 9/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0112 - categorical_accuracy: 0.9383 - val_loss: 0.0137 - val_categorical_accuracy: 0.9232
Epoch 10/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0105 - categorical_accuracy: 0.9418 - val_loss: 0.0137 - val_categorical_accuracy: 0.9229
Epoch 11/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0099 - categorical_accuracy: 0.9455 - val_loss: 0.0126 - val_categorical_accuracy: 0.9278
Epoch 12/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0095 - categorical_accuracy: 0.9481 - val_loss: 0.0131 - val_categorical_accuracy: 0.9270
Epoch 13/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0091 - categorical_accuracy: 0.9499 - val_loss: 0.0130 - val_categorical_accuracy: 0.9281
Epoch 14/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0087 - categorical_accuracy: 0.9519 - val_loss: 0.0125 - val_categorical_accuracy: 0.9291
Epoch 15/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0081 - categorical_accuracy: 0.9553 - val_loss: 0.0129 - val_categorical_accuracy: 0.9276
Epoch 16/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0082 - categorical_accuracy: 0.9554 - val_loss: 0.0140 - val_categorical_accuracy: 0.9226
Epoch 17/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0078 - categorical_accuracy: 0.9570 - val_loss: 0.0129 - val_categorical_accuracy: 0.9280
Epoch 00017: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 18/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0058 - categorical_accuracy: 0.9689 - val_loss: 0.0107 - val_categorical_accuracy: 0.9407
Epoch 19/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0047 - categorical_accuracy: 0.9750 - val_loss: 0.0106 - val_categorical_accuracy: 0.9415
Epoch 20/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0044 - categorical_accuracy: 0.9768 - val_loss: 0.0106 - val_categorical_accuracy: 0.9412
Epoch 21/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0040 - categorical_accuracy: 0.9789 - val_loss: 0.0105 - val_categorical_accuracy: 0.9427
Epoch 22/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0039 - categorical_accuracy: 0.9804 - val_loss: 0.0105 - val_categorical_accuracy: 0.9434
Epoch 23/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0037 - categorical_accuracy: 0.9810 - val_loss: 0.0105 - val_categorical_accuracy: 0.9439
Epoch 24/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0106 - val_categorical_accuracy: 0.9436
Epoch 25/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0034 - categorical_accuracy: 0.9825 - val_loss: 0.0106 - val_categorical_accuracy: 0.9433
Epoch 26/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0033 - categorical_accuracy: 0.9838 - val_loss: 0.0106 - val_categorical_accuracy: 0.9425
Epoch 27/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0032 - categorical_accuracy: 0.9841 - val_loss: 0.0107 - val_categorical_accuracy: 0.9425
Epoch 00027: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 28/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0029 - categorical_accuracy: 0.9855 - val_loss: 0.0104 - val_categorical_accuracy: 0.9457
Epoch 29/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0028 - categorical_accuracy: 0.9857 - val_loss: 0.0104 - val_categorical_accuracy: 0.9448
Epoch 30/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9864 - val_loss: 0.0105 - val_categorical_accuracy: 0.9451
Epoch 31/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9868 - val_loss: 0.0105 - val_categorical_accuracy: 0.9454
Epoch 32/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0026 - categorical_accuracy: 0.9870 - val_loss: 0.0105 - val_categorical_accuracy: 0.9458
Epoch 33/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0026 - categorical_accuracy: 0.9868 - val_loss: 0.0105 - val_categorical_accuracy: 0.9458
Epoch 34/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0026 - categorical_accuracy: 0.9871 - val_loss: 0.0105 - val_categorical_accuracy: 0.9450
Epoch 00034: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 35/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9871 - val_loss: 0.0105 - val_categorical_accuracy: 0.9458
Epoch 36/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9875 - val_loss: 0.0105 - val_categorical_accuracy: 0.9461
Epoch 37/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9877 - val_loss: 0.0105 - val_categorical_accuracy: 0.9457
Epoch 38/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9876 - val_loss: 0.0105 - val_categorical_accuracy: 0.9455
Epoch 39/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9874 - val_loss: 0.0105 - val_categorical_accuracy: 0.9457
Epoch 40/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0105 - val_categorical_accuracy: 0.9459
Epoch 41/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0105 - val_categorical_accuracy: 0.9457
Epoch 42/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0106 - val_categorical_accuracy: 0.9450
Epoch 43/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0106 - val_categorical_accuracy: 0.9454
Epoch 00043: early stopping
========= generating oof predictions 08:54:19 =========
========= generating test set predictions 08:54:20 =========
========= fitting 3 th model 08:54:30 =========
Epoch 1/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0801 - categorical_accuracy: 0.5144 - val_loss: 0.0386 - val_categorical_accuracy: 0.7959
Epoch 2/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0324 - categorical_accuracy: 0.8189 - val_loss: 0.0248 - val_categorical_accuracy: 0.8650
Epoch 3/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0231 - categorical_accuracy: 0.8721 - val_loss: 0.0200 - val_categorical_accuracy: 0.8921
Epoch 4/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0191 - categorical_accuracy: 0.8921 - val_loss: 0.0173 - val_categorical_accuracy: 0.9042
Epoch 5/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0163 - categorical_accuracy: 0.9103 - val_loss: 0.0161 - val_categorical_accuracy: 0.9109
Epoch 6/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0146 - categorical_accuracy: 0.9199 - val_loss: 0.0149 - val_categorical_accuracy: 0.9168
Epoch 7/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0133 - categorical_accuracy: 0.9272 - val_loss: 0.0143 - val_categorical_accuracy: 0.9213
Epoch 8/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0124 - categorical_accuracy: 0.9317 - val_loss: 0.0140 - val_categorical_accuracy: 0.9239
Epoch 9/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0115 - categorical_accuracy: 0.9367 - val_loss: 0.0136 - val_categorical_accuracy: 0.9236
Epoch 10/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0108 - categorical_accuracy: 0.9397 - val_loss: 0.0132 - val_categorical_accuracy: 0.9263
Epoch 11/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0101 - categorical_accuracy: 0.9448 - val_loss: 0.0135 - val_categorical_accuracy: 0.9265
Epoch 12/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0098 - categorical_accuracy: 0.9446 - val_loss: 0.0131 - val_categorical_accuracy: 0.9275
Epoch 13/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0090 - categorical_accuracy: 0.9508 - val_loss: 0.0124 - val_categorical_accuracy: 0.9339
Epoch 14/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0087 - categorical_accuracy: 0.9516 - val_loss: 0.0127 - val_categorical_accuracy: 0.9322
Epoch 15/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0084 - categorical_accuracy: 0.9541 - val_loss: 0.0127 - val_categorical_accuracy: 0.9294
Epoch 16/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0079 - categorical_accuracy: 0.9572 - val_loss: 0.0123 - val_categorical_accuracy: 0.9338
Epoch 17/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0078 - categorical_accuracy: 0.9571 - val_loss: 0.0119 - val_categorical_accuracy: 0.9360
Epoch 18/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0079 - categorical_accuracy: 0.9561 - val_loss: 0.0121 - val_categorical_accuracy: 0.9356
Epoch 19/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0073 - categorical_accuracy: 0.9595 - val_loss: 0.0121 - val_categorical_accuracy: 0.9347
Epoch 20/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0067 - categorical_accuracy: 0.9631 - val_loss: 0.0123 - val_categorical_accuracy: 0.9329
Epoch 21/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0067 - categorical_accuracy: 0.9638 - val_loss: 0.0124 - val_categorical_accuracy: 0.9356
Epoch 22/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0067 - categorical_accuracy: 0.9624 - val_loss: 0.0127 - val_categorical_accuracy: 0.9340
Epoch 23/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0067 - categorical_accuracy: 0.9624 - val_loss: 0.0133 - val_categorical_accuracy: 0.9306
Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 24/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0048 - categorical_accuracy: 0.9744 - val_loss: 0.0105 - val_categorical_accuracy: 0.9461
Epoch 25/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0039 - categorical_accuracy: 0.9789 - val_loss: 0.0105 - val_categorical_accuracy: 0.9457
Epoch 26/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0036 - categorical_accuracy: 0.9806 - val_loss: 0.0104 - val_categorical_accuracy: 0.9465
Epoch 27/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0034 - categorical_accuracy: 0.9824 - val_loss: 0.0105 - val_categorical_accuracy: 0.9458
Epoch 28/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0032 - categorical_accuracy: 0.9835 - val_loss: 0.0104 - val_categorical_accuracy: 0.9470
Epoch 29/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9845 - val_loss: 0.0106 - val_categorical_accuracy: 0.9461
Epoch 30/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9845 - val_loss: 0.0106 - val_categorical_accuracy: 0.9461
Epoch 31/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0028 - categorical_accuracy: 0.9852 - val_loss: 0.0105 - val_categorical_accuracy: 0.9480
Epoch 32/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9857 - val_loss: 0.0105 - val_categorical_accuracy: 0.9471
Epoch 00032: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 33/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9869 - val_loss: 0.0104 - val_categorical_accuracy: 0.9484
Epoch 34/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9876 - val_loss: 0.0104 - val_categorical_accuracy: 0.9477
Epoch 35/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9876 - val_loss: 0.0104 - val_categorical_accuracy: 0.9477
Epoch 36/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9880 - val_loss: 0.0105 - val_categorical_accuracy: 0.9477
Epoch 37/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9885 - val_loss: 0.0105 - val_categorical_accuracy: 0.9478
Epoch 00037: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 38/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9886 - val_loss: 0.0105 - val_categorical_accuracy: 0.9476
Epoch 39/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9883 - val_loss: 0.0105 - val_categorical_accuracy: 0.9476
Epoch 40/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9884 - val_loss: 0.0104 - val_categorical_accuracy: 0.9484
Epoch 41/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9886 - val_loss: 0.0105 - val_categorical_accuracy: 0.9484
Epoch 42/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0105 - val_categorical_accuracy: 0.9484
Epoch 43/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9887 - val_loss: 0.0105 - val_categorical_accuracy: 0.9477
Epoch 44/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0105 - val_categorical_accuracy: 0.9481
Epoch 45/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9887 - val_loss: 0.0105 - val_categorical_accuracy: 0.9479
Epoch 46/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0105 - val_categorical_accuracy: 0.9480
Epoch 47/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0021 - categorical_accuracy: 0.9888 - val_loss: 0.0105 - val_categorical_accuracy: 0.9478
Epoch 48/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0021 - categorical_accuracy: 0.9889 - val_loss: 0.0105 - val_categorical_accuracy: 0.9487
Epoch 00048: early stopping
========= generating oof predictions 08:59:12 =========
========= generating test set predictions 08:59:12 =========
========= fitting 4 th model 08:59:22 =========
Epoch 1/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0792 - categorical_accuracy: 0.5196 - val_loss: 0.0380 - val_categorical_accuracy: 0.7924
Epoch 2/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0311 - categorical_accuracy: 0.8291 - val_loss: 0.0234 - val_categorical_accuracy: 0.8718
Epoch 3/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0222 - categorical_accuracy: 0.8774 - val_loss: 0.0192 - val_categorical_accuracy: 0.8928
Epoch 4/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0182 - categorical_accuracy: 0.8999 - val_loss: 0.0174 - val_categorical_accuracy: 0.9012
Epoch 5/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0157 - categorical_accuracy: 0.9127 - val_loss: 0.0150 - val_categorical_accuracy: 0.9176
Epoch 6/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0140 - categorical_accuracy: 0.9226 - val_loss: 0.0149 - val_categorical_accuracy: 0.9200
Epoch 7/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0127 - categorical_accuracy: 0.9301 - val_loss: 0.0143 - val_categorical_accuracy: 0.9213
Epoch 8/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0117 - categorical_accuracy: 0.9349 - val_loss: 0.0144 - val_categorical_accuracy: 0.9172
Epoch 9/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0110 - categorical_accuracy: 0.9388 - val_loss: 0.0129 - val_categorical_accuracy: 0.9287
Epoch 10/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0101 - categorical_accuracy: 0.9448 - val_loss: 0.0130 - val_categorical_accuracy: 0.9284
Epoch 11/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0093 - categorical_accuracy: 0.9493 - val_loss: 0.0130 - val_categorical_accuracy: 0.9294
Epoch 12/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0092 - categorical_accuracy: 0.9485 - val_loss: 0.0128 - val_categorical_accuracy: 0.9296
Epoch 13/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0087 - categorical_accuracy: 0.9519 - val_loss: 0.0125 - val_categorical_accuracy: 0.9331
Epoch 14/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0082 - categorical_accuracy: 0.9543 - val_loss: 0.0124 - val_categorical_accuracy: 0.9304
Epoch 15/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0079 - categorical_accuracy: 0.9561 - val_loss: 0.0129 - val_categorical_accuracy: 0.9308
Epoch 16/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0076 - categorical_accuracy: 0.9577 - val_loss: 0.0122 - val_categorical_accuracy: 0.9334
Epoch 17/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0073 - categorical_accuracy: 0.9594 - val_loss: 0.0125 - val_categorical_accuracy: 0.9338
Epoch 18/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0074 - categorical_accuracy: 0.9596 - val_loss: 0.0131 - val_categorical_accuracy: 0.9299
Epoch 19/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0072 - categorical_accuracy: 0.9592 - val_loss: 0.0131 - val_categorical_accuracy: 0.9299
Epoch 20/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0067 - categorical_accuracy: 0.9633 - val_loss: 0.0127 - val_categorical_accuracy: 0.9324
Epoch 21/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0065 - categorical_accuracy: 0.9641 - val_loss: 0.0138 - val_categorical_accuracy: 0.9281
Epoch 22/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0062 - categorical_accuracy: 0.9664 - val_loss: 0.0124 - val_categorical_accuracy: 0.9333
Epoch 00022: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 23/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0044 - categorical_accuracy: 0.9759 - val_loss: 0.0105 - val_categorical_accuracy: 0.9445
Epoch 24/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0036 - categorical_accuracy: 0.9807 - val_loss: 0.0105 - val_categorical_accuracy: 0.9457
Epoch 25/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0033 - categorical_accuracy: 0.9824 - val_loss: 0.0104 - val_categorical_accuracy: 0.9457
Epoch 26/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0032 - categorical_accuracy: 0.9832 - val_loss: 0.0106 - val_categorical_accuracy: 0.9454
Epoch 27/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0031 - categorical_accuracy: 0.9839 - val_loss: 0.0107 - val_categorical_accuracy: 0.9451
Epoch 28/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0029 - categorical_accuracy: 0.9846 - val_loss: 0.0106 - val_categorical_accuracy: 0.9454
Epoch 29/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0028 - categorical_accuracy: 0.9853 - val_loss: 0.0107 - val_categorical_accuracy: 0.9449
Epoch 00029: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 30/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0026 - categorical_accuracy: 0.9867 - val_loss: 0.0103 - val_categorical_accuracy: 0.9472
Epoch 31/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9871 - val_loss: 0.0103 - val_categorical_accuracy: 0.9467
Epoch 32/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9872 - val_loss: 0.0103 - val_categorical_accuracy: 0.9478
Epoch 33/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9878 - val_loss: 0.0103 - val_categorical_accuracy: 0.9470
Epoch 34/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0103 - val_categorical_accuracy: 0.9467
Epoch 35/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9877 - val_loss: 0.0104 - val_categorical_accuracy: 0.9471
Epoch 36/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0023 - categorical_accuracy: 0.9881 - val_loss: 0.0104 - val_categorical_accuracy: 0.9473
Epoch 00036: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 37/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9885 - val_loss: 0.0104 - val_categorical_accuracy: 0.9477
Epoch 38/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9883 - val_loss: 0.0104 - val_categorical_accuracy: 0.9466
Epoch 39/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9888 - val_loss: 0.0104 - val_categorical_accuracy: 0.9472
Epoch 40/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0104 - val_categorical_accuracy: 0.9473
Epoch 41/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9885 - val_loss: 0.0104 - val_categorical_accuracy: 0.9472
Epoch 42/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9886 - val_loss: 0.0104 - val_categorical_accuracy: 0.9472
Epoch 43/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0104 - val_categorical_accuracy: 0.9470
Epoch 44/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0021 - categorical_accuracy: 0.9892 - val_loss: 0.0104 - val_categorical_accuracy: 0.9477
Epoch 45/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0021 - categorical_accuracy: 0.9891 - val_loss: 0.0104 - val_categorical_accuracy: 0.9473
Epoch 00045: early stopping
========= generating oof predictions 09:03:48 =========
========= generating test set predictions 09:03:49 =========
========= fitting 5 th model 09:03:59 =========
Epoch 1/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0771 - categorical_accuracy: 0.5365 - val_loss: 0.0369 - val_categorical_accuracy: 0.8050
Epoch 2/65535
420/420 [==============================] - 6s 13ms/step - loss: 0.0312 - categorical_accuracy: 0.8272 - val_loss: 0.0232 - val_categorical_accuracy: 0.8773
Epoch 3/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0225 - categorical_accuracy: 0.8754 - val_loss: 0.0193 - val_categorical_accuracy: 0.8937
Epoch 4/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0186 - categorical_accuracy: 0.8971 - val_loss: 0.0173 - val_categorical_accuracy: 0.9049
Epoch 5/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0162 - categorical_accuracy: 0.9100 - val_loss: 0.0162 - val_categorical_accuracy: 0.9086
Epoch 6/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0145 - categorical_accuracy: 0.9194 - val_loss: 0.0152 - val_categorical_accuracy: 0.9140
Epoch 7/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0131 - categorical_accuracy: 0.9279 - val_loss: 0.0142 - val_categorical_accuracy: 0.9191
Epoch 8/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0120 - categorical_accuracy: 0.9338 - val_loss: 0.0144 - val_categorical_accuracy: 0.9207
Epoch 9/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0112 - categorical_accuracy: 0.9381 - val_loss: 0.0143 - val_categorical_accuracy: 0.9175
Epoch 10/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0103 - categorical_accuracy: 0.9434 - val_loss: 0.0137 - val_categorical_accuracy: 0.9257
Epoch 11/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0098 - categorical_accuracy: 0.9459 - val_loss: 0.0140 - val_categorical_accuracy: 0.9218
Epoch 12/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0093 - categorical_accuracy: 0.9497 - val_loss: 0.0137 - val_categorical_accuracy: 0.9211
Epoch 13/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0087 - categorical_accuracy: 0.9529 - val_loss: 0.0132 - val_categorical_accuracy: 0.9281
Epoch 14/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0084 - categorical_accuracy: 0.9533 - val_loss: 0.0134 - val_categorical_accuracy: 0.9269
Epoch 15/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0081 - categorical_accuracy: 0.9547 - val_loss: 0.0128 - val_categorical_accuracy: 0.9293
Epoch 16/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0077 - categorical_accuracy: 0.9567 - val_loss: 0.0126 - val_categorical_accuracy: 0.9298
Epoch 17/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0073 - categorical_accuracy: 0.9594 - val_loss: 0.0131 - val_categorical_accuracy: 0.9294
Epoch 18/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0072 - categorical_accuracy: 0.9605 - val_loss: 0.0128 - val_categorical_accuracy: 0.9301
Epoch 19/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0070 - categorical_accuracy: 0.9612 - val_loss: 0.0128 - val_categorical_accuracy: 0.9313
Epoch 20/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0071 - categorical_accuracy: 0.9603 - val_loss: 0.0128 - val_categorical_accuracy: 0.9300
Epoch 21/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0067 - categorical_accuracy: 0.9632 - val_loss: 0.0130 - val_categorical_accuracy: 0.9311
Epoch 22/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0064 - categorical_accuracy: 0.9643 - val_loss: 0.0120 - val_categorical_accuracy: 0.9350
Epoch 23/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0063 - categorical_accuracy: 0.9648 - val_loss: 0.0126 - val_categorical_accuracy: 0.9316
Epoch 24/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0061 - categorical_accuracy: 0.9650 - val_loss: 0.0126 - val_categorical_accuracy: 0.9340
Epoch 25/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0061 - categorical_accuracy: 0.9661 - val_loss: 0.0123 - val_categorical_accuracy: 0.9363
Epoch 26/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0060 - categorical_accuracy: 0.9668 - val_loss: 0.0131 - val_categorical_accuracy: 0.9305
Epoch 27/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0060 - categorical_accuracy: 0.9663 - val_loss: 0.0134 - val_categorical_accuracy: 0.9287
Epoch 28/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0060 - categorical_accuracy: 0.9664 - val_loss: 0.0123 - val_categorical_accuracy: 0.9354
Epoch 00028: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 29/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0041 - categorical_accuracy: 0.9775 - val_loss: 0.0111 - val_categorical_accuracy: 0.9430
Epoch 30/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0033 - categorical_accuracy: 0.9825 - val_loss: 0.0110 - val_categorical_accuracy: 0.9444
Epoch 31/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0031 - categorical_accuracy: 0.9840 - val_loss: 0.0110 - val_categorical_accuracy: 0.9445
Epoch 32/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0029 - categorical_accuracy: 0.9846 - val_loss: 0.0110 - val_categorical_accuracy: 0.9459
Epoch 33/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9856 - val_loss: 0.0109 - val_categorical_accuracy: 0.9459
Epoch 34/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0026 - categorical_accuracy: 0.9862 - val_loss: 0.0110 - val_categorical_accuracy: 0.9460
Epoch 35/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0026 - categorical_accuracy: 0.9866 - val_loss: 0.0110 - val_categorical_accuracy: 0.9458
Epoch 36/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0025 - categorical_accuracy: 0.9873 - val_loss: 0.0110 - val_categorical_accuracy: 0.9444
Epoch 37/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0024 - categorical_accuracy: 0.9876 - val_loss: 0.0110 - val_categorical_accuracy: 0.9476
Epoch 00037: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 38/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0022 - categorical_accuracy: 0.9884 - val_loss: 0.0107 - val_categorical_accuracy: 0.9483
Epoch 39/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0021 - categorical_accuracy: 0.9891 - val_loss: 0.0108 - val_categorical_accuracy: 0.9474
Epoch 40/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0021 - categorical_accuracy: 0.9894 - val_loss: 0.0108 - val_categorical_accuracy: 0.9480
Epoch 41/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0020 - categorical_accuracy: 0.9895 - val_loss: 0.0107 - val_categorical_accuracy: 0.9486
Epoch 42/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0020 - categorical_accuracy: 0.9894 - val_loss: 0.0107 - val_categorical_accuracy: 0.9480
Epoch 43/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0020 - categorical_accuracy: 0.9898 - val_loss: 0.0107 - val_categorical_accuracy: 0.9484
Epoch 44/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0020 - categorical_accuracy: 0.9897 - val_loss: 0.0108 - val_categorical_accuracy: 0.9475
Epoch 00044: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 45/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0019 - categorical_accuracy: 0.9901 - val_loss: 0.0107 - val_categorical_accuracy: 0.9479
Epoch 46/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0019 - categorical_accuracy: 0.9900 - val_loss: 0.0107 - val_categorical_accuracy: 0.9477
Epoch 47/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0019 - categorical_accuracy: 0.9901 - val_loss: 0.0107 - val_categorical_accuracy: 0.9483
Epoch 48/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0019 - categorical_accuracy: 0.9902 - val_loss: 0.0107 - val_categorical_accuracy: 0.9480
Epoch 49/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0019 - categorical_accuracy: 0.9900 - val_loss: 0.0107 - val_categorical_accuracy: 0.9483
Epoch 50/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0019 - categorical_accuracy: 0.9906 - val_loss: 0.0107 - val_categorical_accuracy: 0.9483
Epoch 51/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0019 - categorical_accuracy: 0.9902 - val_loss: 0.0107 - val_categorical_accuracy: 0.9482
Epoch 52/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9906 - val_loss: 0.0107 - val_categorical_accuracy: 0.9477
Epoch 53/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9905 - val_loss: 0.0107 - val_categorical_accuracy: 0.9483
Epoch 54/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9906 - val_loss: 0.0107 - val_categorical_accuracy: 0.9480
Epoch 55/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9906 - val_loss: 0.0107 - val_categorical_accuracy: 0.9484
Epoch 56/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9908 - val_loss: 0.0107 - val_categorical_accuracy: 0.9477
Epoch 57/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9908 - val_loss: 0.0107 - val_categorical_accuracy: 0.9484
Epoch 58/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9906 - val_loss: 0.0108 - val_categorical_accuracy: 0.9480
Epoch 59/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9907 - val_loss: 0.0107 - val_categorical_accuracy: 0.9483
Epoch 60/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9911 - val_loss: 0.0107 - val_categorical_accuracy: 0.9483
Epoch 61/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0018 - categorical_accuracy: 0.9908 - val_loss: 0.0108 - val_categorical_accuracy: 0.9481
Epoch 62/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0017 - categorical_accuracy: 0.9911 - val_loss: 0.0108 - val_categorical_accuracy: 0.9485
Epoch 63/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0017 - categorical_accuracy: 0.9908 - val_loss: 0.0108 - val_categorical_accuracy: 0.9486
Epoch 64/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0017 - categorical_accuracy: 0.9910 - val_loss: 0.0108 - val_categorical_accuracy: 0.9490
Epoch 65/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0017 - categorical_accuracy: 0.9909 - val_loss: 0.0108 - val_categorical_accuracy: 0.9488
Epoch 00065: early stopping
========= generating oof predictions 09:10:22 =========
========= generating test set predictions 09:10:23 =========
train loss avg 0.002140920160989484 -- std 0.0002287141389321899, val loss avg 0.010541013808101417 -- std 0.00016018665668791188
train acc avg 0.9890681080694776 -- std 0.0010179895887899078, val acc avg 0.9474089993664838 -- std 0.0012584005901682643
mean nb epochs 48.2
dump oof predicted probs
dump test set predicted probs
