ren (master *+) python $ python train.py mfcc_10_40_20 arm_crnn l 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (51, 10) ===========
------------- SUMMARY OF MODEL -------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 51, 10)            0
_________________________________________________________________
reshape_1 (Reshape)          (None, 51, 10, 1)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 21, 7, 100)        4100
_________________________________________________________________
activation_1 (Activation)    (None, 21, 7, 100)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 21, 7, 100)        0
_________________________________________________________________
permute_1 (Permute)          (None, 21, 100, 7)        0
_________________________________________________________________
reshape_2 (Reshape)          (None, 21, 700)           0
_________________________________________________________________
cu_dnngru_1 (CuDNNGRU)       (None, 21, 136)           341904
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
Total params: 489,411
Trainable params: 489,411
Non-trainable params: 0
_________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 21:18:35 =========
Epoch 1/65535
420/420 [==============================] - 4s 10ms/step - loss: 0.0753 - categorical_accuracy: 0.5174 - val_loss: 0.0295 - val_categorical_accuracy: 0.8333
Epoch 2/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0283 - categorical_accuracy: 0.8387 - val_loss: 0.0211 - val_categorical_accuracy: 0.8823
Epoch 3/65535
420/420 [==============================] - 4s 8ms/step - loss: 0.0213 - categorical_accuracy: 0.8789 - val_loss: 0.0180 - val_categorical_accuracy: 0.9005
Epoch 4/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0185 - categorical_accuracy: 0.8947 - val_loss: 0.0160 - val_categorical_accuracy: 0.9067
Epoch 5/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0165 - categorical_accuracy: 0.9062 - val_loss: 0.0142 - val_categorical_accuracy: 0.9207
Epoch 6/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0150 - categorical_accuracy: 0.9135 - val_loss: 0.0138 - val_categorical_accuracy: 0.9235
Epoch 7/65535
420/420 [==============================] - 4s 8ms/step - loss: 0.0137 - categorical_accuracy: 0.9218 - val_loss: 0.0133 - val_categorical_accuracy: 0.9247
Epoch 8/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0129 - categorical_accuracy: 0.9249 - val_loss: 0.0128 - val_categorical_accuracy: 0.9266
Epoch 9/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0124 - categorical_accuracy: 0.9292 - val_loss: 0.0124 - val_categorical_accuracy: 0.9293
Epoch 10/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0118 - categorical_accuracy: 0.9314 - val_loss: 0.0122 - val_categorical_accuracy: 0.9319
Epoch 11/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0111 - categorical_accuracy: 0.9360 - val_loss: 0.0132 - val_categorical_accuracy: 0.9236
Epoch 12/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0111 - categorical_accuracy: 0.9361 - val_loss: 0.0118 - val_categorical_accuracy: 0.9326
Epoch 13/65535
420/420 [==============================] - 4s 8ms/step - loss: 0.0108 - categorical_accuracy: 0.9382 - val_loss: 0.0118 - val_categorical_accuracy: 0.9333
Epoch 14/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0104 - categorical_accuracy: 0.9404 - val_loss: 0.0116 - val_categorical_accuracy: 0.9353
Epoch 15/65535
420/420 [==============================] - 4s 8ms/step - loss: 0.0100 - categorical_accuracy: 0.9424 - val_loss: 0.0117 - val_categorical_accuracy: 0.9341
Epoch 16/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0099 - categorical_accuracy: 0.9423 - val_loss: 0.0117 - val_categorical_accuracy: 0.9344
Epoch 17/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0094 - categorical_accuracy: 0.9457 - val_loss: 0.0117 - val_categorical_accuracy: 0.9328
Epoch 18/65535
420/420 [==============================] - 4s 8ms/step - loss: 0.0092 - categorical_accuracy: 0.9465 - val_loss: 0.0114 - val_categorical_accuracy: 0.9388
Epoch 19/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0090 - categorical_accuracy: 0.9472 - val_loss: 0.0113 - val_categorical_accuracy: 0.9362
Epoch 20/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0093 - categorical_accuracy: 0.9472 - val_loss: 0.0111 - val_categorical_accuracy: 0.9367
Epoch 21/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0088 - categorical_accuracy: 0.9489 - val_loss: 0.0110 - val_categorical_accuracy: 0.9392
Epoch 22/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0087 - categorical_accuracy: 0.9493 - val_loss: 0.0110 - val_categorical_accuracy: 0.9387
Epoch 23/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0084 - categorical_accuracy: 0.9512 - val_loss: 0.0106 - val_categorical_accuracy: 0.9405
Epoch 24/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0084 - categorical_accuracy: 0.9504 - val_loss: 0.0112 - val_categorical_accuracy: 0.9381
Epoch 25/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0083 - categorical_accuracy: 0.9507 - val_loss: 0.0108 - val_categorical_accuracy: 0.9396
Epoch 26/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0080 - categorical_accuracy: 0.9534 - val_loss: 0.0112 - val_categorical_accuracy: 0.9383
Epoch 27/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0079 - categorical_accuracy: 0.9536 - val_loss: 0.0112 - val_categorical_accuracy: 0.9377
Epoch 28/65535
420/420 [==============================] - 4s 8ms/step - loss: 0.0076 - categorical_accuracy: 0.9555 - val_loss: 0.0106 - val_categorical_accuracy: 0.9405
Epoch 29/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0075 - categorical_accuracy: 0.9551 - val_loss: 0.0105 - val_categorical_accuracy: 0.9422

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 30/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0060 - categorical_accuracy: 0.9647 - val_loss: 0.0097 - val_categorical_accuracy: 0.9489
Epoch 31/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0053 - categorical_accuracy: 0.9686 - val_loss: 0.0097 - val_categorical_accuracy: 0.9495
Epoch 32/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0052 - categorical_accuracy: 0.9692 - val_loss: 0.0095 - val_categorical_accuracy: 0.9496
Epoch 33/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0050 - categorical_accuracy: 0.9713 - val_loss: 0.0093 - val_categorical_accuracy: 0.9487
Epoch 34/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0046 - categorical_accuracy: 0.9721 - val_loss: 0.0093 - val_categorical_accuracy: 0.9507
Epoch 35/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0046 - categorical_accuracy: 0.9737 - val_loss: 0.0095 - val_categorical_accuracy: 0.9489
Epoch 36/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0045 - categorical_accuracy: 0.9731 - val_loss: 0.0094 - val_categorical_accuracy: 0.9506
Epoch 37/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0044 - categorical_accuracy: 0.9745 - val_loss: 0.0093 - val_categorical_accuracy: 0.9511
Epoch 38/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0044 - categorical_accuracy: 0.9741 - val_loss: 0.0094 - val_categorical_accuracy: 0.9499
Epoch 39/65535
420/420 [==============================] - 4s 8ms/step - loss: 0.0043 - categorical_accuracy: 0.9749 - val_loss: 0.0093 - val_categorical_accuracy: 0.9505

Epoch 00039: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 40/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0040 - categorical_accuracy: 0.9766 - val_loss: 0.0093 - val_categorical_accuracy: 0.9504
Epoch 41/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0040 - categorical_accuracy: 0.9771 - val_loss: 0.0093 - val_categorical_accuracy: 0.9507
Epoch 42/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0039 - categorical_accuracy: 0.9776 - val_loss: 0.0093 - val_categorical_accuracy: 0.9507
Epoch 43/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0038 - categorical_accuracy: 0.9780 - val_loss: 0.0093 - val_categorical_accuracy: 0.9500
Epoch 44/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0038 - categorical_accuracy: 0.9780 - val_loss: 0.0094 - val_categorical_accuracy: 0.9502

Epoch 00044: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 45/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0037 - categorical_accuracy: 0.9781 - val_loss: 0.0094 - val_categorical_accuracy: 0.9496
Epoch 46/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0037 - categorical_accuracy: 0.9782 - val_loss: 0.0094 - val_categorical_accuracy: 0.9498
Epoch 47/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0037 - categorical_accuracy: 0.9788 - val_loss: 0.0094 - val_categorical_accuracy: 0.9501
Epoch 48/65535
420/420 [==============================] - 3s 8ms/step - loss: 0.0037 - categorical_accuracy: 0.9787 - val_loss: 0.0094 - val_categorical_accuracy: 0.9500
Epoch 49/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9785 - val_loss: 0.0094 - val_categorical_accuracy: 0.9504
Epoch 50/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9790 - val_loss: 0.0094 - val_categorical_accuracy: 0.9504
Epoch 51/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9789 - val_loss: 0.0094 - val_categorical_accuracy: 0.9510
Epoch 52/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9785 - val_loss: 0.0094 - val_categorical_accuracy: 0.9504
Epoch 53/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9792 - val_loss: 0.0095 - val_categorical_accuracy: 0.9498
Epoch 54/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9797 - val_loss: 0.0095 - val_categorical_accuracy: 0.9499
Epoch 55/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9787 - val_loss: 0.0094 - val_categorical_accuracy: 0.9503
Epoch 00055: early stopping
========= generating oof predictions 21:21:48 =========
========= generating test set predictions 21:21:49 =========
========= fitting 2 th model 21:21:57 =========
Epoch 1/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0748 - categorical_accuracy: 0.5264 - val_loss: 0.0295 - val_categorical_accuracy: 0.8307
Epoch 2/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0278 - categorical_accuracy: 0.8422 - val_loss: 0.0198 - val_categorical_accuracy: 0.8852
Epoch 3/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0210 - categorical_accuracy: 0.8807 - val_loss: 0.0175 - val_categorical_accuracy: 0.8977
Epoch 4/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0180 - categorical_accuracy: 0.8988 - val_loss: 0.0151 - val_categorical_accuracy: 0.9130
Epoch 5/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0161 - categorical_accuracy: 0.9082 - val_loss: 0.0144 - val_categorical_accuracy: 0.9149
Epoch 6/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0149 - categorical_accuracy: 0.9156 - val_loss: 0.0138 - val_categorical_accuracy: 0.9203
Epoch 7/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0135 - categorical_accuracy: 0.9225 - val_loss: 0.0132 - val_categorical_accuracy: 0.9213
Epoch 8/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0130 - categorical_accuracy: 0.9258 - val_loss: 0.0126 - val_categorical_accuracy: 0.9276
Epoch 9/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0123 - categorical_accuracy: 0.9295 - val_loss: 0.0125 - val_categorical_accuracy: 0.9273
Epoch 10/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0120 - categorical_accuracy: 0.9315 - val_loss: 0.0119 - val_categorical_accuracy: 0.9320
Epoch 11/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0113 - categorical_accuracy: 0.9356 - val_loss: 0.0123 - val_categorical_accuracy: 0.9303
Epoch 12/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0107 - categorical_accuracy: 0.9378 - val_loss: 0.0119 - val_categorical_accuracy: 0.9330
Epoch 13/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0105 - categorical_accuracy: 0.9395 - val_loss: 0.0111 - val_categorical_accuracy: 0.9361
Epoch 14/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0102 - categorical_accuracy: 0.9422 - val_loss: 0.0115 - val_categorical_accuracy: 0.9349
Epoch 15/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0101 - categorical_accuracy: 0.9425 - val_loss: 0.0105 - val_categorical_accuracy: 0.9404
Epoch 16/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0096 - categorical_accuracy: 0.9455 - val_loss: 0.0119 - val_categorical_accuracy: 0.9320
Epoch 17/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0097 - categorical_accuracy: 0.9444 - val_loss: 0.0108 - val_categorical_accuracy: 0.9364
Epoch 18/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0091 - categorical_accuracy: 0.9471 - val_loss: 0.0114 - val_categorical_accuracy: 0.9358
Epoch 19/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0090 - categorical_accuracy: 0.9478 - val_loss: 0.0111 - val_categorical_accuracy: 0.9370
Epoch 20/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0090 - categorical_accuracy: 0.9472 - val_loss: 0.0113 - val_categorical_accuracy: 0.9377
Epoch 21/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0086 - categorical_accuracy: 0.9503 - val_loss: 0.0112 - val_categorical_accuracy: 0.9358

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 22/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0070 - categorical_accuracy: 0.9588 - val_loss: 0.0099 - val_categorical_accuracy: 0.9445
Epoch 23/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0063 - categorical_accuracy: 0.9633 - val_loss: 0.0099 - val_categorical_accuracy: 0.9438
Epoch 24/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0061 - categorical_accuracy: 0.9651 - val_loss: 0.0098 - val_categorical_accuracy: 0.9459
Epoch 25/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0058 - categorical_accuracy: 0.9661 - val_loss: 0.0098 - val_categorical_accuracy: 0.9448
Epoch 26/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0056 - categorical_accuracy: 0.9674 - val_loss: 0.0098 - val_categorical_accuracy: 0.9468
Epoch 27/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0056 - categorical_accuracy: 0.9677 - val_loss: 0.0098 - val_categorical_accuracy: 0.9463
Epoch 28/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0054 - categorical_accuracy: 0.9681 - val_loss: 0.0097 - val_categorical_accuracy: 0.9475
Epoch 29/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0052 - categorical_accuracy: 0.9708 - val_loss: 0.0097 - val_categorical_accuracy: 0.9480
Epoch 30/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0051 - categorical_accuracy: 0.9702 - val_loss: 0.0097 - val_categorical_accuracy: 0.9481

Epoch 00030: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 31/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0049 - categorical_accuracy: 0.9714 - val_loss: 0.0095 - val_categorical_accuracy: 0.9499
Epoch 32/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0047 - categorical_accuracy: 0.9735 - val_loss: 0.0095 - val_categorical_accuracy: 0.9490
Epoch 33/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0047 - categorical_accuracy: 0.9732 - val_loss: 0.0094 - val_categorical_accuracy: 0.9492
Epoch 34/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0047 - categorical_accuracy: 0.9735 - val_loss: 0.0094 - val_categorical_accuracy: 0.9486
Epoch 35/65535
420/420 [==============================] - 4s 10ms/step - loss: 0.0047 - categorical_accuracy: 0.9735 - val_loss: 0.0093 - val_categorical_accuracy: 0.9494
Epoch 36/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0046 - categorical_accuracy: 0.9727 - val_loss: 0.0094 - val_categorical_accuracy: 0.9489
Epoch 37/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0046 - categorical_accuracy: 0.9738 - val_loss: 0.0093 - val_categorical_accuracy: 0.9484
Epoch 38/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0044 - categorical_accuracy: 0.9740 - val_loss: 0.0093 - val_categorical_accuracy: 0.9485
Epoch 39/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0045 - categorical_accuracy: 0.9740 - val_loss: 0.0094 - val_categorical_accuracy: 0.9489
Epoch 40/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0044 - categorical_accuracy: 0.9744 - val_loss: 0.0094 - val_categorical_accuracy: 0.9483
Epoch 41/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0044 - categorical_accuracy: 0.9742 - val_loss: 0.0094 - val_categorical_accuracy: 0.9495

Epoch 00041: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 42/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0043 - categorical_accuracy: 0.9749 - val_loss: 0.0094 - val_categorical_accuracy: 0.9497
Epoch 43/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0044 - categorical_accuracy: 0.9752 - val_loss: 0.0094 - val_categorical_accuracy: 0.9506
Epoch 44/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0043 - categorical_accuracy: 0.9750 - val_loss: 0.0094 - val_categorical_accuracy: 0.9498
Epoch 45/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0042 - categorical_accuracy: 0.9757 - val_loss: 0.0094 - val_categorical_accuracy: 0.9506
Epoch 46/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0043 - categorical_accuracy: 0.9746 - val_loss: 0.0095 - val_categorical_accuracy: 0.9499
Epoch 47/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0044 - categorical_accuracy: 0.9743 - val_loss: 0.0095 - val_categorical_accuracy: 0.9495
Epoch 48/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0043 - categorical_accuracy: 0.9745 - val_loss: 0.0095 - val_categorical_accuracy: 0.9485
Epoch 49/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0043 - categorical_accuracy: 0.9753 - val_loss: 0.0095 - val_categorical_accuracy: 0.9485
Epoch 50/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0043 - categorical_accuracy: 0.9754 - val_loss: 0.0095 - val_categorical_accuracy: 0.9494
Epoch 51/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0042 - categorical_accuracy: 0.9759 - val_loss: 0.0095 - val_categorical_accuracy: 0.9489
Epoch 52/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0043 - categorical_accuracy: 0.9756 - val_loss: 0.0094 - val_categorical_accuracy: 0.9493
Epoch 00052: early stopping
========= generating oof predictions 21:25:15 =========
========= generating test set predictions 21:25:15 =========
========= fitting 3 th model 21:25:23 =========
Epoch 1/65535
420/420 [==============================] - 4s 10ms/step - loss: 0.0759 - categorical_accuracy: 0.5140 - val_loss: 0.0314 - val_categorical_accuracy: 0.8219
Epoch 2/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0277 - categorical_accuracy: 0.8425 - val_loss: 0.0203 - val_categorical_accuracy: 0.8824
Epoch 3/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0211 - categorical_accuracy: 0.8800 - val_loss: 0.0176 - val_categorical_accuracy: 0.8987
Epoch 4/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0184 - categorical_accuracy: 0.8953 - val_loss: 0.0164 - val_categorical_accuracy: 0.9072
Epoch 5/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0164 - categorical_accuracy: 0.9072 - val_loss: 0.0148 - val_categorical_accuracy: 0.9152
Epoch 6/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0149 - categorical_accuracy: 0.9153 - val_loss: 0.0140 - val_categorical_accuracy: 0.9195
Epoch 7/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0139 - categorical_accuracy: 0.9212 - val_loss: 0.0135 - val_categorical_accuracy: 0.9227
Epoch 8/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0131 - categorical_accuracy: 0.9237 - val_loss: 0.0137 - val_categorical_accuracy: 0.9215
Epoch 9/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0124 - categorical_accuracy: 0.9281 - val_loss: 0.0129 - val_categorical_accuracy: 0.9256
Epoch 10/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0120 - categorical_accuracy: 0.9308 - val_loss: 0.0132 - val_categorical_accuracy: 0.9245
Epoch 11/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0114 - categorical_accuracy: 0.9340 - val_loss: 0.0121 - val_categorical_accuracy: 0.9318
Epoch 12/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0111 - categorical_accuracy: 0.9357 - val_loss: 0.0123 - val_categorical_accuracy: 0.9313
Epoch 13/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0105 - categorical_accuracy: 0.9384 - val_loss: 0.0121 - val_categorical_accuracy: 0.9312
Epoch 14/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0105 - categorical_accuracy: 0.9388 - val_loss: 0.0115 - val_categorical_accuracy: 0.9359
Epoch 15/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0102 - categorical_accuracy: 0.9407 - val_loss: 0.0121 - val_categorical_accuracy: 0.9312
Epoch 16/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0098 - categorical_accuracy: 0.9427 - val_loss: 0.0118 - val_categorical_accuracy: 0.9329
Epoch 17/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0095 - categorical_accuracy: 0.9449 - val_loss: 0.0116 - val_categorical_accuracy: 0.9349
Epoch 18/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0094 - categorical_accuracy: 0.9453 - val_loss: 0.0120 - val_categorical_accuracy: 0.9313
Epoch 19/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0092 - categorical_accuracy: 0.9460 - val_loss: 0.0114 - val_categorical_accuracy: 0.9369
Epoch 20/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0090 - categorical_accuracy: 0.9473 - val_loss: 0.0113 - val_categorical_accuracy: 0.9380
Epoch 21/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0088 - categorical_accuracy: 0.9479 - val_loss: 0.0116 - val_categorical_accuracy: 0.9352
Epoch 22/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0088 - categorical_accuracy: 0.9485 - val_loss: 0.0118 - val_categorical_accuracy: 0.9349
Epoch 23/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0088 - categorical_accuracy: 0.9485 - val_loss: 0.0110 - val_categorical_accuracy: 0.9394
Epoch 24/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0081 - categorical_accuracy: 0.9519 - val_loss: 0.0117 - val_categorical_accuracy: 0.9363
Epoch 25/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0078 - categorical_accuracy: 0.9539 - val_loss: 0.0109 - val_categorical_accuracy: 0.9410
Epoch 26/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0080 - categorical_accuracy: 0.9528 - val_loss: 0.0115 - val_categorical_accuracy: 0.9365
Epoch 27/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0077 - categorical_accuracy: 0.9542 - val_loss: 0.0115 - val_categorical_accuracy: 0.9361
Epoch 28/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0080 - categorical_accuracy: 0.9536 - val_loss: 0.0112 - val_categorical_accuracy: 0.9411
Epoch 29/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0077 - categorical_accuracy: 0.9545 - val_loss: 0.0112 - val_categorical_accuracy: 0.9381
Epoch 30/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0078 - categorical_accuracy: 0.9537 - val_loss: 0.0126 - val_categorical_accuracy: 0.9341
Epoch 31/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0075 - categorical_accuracy: 0.9561 - val_loss: 0.0110 - val_categorical_accuracy: 0.9406

Epoch 00031: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 32/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0061 - categorical_accuracy: 0.9645 - val_loss: 0.0100 - val_categorical_accuracy: 0.9474
Epoch 33/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0054 - categorical_accuracy: 0.9682 - val_loss: 0.0101 - val_categorical_accuracy: 0.9480
Epoch 34/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0053 - categorical_accuracy: 0.9691 - val_loss: 0.0099 - val_categorical_accuracy: 0.9471
Epoch 35/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0048 - categorical_accuracy: 0.9711 - val_loss: 0.0098 - val_categorical_accuracy: 0.9482
Epoch 36/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0048 - categorical_accuracy: 0.9716 - val_loss: 0.0098 - val_categorical_accuracy: 0.9488
Epoch 37/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0046 - categorical_accuracy: 0.9733 - val_loss: 0.0099 - val_categorical_accuracy: 0.9484
Epoch 38/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0046 - categorical_accuracy: 0.9726 - val_loss: 0.0098 - val_categorical_accuracy: 0.9498
Epoch 39/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0045 - categorical_accuracy: 0.9733 - val_loss: 0.0100 - val_categorical_accuracy: 0.9491
Epoch 40/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0044 - categorical_accuracy: 0.9742 - val_loss: 0.0099 - val_categorical_accuracy: 0.9490

Epoch 00040: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 41/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0041 - categorical_accuracy: 0.9759 - val_loss: 0.0099 - val_categorical_accuracy: 0.9503
Epoch 42/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0040 - categorical_accuracy: 0.9762 - val_loss: 0.0098 - val_categorical_accuracy: 0.9496
Epoch 43/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0040 - categorical_accuracy: 0.9766 - val_loss: 0.0098 - val_categorical_accuracy: 0.9495
Epoch 44/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0040 - categorical_accuracy: 0.9762 - val_loss: 0.0099 - val_categorical_accuracy: 0.9499
Epoch 45/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0039 - categorical_accuracy: 0.9769 - val_loss: 0.0099 - val_categorical_accuracy: 0.9488

Epoch 00045: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 46/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9776 - val_loss: 0.0098 - val_categorical_accuracy: 0.9493
Epoch 47/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9774 - val_loss: 0.0099 - val_categorical_accuracy: 0.9488
Epoch 48/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9772 - val_loss: 0.0099 - val_categorical_accuracy: 0.9494
Epoch 49/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9778 - val_loss: 0.0099 - val_categorical_accuracy: 0.9487
Epoch 50/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9780 - val_loss: 0.0099 - val_categorical_accuracy: 0.9490
Epoch 51/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9781 - val_loss: 0.0099 - val_categorical_accuracy: 0.9487
Epoch 52/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9778 - val_loss: 0.0099 - val_categorical_accuracy: 0.9487
Epoch 53/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9777 - val_loss: 0.0099 - val_categorical_accuracy: 0.9499
Epoch 00053: early stopping
========= generating oof predictions 21:28:44 =========
========= generating test set predictions 21:28:44 =========
========= fitting 4 th model 21:28:52 =========
Epoch 1/65535
420/420 [==============================] - 4s 10ms/step - loss: 0.0749 - categorical_accuracy: 0.5254 - val_loss: 0.0298 - val_categorical_accuracy: 0.8327
Epoch 2/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0276 - categorical_accuracy: 0.8440 - val_loss: 0.0206 - val_categorical_accuracy: 0.8819
Epoch 3/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0209 - categorical_accuracy: 0.8822 - val_loss: 0.0172 - val_categorical_accuracy: 0.9019
Epoch 4/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0176 - categorical_accuracy: 0.9004 - val_loss: 0.0158 - val_categorical_accuracy: 0.9077
Epoch 5/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0159 - categorical_accuracy: 0.9095 - val_loss: 0.0151 - val_categorical_accuracy: 0.9134
Epoch 6/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0146 - categorical_accuracy: 0.9173 - val_loss: 0.0146 - val_categorical_accuracy: 0.9170
Epoch 7/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0137 - categorical_accuracy: 0.9223 - val_loss: 0.0136 - val_categorical_accuracy: 0.9225
Epoch 8/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0131 - categorical_accuracy: 0.9255 - val_loss: 0.0131 - val_categorical_accuracy: 0.9251
Epoch 9/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0123 - categorical_accuracy: 0.9303 - val_loss: 0.0133 - val_categorical_accuracy: 0.9236
Epoch 10/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0117 - categorical_accuracy: 0.9327 - val_loss: 0.0126 - val_categorical_accuracy: 0.9279
Epoch 11/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0113 - categorical_accuracy: 0.9348 - val_loss: 0.0126 - val_categorical_accuracy: 0.9286
Epoch 12/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0110 - categorical_accuracy: 0.9366 - val_loss: 0.0124 - val_categorical_accuracy: 0.9294
Epoch 13/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0106 - categorical_accuracy: 0.9386 - val_loss: 0.0126 - val_categorical_accuracy: 0.9282
Epoch 14/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0102 - categorical_accuracy: 0.9409 - val_loss: 0.0122 - val_categorical_accuracy: 0.9315
Epoch 15/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0099 - categorical_accuracy: 0.9432 - val_loss: 0.0118 - val_categorical_accuracy: 0.9330
Epoch 16/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0098 - categorical_accuracy: 0.9432 - val_loss: 0.0116 - val_categorical_accuracy: 0.9334
Epoch 17/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0094 - categorical_accuracy: 0.9446 - val_loss: 0.0121 - val_categorical_accuracy: 0.9358
Epoch 18/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0093 - categorical_accuracy: 0.9456 - val_loss: 0.0120 - val_categorical_accuracy: 0.9337
Epoch 19/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0091 - categorical_accuracy: 0.9473 - val_loss: 0.0116 - val_categorical_accuracy: 0.9350
Epoch 20/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0089 - categorical_accuracy: 0.9478 - val_loss: 0.0114 - val_categorical_accuracy: 0.9346
Epoch 21/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0088 - categorical_accuracy: 0.9486 - val_loss: 0.0118 - val_categorical_accuracy: 0.9347
Epoch 22/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0083 - categorical_accuracy: 0.9519 - val_loss: 0.0117 - val_categorical_accuracy: 0.9363
Epoch 23/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0084 - categorical_accuracy: 0.9518 - val_loss: 0.0116 - val_categorical_accuracy: 0.9369
Epoch 24/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0084 - categorical_accuracy: 0.9510 - val_loss: 0.0114 - val_categorical_accuracy: 0.9372
Epoch 25/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0081 - categorical_accuracy: 0.9521 - val_loss: 0.0117 - val_categorical_accuracy: 0.9374
Epoch 26/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0080 - categorical_accuracy: 0.9533 - val_loss: 0.0119 - val_categorical_accuracy: 0.9370

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 27/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0065 - categorical_accuracy: 0.9617 - val_loss: 0.0104 - val_categorical_accuracy: 0.9434
Epoch 28/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0059 - categorical_accuracy: 0.9652 - val_loss: 0.0101 - val_categorical_accuracy: 0.9457
Epoch 29/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0056 - categorical_accuracy: 0.9670 - val_loss: 0.0101 - val_categorical_accuracy: 0.9465
Epoch 30/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0054 - categorical_accuracy: 0.9681 - val_loss: 0.0100 - val_categorical_accuracy: 0.9475
Epoch 31/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0051 - categorical_accuracy: 0.9705 - val_loss: 0.0100 - val_categorical_accuracy: 0.9486
Epoch 32/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0051 - categorical_accuracy: 0.9696 - val_loss: 0.0100 - val_categorical_accuracy: 0.9464
Epoch 33/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0049 - categorical_accuracy: 0.9710 - val_loss: 0.0100 - val_categorical_accuracy: 0.9469
Epoch 34/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0049 - categorical_accuracy: 0.9716 - val_loss: 0.0100 - val_categorical_accuracy: 0.9469
Epoch 35/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0047 - categorical_accuracy: 0.9723 - val_loss: 0.0098 - val_categorical_accuracy: 0.9483
Epoch 36/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0046 - categorical_accuracy: 0.9726 - val_loss: 0.0099 - val_categorical_accuracy: 0.9489
Epoch 37/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0045 - categorical_accuracy: 0.9731 - val_loss: 0.0099 - val_categorical_accuracy: 0.9493
Epoch 38/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0044 - categorical_accuracy: 0.9738 - val_loss: 0.0100 - val_categorical_accuracy: 0.9489
Epoch 39/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0044 - categorical_accuracy: 0.9729 - val_loss: 0.0100 - val_categorical_accuracy: 0.9485
Epoch 40/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0043 - categorical_accuracy: 0.9752 - val_loss: 0.0101 - val_categorical_accuracy: 0.9490
Epoch 41/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0042 - categorical_accuracy: 0.9746 - val_loss: 0.0100 - val_categorical_accuracy: 0.9485

Epoch 00041: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 42/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0040 - categorical_accuracy: 0.9762 - val_loss: 0.0098 - val_categorical_accuracy: 0.9490
Epoch 43/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0039 - categorical_accuracy: 0.9769 - val_loss: 0.0097 - val_categorical_accuracy: 0.9492
Epoch 44/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0039 - categorical_accuracy: 0.9776 - val_loss: 0.0097 - val_categorical_accuracy: 0.9504
Epoch 45/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9777 - val_loss: 0.0097 - val_categorical_accuracy: 0.9506
Epoch 46/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9777 - val_loss: 0.0097 - val_categorical_accuracy: 0.9511

Epoch 00046: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 47/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9775 - val_loss: 0.0097 - val_categorical_accuracy: 0.9504
Epoch 48/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9774 - val_loss: 0.0097 - val_categorical_accuracy: 0.9511
Epoch 49/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9781 - val_loss: 0.0097 - val_categorical_accuracy: 0.9509
Epoch 50/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9787 - val_loss: 0.0097 - val_categorical_accuracy: 0.9513
Epoch 51/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9784 - val_loss: 0.0097 - val_categorical_accuracy: 0.9504
Epoch 52/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9782 - val_loss: 0.0097 - val_categorical_accuracy: 0.9510
Epoch 53/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9791 - val_loss: 0.0097 - val_categorical_accuracy: 0.9512
Epoch 54/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9785 - val_loss: 0.0097 - val_categorical_accuracy: 0.9517
Epoch 55/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9784 - val_loss: 0.0097 - val_categorical_accuracy: 0.9508
Epoch 56/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9788 - val_loss: 0.0097 - val_categorical_accuracy: 0.9516
Epoch 57/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9789 - val_loss: 0.0097 - val_categorical_accuracy: 0.9513
Epoch 58/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9785 - val_loss: 0.0097 - val_categorical_accuracy: 0.9516
Epoch 59/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9790 - val_loss: 0.0097 - val_categorical_accuracy: 0.9514
Epoch 60/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0035 - categorical_accuracy: 0.9790 - val_loss: 0.0096 - val_categorical_accuracy: 0.9522
Epoch 61/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0035 - categorical_accuracy: 0.9788 - val_loss: 0.0096 - val_categorical_accuracy: 0.9522
Epoch 62/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0035 - categorical_accuracy: 0.9795 - val_loss: 0.0097 - val_categorical_accuracy: 0.9513
Epoch 63/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9786 - val_loss: 0.0097 - val_categorical_accuracy: 0.9512
Epoch 64/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9787 - val_loss: 0.0097 - val_categorical_accuracy: 0.9514
Epoch 65/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0036 - categorical_accuracy: 0.9787 - val_loss: 0.0097 - val_categorical_accuracy: 0.9510
Epoch 66/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0035 - categorical_accuracy: 0.9792 - val_loss: 0.0098 - val_categorical_accuracy: 0.9513
Epoch 67/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0035 - categorical_accuracy: 0.9804 - val_loss: 0.0097 - val_categorical_accuracy: 0.9516
Epoch 68/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0035 - categorical_accuracy: 0.9794 - val_loss: 0.0098 - val_categorical_accuracy: 0.9514
Epoch 69/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0035 - categorical_accuracy: 0.9797 - val_loss: 0.0098 - val_categorical_accuracy: 0.9515
Epoch 70/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0035 - categorical_accuracy: 0.9795 - val_loss: 0.0098 - val_categorical_accuracy: 0.9513
Epoch 71/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0034 - categorical_accuracy: 0.9799 - val_loss: 0.0098 - val_categorical_accuracy: 0.9513
Epoch 72/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0034 - categorical_accuracy: 0.9800 - val_loss: 0.0098 - val_categorical_accuracy: 0.9511
Epoch 73/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0034 - categorical_accuracy: 0.9800 - val_loss: 0.0098 - val_categorical_accuracy: 0.9506
Epoch 74/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0034 - categorical_accuracy: 0.9801 - val_loss: 0.0098 - val_categorical_accuracy: 0.9507
Epoch 75/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0034 - categorical_accuracy: 0.9805 - val_loss: 0.0098 - val_categorical_accuracy: 0.9511
Epoch 76/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0034 - categorical_accuracy: 0.9798 - val_loss: 0.0098 - val_categorical_accuracy: 0.9506
Epoch 00076: early stopping
========= generating oof predictions 21:33:42 =========
========= generating test set predictions 21:33:43 =========
========= fitting 5 th model 21:33:50 =========
Epoch 1/65535
420/420 [==============================] - 4s 10ms/step - loss: 0.0793 - categorical_accuracy: 0.4889 - val_loss: 0.0309 - val_categorical_accuracy: 0.8238
Epoch 2/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0282 - categorical_accuracy: 0.8383 - val_loss: 0.0217 - val_categorical_accuracy: 0.8738
Epoch 3/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0215 - categorical_accuracy: 0.8792 - val_loss: 0.0176 - val_categorical_accuracy: 0.9028
Epoch 4/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0181 - categorical_accuracy: 0.8983 - val_loss: 0.0163 - val_categorical_accuracy: 0.9071
Epoch 5/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0163 - categorical_accuracy: 0.9070 - val_loss: 0.0144 - val_categorical_accuracy: 0.9172
Epoch 6/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0148 - categorical_accuracy: 0.9148 - val_loss: 0.0140 - val_categorical_accuracy: 0.9199
Epoch 7/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0136 - categorical_accuracy: 0.9217 - val_loss: 0.0131 - val_categorical_accuracy: 0.9254
Epoch 8/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0130 - categorical_accuracy: 0.9256 - val_loss: 0.0135 - val_categorical_accuracy: 0.9248
Epoch 9/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0122 - categorical_accuracy: 0.9312 - val_loss: 0.0128 - val_categorical_accuracy: 0.9275
Epoch 10/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0119 - categorical_accuracy: 0.9314 - val_loss: 0.0126 - val_categorical_accuracy: 0.9288
Epoch 11/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0117 - categorical_accuracy: 0.9328 - val_loss: 0.0130 - val_categorical_accuracy: 0.9284
Epoch 12/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0107 - categorical_accuracy: 0.9376 - val_loss: 0.0122 - val_categorical_accuracy: 0.9304
Epoch 13/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0104 - categorical_accuracy: 0.9408 - val_loss: 0.0120 - val_categorical_accuracy: 0.9313
Epoch 14/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0101 - categorical_accuracy: 0.9424 - val_loss: 0.0123 - val_categorical_accuracy: 0.9320
Epoch 15/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0097 - categorical_accuracy: 0.9453 - val_loss: 0.0115 - val_categorical_accuracy: 0.9354
Epoch 16/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0097 - categorical_accuracy: 0.9451 - val_loss: 0.0122 - val_categorical_accuracy: 0.9339
Epoch 17/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0093 - categorical_accuracy: 0.9468 - val_loss: 0.0117 - val_categorical_accuracy: 0.9333
Epoch 18/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0091 - categorical_accuracy: 0.9475 - val_loss: 0.0114 - val_categorical_accuracy: 0.9366
Epoch 19/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0088 - categorical_accuracy: 0.9486 - val_loss: 0.0114 - val_categorical_accuracy: 0.9351
Epoch 20/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0087 - categorical_accuracy: 0.9496 - val_loss: 0.0113 - val_categorical_accuracy: 0.9368
Epoch 21/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0087 - categorical_accuracy: 0.9499 - val_loss: 0.0115 - val_categorical_accuracy: 0.9345
Epoch 22/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0083 - categorical_accuracy: 0.9515 - val_loss: 0.0113 - val_categorical_accuracy: 0.9375
Epoch 23/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0081 - categorical_accuracy: 0.9530 - val_loss: 0.0110 - val_categorical_accuracy: 0.9384
Epoch 24/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0081 - categorical_accuracy: 0.9523 - val_loss: 0.0112 - val_categorical_accuracy: 0.9401
Epoch 25/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0079 - categorical_accuracy: 0.9535 - val_loss: 0.0118 - val_categorical_accuracy: 0.9348
Epoch 26/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0077 - categorical_accuracy: 0.9549 - val_loss: 0.0112 - val_categorical_accuracy: 0.9383
Epoch 27/65535
420/420 [==============================] - 4s 10ms/step - loss: 0.0078 - categorical_accuracy: 0.9552 - val_loss: 0.0115 - val_categorical_accuracy: 0.9369
Epoch 28/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0074 - categorical_accuracy: 0.9569 - val_loss: 0.0112 - val_categorical_accuracy: 0.9380
Epoch 29/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0076 - categorical_accuracy: 0.9558 - val_loss: 0.0114 - val_categorical_accuracy: 0.9363

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 30/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0061 - categorical_accuracy: 0.9640 - val_loss: 0.0098 - val_categorical_accuracy: 0.9456
Epoch 31/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0055 - categorical_accuracy: 0.9678 - val_loss: 0.0095 - val_categorical_accuracy: 0.9477
Epoch 32/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0051 - categorical_accuracy: 0.9698 - val_loss: 0.0094 - val_categorical_accuracy: 0.9489
Epoch 33/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0049 - categorical_accuracy: 0.9709 - val_loss: 0.0096 - val_categorical_accuracy: 0.9483
Epoch 34/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0047 - categorical_accuracy: 0.9724 - val_loss: 0.0095 - val_categorical_accuracy: 0.9495
Epoch 35/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0046 - categorical_accuracy: 0.9729 - val_loss: 0.0096 - val_categorical_accuracy: 0.9483
Epoch 36/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0045 - categorical_accuracy: 0.9735 - val_loss: 0.0097 - val_categorical_accuracy: 0.9468
Epoch 37/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0044 - categorical_accuracy: 0.9741 - val_loss: 0.0096 - val_categorical_accuracy: 0.9477
Epoch 38/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0043 - categorical_accuracy: 0.9740 - val_loss: 0.0098 - val_categorical_accuracy: 0.9468

Epoch 00038: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 39/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0041 - categorical_accuracy: 0.9761 - val_loss: 0.0096 - val_categorical_accuracy: 0.9484
Epoch 40/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0041 - categorical_accuracy: 0.9760 - val_loss: 0.0095 - val_categorical_accuracy: 0.9482
Epoch 41/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0040 - categorical_accuracy: 0.9767 - val_loss: 0.0095 - val_categorical_accuracy: 0.9480
Epoch 42/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0039 - categorical_accuracy: 0.9774 - val_loss: 0.0095 - val_categorical_accuracy: 0.9477
Epoch 43/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0039 - categorical_accuracy: 0.9770 - val_loss: 0.0095 - val_categorical_accuracy: 0.9491

Epoch 00043: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 44/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9774 - val_loss: 0.0095 - val_categorical_accuracy: 0.9486
Epoch 45/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9778 - val_loss: 0.0095 - val_categorical_accuracy: 0.9492
Epoch 46/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0038 - categorical_accuracy: 0.9773 - val_loss: 0.0096 - val_categorical_accuracy: 0.9489
Epoch 47/65535
420/420 [==============================] - 4s 9ms/step - loss: 0.0037 - categorical_accuracy: 0.9784 - val_loss: 0.0095 - val_categorical_accuracy: 0.9488
Epoch 00047: early stopping
========= generating oof predictions 21:36:48 =========
========= generating test set predictions 21:36:49 =========
train loss avg 0.003755070107964719 -- std 0.0002931278293488388, val loss avg 0.009620129596148774 -- std 0.00018584616141675652
train acc avg 0.9780246654653861 -- std 0.0014068966372797198, val acc avg 0.9497771517189488 -- std 0.0006566934062390121
mean nb epochs 56.6
dump oof predicted probs
dump test set predicted probs
