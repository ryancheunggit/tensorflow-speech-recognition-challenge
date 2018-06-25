ren (master *+) python $ python train.py logmelspectrogram_40_25_10 arm_dscnn l 128
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
conv2d_1 (Conv2D)            (None, 51, 40, 276)       11316
_________________________________________________________________
batch_normalization_1 (Batch (None, 51, 40, 276)       1104
_________________________________________________________________
activation_1 (Activation)    (None, 51, 40, 276)       0
_________________________________________________________________
dropout_1 (Dropout)          (None, 51, 40, 276)       0
_________________________________________________________________
separable_conv2d_1 (Separabl (None, 51, 40, 276)       78936
_________________________________________________________________
batch_normalization_2 (Batch (None, 51, 40, 276)       1104
_________________________________________________________________
activation_2 (Activation)    (None, 51, 40, 276)       0
_________________________________________________________________
dropout_2 (Dropout)          (None, 51, 40, 276)       0
_________________________________________________________________
separable_conv2d_2 (Separabl (None, 51, 40, 276)       78936
_________________________________________________________________
batch_normalization_3 (Batch (None, 51, 40, 276)       1104
_________________________________________________________________
activation_3 (Activation)    (None, 51, 40, 276)       0
_________________________________________________________________
dropout_3 (Dropout)          (None, 51, 40, 276)       0
_________________________________________________________________
separable_conv2d_3 (Separabl (None, 51, 40, 276)       78936
_________________________________________________________________
batch_normalization_4 (Batch (None, 51, 40, 276)       1104
_________________________________________________________________
activation_4 (Activation)    (None, 51, 40, 276)       0
_________________________________________________________________
dropout_4 (Dropout)          (None, 51, 40, 276)       0
_________________________________________________________________
separable_conv2d_4 (Separabl (None, 51, 40, 276)       78936
_________________________________________________________________
batch_normalization_5 (Batch (None, 51, 40, 276)       1104
_________________________________________________________________
activation_5 (Activation)    (None, 51, 40, 276)       0
_________________________________________________________________
dropout_5 (Dropout)          (None, 51, 40, 276)       0
_________________________________________________________________
global_average_pooling2d_1 ( (None, 276)               0
_________________________________________________________________
dense_1 (Dense)              (None, 31)                8587
=================================================================
Total params: 341,167
Trainable params: 338,407
Non-trainable params: 2,760
_________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 20:57:31 =========
Epoch 1/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.1090 - categorical_accuracy: 0.3590 - val_loss: 0.0810 - val_categorical_accuracy: 0.4924
Epoch 2/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0545 - categorical_accuracy: 0.7746 - val_loss: 0.0527 - val_categorical_accuracy: 0.6928
Epoch 3/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0310 - categorical_accuracy: 0.8748 - val_loss: 0.0296 - val_categorical_accuracy: 0.8460
Epoch 4/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0222 - categorical_accuracy: 0.9054 - val_loss: 0.0202 - val_categorical_accuracy: 0.8958
Epoch 5/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0180 - categorical_accuracy: 0.9196 - val_loss: 0.0193 - val_categorical_accuracy: 0.8949
Epoch 6/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0152 - categorical_accuracy: 0.9294 - val_loss: 0.0219 - val_categorical_accuracy: 0.8783
Epoch 7/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0135 - categorical_accuracy: 0.9365 - val_loss: 0.0171 - val_categorical_accuracy: 0.9074
Epoch 8/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0122 - categorical_accuracy: 0.9418 - val_loss: 0.0176 - val_categorical_accuracy: 0.9037
Epoch 9/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0112 - categorical_accuracy: 0.9466 - val_loss: 0.0194 - val_categorical_accuracy: 0.8906
Epoch 10/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0103 - categorical_accuracy: 0.9502 - val_loss: 0.0234 - val_categorical_accuracy: 0.8704
Epoch 11/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0097 - categorical_accuracy: 0.9527 - val_loss: 0.0222 - val_categorical_accuracy: 0.8801
Epoch 12/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0091 - categorical_accuracy: 0.9555 - val_loss: 0.0256 - val_categorical_accuracy: 0.8612
Epoch 13/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0086 - categorical_accuracy: 0.9577 - val_loss: 0.0177 - val_categorical_accuracy: 0.9048

Epoch 00013: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 14/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0069 - categorical_accuracy: 0.9678 - val_loss: 0.0093 - val_categorical_accuracy: 0.9495
Epoch 15/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0065 - categorical_accuracy: 0.9694 - val_loss: 0.0087 - val_categorical_accuracy: 0.9525
Epoch 16/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0064 - categorical_accuracy: 0.9704 - val_loss: 0.0092 - val_categorical_accuracy: 0.9487
Epoch 17/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0062 - categorical_accuracy: 0.9714 - val_loss: 0.0097 - val_categorical_accuracy: 0.9488
Epoch 18/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0061 - categorical_accuracy: 0.9717 - val_loss: 0.0102 - val_categorical_accuracy: 0.9430
Epoch 19/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0059 - categorical_accuracy: 0.9727 - val_loss: 0.0092 - val_categorical_accuracy: 0.9501
Epoch 20/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0058 - categorical_accuracy: 0.9736 - val_loss: 0.0087 - val_categorical_accuracy: 0.9524
Epoch 21/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0057 - categorical_accuracy: 0.9739 - val_loss: 0.0091 - val_categorical_accuracy: 0.9507

Epoch 00021: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 22/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0054 - categorical_accuracy: 0.9755 - val_loss: 0.0075 - val_categorical_accuracy: 0.9597
Epoch 23/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0052 - categorical_accuracy: 0.9769 - val_loss: 0.0074 - val_categorical_accuracy: 0.9604
Epoch 24/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0052 - categorical_accuracy: 0.9772 - val_loss: 0.0074 - val_categorical_accuracy: 0.9606
Epoch 25/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0052 - categorical_accuracy: 0.9769 - val_loss: 0.0074 - val_categorical_accuracy: 0.9600
Epoch 26/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0052 - categorical_accuracy: 0.9772 - val_loss: 0.0073 - val_categorical_accuracy: 0.9612
Epoch 27/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0051 - categorical_accuracy: 0.9773 - val_loss: 0.0075 - val_categorical_accuracy: 0.9611
Epoch 28/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0051 - categorical_accuracy: 0.9778 - val_loss: 0.0074 - val_categorical_accuracy: 0.9601
Epoch 29/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0051 - categorical_accuracy: 0.9778 - val_loss: 0.0074 - val_categorical_accuracy: 0.9608
Epoch 30/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0051 - categorical_accuracy: 0.9775 - val_loss: 0.0075 - val_categorical_accuracy: 0.9601
Epoch 31/65535
420/420 [==============================] - 225s 537ms/step - loss: 0.0050 - categorical_accuracy: 0.9779 - val_loss: 0.0074 - val_categorical_accuracy: 0.9606

Epoch 00031: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 32/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0050 - categorical_accuracy: 0.9777 - val_loss: 0.0072 - val_categorical_accuracy: 0.9609
Epoch 33/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0049 - categorical_accuracy: 0.9785 - val_loss: 0.0072 - val_categorical_accuracy: 0.9606
Epoch 34/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0049 - categorical_accuracy: 0.9786 - val_loss: 0.0073 - val_categorical_accuracy: 0.9605
Epoch 35/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0049 - categorical_accuracy: 0.9785 - val_loss: 0.0072 - val_categorical_accuracy: 0.9611
Epoch 36/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0049 - categorical_accuracy: 0.9782 - val_loss: 0.0073 - val_categorical_accuracy: 0.9606
Epoch 37/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0049 - categorical_accuracy: 0.9785 - val_loss: 0.0072 - val_categorical_accuracy: 0.9609
Epoch 38/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0049 - categorical_accuracy: 0.9787 - val_loss: 0.0073 - val_categorical_accuracy: 0.9607
Epoch 39/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0049 - categorical_accuracy: 0.9787 - val_loss: 0.0072 - val_categorical_accuracy: 0.9609
Epoch 40/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0049 - categorical_accuracy: 0.9786 - val_loss: 0.0072 - val_categorical_accuracy: 0.9609
Epoch 41/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0049 - categorical_accuracy: 0.9790 - val_loss: 0.0073 - val_categorical_accuracy: 0.9605
Epoch 42/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0048 - categorical_accuracy: 0.9789 - val_loss: 0.0072 - val_categorical_accuracy: 0.9614
Epoch 43/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0048 - categorical_accuracy: 0.9790 - val_loss: 0.0072 - val_categorical_accuracy: 0.9606
Epoch 44/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0048 - categorical_accuracy: 0.9789 - val_loss: 0.0072 - val_categorical_accuracy: 0.9610
Epoch 45/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0048 - categorical_accuracy: 0.9788 - val_loss: 0.0072 - val_categorical_accuracy: 0.9611
Epoch 46/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0048 - categorical_accuracy: 0.9789 - val_loss: 0.0072 - val_categorical_accuracy: 0.9605
Epoch 47/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0048 - categorical_accuracy: 0.9790 - val_loss: 0.0072 - val_categorical_accuracy: 0.9612
Epoch 48/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0048 - categorical_accuracy: 0.9791 - val_loss: 0.0072 - val_categorical_accuracy: 0.9605
Epoch 49/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0048 - categorical_accuracy: 0.9794 - val_loss: 0.0072 - val_categorical_accuracy: 0.9614
Epoch 50/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0047 - categorical_accuracy: 0.9795 - val_loss: 0.0072 - val_categorical_accuracy: 0.9611
Epoch 51/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0047 - categorical_accuracy: 0.9796 - val_loss: 0.0072 - val_categorical_accuracy: 0.9612
Epoch 52/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0047 - categorical_accuracy: 0.9794 - val_loss: 0.0071 - val_categorical_accuracy: 0.9615
Epoch 53/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0047 - categorical_accuracy: 0.9797 - val_loss: 0.0072 - val_categorical_accuracy: 0.9609
Epoch 54/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0047 - categorical_accuracy: 0.9794 - val_loss: 0.0072 - val_categorical_accuracy: 0.9615
Epoch 55/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0047 - categorical_accuracy: 0.9798 - val_loss: 0.0072 - val_categorical_accuracy: 0.9611
Epoch 56/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0047 - categorical_accuracy: 0.9795 - val_loss: 0.0073 - val_categorical_accuracy: 0.9603
Epoch 57/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0046 - categorical_accuracy: 0.9799 - val_loss: 0.0072 - val_categorical_accuracy: 0.9609
Epoch 58/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0047 - categorical_accuracy: 0.9799 - val_loss: 0.0073 - val_categorical_accuracy: 0.9610
Epoch 59/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0047 - categorical_accuracy: 0.9797 - val_loss: 0.0072 - val_categorical_accuracy: 0.9617
Epoch 60/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0046 - categorical_accuracy: 0.9802 - val_loss: 0.0072 - val_categorical_accuracy: 0.9612
Epoch 61/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0046 - categorical_accuracy: 0.9797 - val_loss: 0.0072 - val_categorical_accuracy: 0.9614
Epoch 62/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0046 - categorical_accuracy: 0.9803 - val_loss: 0.0072 - val_categorical_accuracy: 0.9612
Epoch 63/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0046 - categorical_accuracy: 0.9806 - val_loss: 0.0072 - val_categorical_accuracy: 0.9612
Epoch 64/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0046 - categorical_accuracy: 0.9804 - val_loss: 0.0072 - val_categorical_accuracy: 0.9613
Epoch 65/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0046 - categorical_accuracy: 0.9803 - val_loss: 0.0072 - val_categorical_accuracy: 0.9611
Epoch 66/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0046 - categorical_accuracy: 0.9803 - val_loss: 0.0072 - val_categorical_accuracy: 0.9613
Epoch 67/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0046 - categorical_accuracy: 0.9803 - val_loss: 0.0072 - val_categorical_accuracy: 0.9610
Epoch 00067: early stopping
========= generating oof predictions 01:07:22 =========
========= generating test set predictions 01:07:33 =========
========= fitting 2 th model 01:09:51 =========
Epoch 1/65535
420/420 [==============================] - 216s 514ms/step - loss: 0.1090 - categorical_accuracy: 0.3580 - val_loss: 0.1334 - val_categorical_accuracy: 0.1917
Epoch 2/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0556 - categorical_accuracy: 0.7653 - val_loss: 0.0429 - val_categorical_accuracy: 0.7810
Epoch 3/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0319 - categorical_accuracy: 0.8704 - val_loss: 0.0363 - val_categorical_accuracy: 0.7951
Epoch 4/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0232 - categorical_accuracy: 0.9002 - val_loss: 0.0334 - val_categorical_accuracy: 0.8026
Epoch 5/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0185 - categorical_accuracy: 0.9171 - val_loss: 0.0258 - val_categorical_accuracy: 0.8515
Epoch 6/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0159 - categorical_accuracy: 0.9277 - val_loss: 0.0196 - val_categorical_accuracy: 0.8939
Epoch 7/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0141 - categorical_accuracy: 0.9343 - val_loss: 0.0213 - val_categorical_accuracy: 0.8803
Epoch 8/65535
420/420 [==============================] - 225s 536ms/step - loss: 0.0127 - categorical_accuracy: 0.9400 - val_loss: 0.0192 - val_categorical_accuracy: 0.8973
Epoch 9/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0116 - categorical_accuracy: 0.9442 - val_loss: 0.0248 - val_categorical_accuracy: 0.8696
Epoch 10/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0108 - categorical_accuracy: 0.9479 - val_loss: 0.0193 - val_categorical_accuracy: 0.8965
Epoch 11/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0100 - categorical_accuracy: 0.9510 - val_loss: 0.0201 - val_categorical_accuracy: 0.8864
Epoch 12/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0093 - categorical_accuracy: 0.9547 - val_loss: 0.0194 - val_categorical_accuracy: 0.8936
Epoch 13/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0089 - categorical_accuracy: 0.9569 - val_loss: 0.0202 - val_categorical_accuracy: 0.8891
Epoch 14/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0084 - categorical_accuracy: 0.9592 - val_loss: 0.0286 - val_categorical_accuracy: 0.8432

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 15/65535
420/420 [==============================] - 225s 536ms/step - loss: 0.0067 - categorical_accuracy: 0.9678 - val_loss: 0.0089 - val_categorical_accuracy: 0.9506
Epoch 16/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0064 - categorical_accuracy: 0.9703 - val_loss: 0.0093 - val_categorical_accuracy: 0.9483
Epoch 17/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0062 - categorical_accuracy: 0.9709 - val_loss: 0.0092 - val_categorical_accuracy: 0.9494
Epoch 18/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0061 - categorical_accuracy: 0.9718 - val_loss: 0.0088 - val_categorical_accuracy: 0.9527
Epoch 19/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0060 - categorical_accuracy: 0.9716 - val_loss: 0.0097 - val_categorical_accuracy: 0.9475
Epoch 20/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0059 - categorical_accuracy: 0.9733 - val_loss: 0.0099 - val_categorical_accuracy: 0.9457
Epoch 21/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0058 - categorical_accuracy: 0.9730 - val_loss: 0.0102 - val_categorical_accuracy: 0.9446
Epoch 22/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0056 - categorical_accuracy: 0.9743 - val_loss: 0.0097 - val_categorical_accuracy: 0.9468
Epoch 23/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0055 - categorical_accuracy: 0.9747 - val_loss: 0.0108 - val_categorical_accuracy: 0.9414
Epoch 24/65535
420/420 [==============================] - 224s 535ms/step - loss: 0.0055 - categorical_accuracy: 0.9750 - val_loss: 0.0093 - val_categorical_accuracy: 0.9505

Epoch 00024: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 25/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0051 - categorical_accuracy: 0.9768 - val_loss: 0.0073 - val_categorical_accuracy: 0.9593
Epoch 26/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0050 - categorical_accuracy: 0.9774 - val_loss: 0.0072 - val_categorical_accuracy: 0.9596
Epoch 27/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0050 - categorical_accuracy: 0.9780 - val_loss: 0.0072 - val_categorical_accuracy: 0.9605
Epoch 28/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0049 - categorical_accuracy: 0.9780 - val_loss: 0.0072 - val_categorical_accuracy: 0.9597
Epoch 29/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0049 - categorical_accuracy: 0.9785 - val_loss: 0.0071 - val_categorical_accuracy: 0.9601
Epoch 30/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0049 - categorical_accuracy: 0.9782 - val_loss: 0.0072 - val_categorical_accuracy: 0.9606
Epoch 31/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0049 - categorical_accuracy: 0.9777 - val_loss: 0.0072 - val_categorical_accuracy: 0.9600
Epoch 32/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0049 - categorical_accuracy: 0.9780 - val_loss: 0.0072 - val_categorical_accuracy: 0.9601
Epoch 33/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0048 - categorical_accuracy: 0.9783 - val_loss: 0.0073 - val_categorical_accuracy: 0.9591
Epoch 34/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0048 - categorical_accuracy: 0.9787 - val_loss: 0.0072 - val_categorical_accuracy: 0.9600
Epoch 35/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0048 - categorical_accuracy: 0.9789 - val_loss: 0.0072 - val_categorical_accuracy: 0.9606

Epoch 00035: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 36/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0047 - categorical_accuracy: 0.9791 - val_loss: 0.0070 - val_categorical_accuracy: 0.9611
Epoch 37/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0047 - categorical_accuracy: 0.9792 - val_loss: 0.0070 - val_categorical_accuracy: 0.9608
Epoch 38/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0047 - categorical_accuracy: 0.9792 - val_loss: 0.0070 - val_categorical_accuracy: 0.9607
Epoch 39/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0047 - categorical_accuracy: 0.9790 - val_loss: 0.0070 - val_categorical_accuracy: 0.9609
Epoch 40/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0047 - categorical_accuracy: 0.9790 - val_loss: 0.0070 - val_categorical_accuracy: 0.9611
Epoch 41/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0047 - categorical_accuracy: 0.9799 - val_loss: 0.0070 - val_categorical_accuracy: 0.9611
Epoch 42/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0046 - categorical_accuracy: 0.9797 - val_loss: 0.0070 - val_categorical_accuracy: 0.9607
Epoch 43/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0046 - categorical_accuracy: 0.9798 - val_loss: 0.0070 - val_categorical_accuracy: 0.9608
Epoch 44/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0046 - categorical_accuracy: 0.9795 - val_loss: 0.0070 - val_categorical_accuracy: 0.9611
Epoch 45/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0046 - categorical_accuracy: 0.9793 - val_loss: 0.0070 - val_categorical_accuracy: 0.9610
Epoch 46/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0046 - categorical_accuracy: 0.9799 - val_loss: 0.0070 - val_categorical_accuracy: 0.9611
Epoch 47/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0046 - categorical_accuracy: 0.9800 - val_loss: 0.0070 - val_categorical_accuracy: 0.9614
Epoch 48/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0046 - categorical_accuracy: 0.9800 - val_loss: 0.0069 - val_categorical_accuracy: 0.9608
Epoch 49/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0046 - categorical_accuracy: 0.9796 - val_loss: 0.0070 - val_categorical_accuracy: 0.9608
Epoch 50/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0045 - categorical_accuracy: 0.9799 - val_loss: 0.0069 - val_categorical_accuracy: 0.9608
Epoch 51/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0046 - categorical_accuracy: 0.9798 - val_loss: 0.0069 - val_categorical_accuracy: 0.9613
Epoch 52/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0045 - categorical_accuracy: 0.9797 - val_loss: 0.0070 - val_categorical_accuracy: 0.9610
Epoch 53/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0045 - categorical_accuracy: 0.9801 - val_loss: 0.0069 - val_categorical_accuracy: 0.9611
Epoch 54/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0045 - categorical_accuracy: 0.9803 - val_loss: 0.0069 - val_categorical_accuracy: 0.9612
Epoch 55/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0045 - categorical_accuracy: 0.9804 - val_loss: 0.0069 - val_categorical_accuracy: 0.9609
Epoch 56/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0045 - categorical_accuracy: 0.9800 - val_loss: 0.0069 - val_categorical_accuracy: 0.9610
Epoch 57/65535
420/420 [==============================] - 225s 535ms/step - loss: 0.0045 - categorical_accuracy: 0.9803 - val_loss: 0.0069 - val_categorical_accuracy: 0.9612
Epoch 58/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0045 - categorical_accuracy: 0.9809 - val_loss: 0.0069 - val_categorical_accuracy: 0.9614
Epoch 59/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0045 - categorical_accuracy: 0.9804 - val_loss: 0.0070 - val_categorical_accuracy: 0.9611
Epoch 60/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0044 - categorical_accuracy: 0.9808 - val_loss: 0.0070 - val_categorical_accuracy: 0.9611
Epoch 61/65535
420/420 [==============================] - 225s 536ms/step - loss: 0.0044 - categorical_accuracy: 0.9804 - val_loss: 0.0070 - val_categorical_accuracy: 0.9609
Epoch 62/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0044 - categorical_accuracy: 0.9807 - val_loss: 0.0069 - val_categorical_accuracy: 0.9612
Epoch 63/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0044 - categorical_accuracy: 0.9801 - val_loss: 0.0070 - val_categorical_accuracy: 0.9611
Epoch 64/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0044 - categorical_accuracy: 0.9803 - val_loss: 0.0069 - val_categorical_accuracy: 0.9611
Epoch 65/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0044 - categorical_accuracy: 0.9809 - val_loss: 0.0069 - val_categorical_accuracy: 0.9609
Epoch 66/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0044 - categorical_accuracy: 0.9805 - val_loss: 0.0070 - val_categorical_accuracy: 0.9608
Epoch 00066: early stopping
========= generating oof predictions 05:15:51 =========
========= generating test set predictions 05:16:02 =========
========= fitting 3 th model 05:18:24 =========
Epoch 1/65535
420/420 [==============================] - 216s 513ms/step - loss: 0.1101 - categorical_accuracy: 0.3459 - val_loss: 0.1549 - val_categorical_accuracy: 0.1964
Epoch 2/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0562 - categorical_accuracy: 0.7603 - val_loss: 0.0546 - val_categorical_accuracy: 0.6714
Epoch 3/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0323 - categorical_accuracy: 0.8683 - val_loss: 0.0487 - val_categorical_accuracy: 0.7041
Epoch 4/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0228 - categorical_accuracy: 0.9021 - val_loss: 0.0219 - val_categorical_accuracy: 0.8865
Epoch 5/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0182 - categorical_accuracy: 0.9187 - val_loss: 0.0244 - val_categorical_accuracy: 0.8618
Epoch 6/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0156 - categorical_accuracy: 0.9282 - val_loss: 0.0282 - val_categorical_accuracy: 0.8336
Epoch 7/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0137 - categorical_accuracy: 0.9358 - val_loss: 0.0222 - val_categorical_accuracy: 0.8755
Epoch 8/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0124 - categorical_accuracy: 0.9418 - val_loss: 0.0158 - val_categorical_accuracy: 0.9151
Epoch 9/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0113 - categorical_accuracy: 0.9456 - val_loss: 0.0212 - val_categorical_accuracy: 0.8782
Epoch 10/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0105 - categorical_accuracy: 0.9495 - val_loss: 0.0160 - val_categorical_accuracy: 0.9113
Epoch 11/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0098 - categorical_accuracy: 0.9522 - val_loss: 0.0170 - val_categorical_accuracy: 0.9058
Epoch 12/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0090 - categorical_accuracy: 0.9552 - val_loss: 0.0176 - val_categorical_accuracy: 0.8999
Epoch 13/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0086 - categorical_accuracy: 0.9574 - val_loss: 0.0186 - val_categorical_accuracy: 0.8951
Epoch 14/65535
420/420 [==============================] - 231s 550ms/step - loss: 0.0081 - categorical_accuracy: 0.9595 - val_loss: 0.0175 - val_categorical_accuracy: 0.9025

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 15/65535
420/420 [==============================] - 231s 551ms/step - loss: 0.0066 - categorical_accuracy: 0.9679 - val_loss: 0.0084 - val_categorical_accuracy: 0.9546
Epoch 16/65535
420/420 [==============================] - 240s 571ms/step - loss: 0.0062 - categorical_accuracy: 0.9704 - val_loss: 0.0082 - val_categorical_accuracy: 0.9569
Epoch 17/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0060 - categorical_accuracy: 0.9718 - val_loss: 0.0087 - val_categorical_accuracy: 0.9532
Epoch 18/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0059 - categorical_accuracy: 0.9726 - val_loss: 0.0084 - val_categorical_accuracy: 0.9550
Epoch 19/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0058 - categorical_accuracy: 0.9732 - val_loss: 0.0084 - val_categorical_accuracy: 0.9549
Epoch 20/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0057 - categorical_accuracy: 0.9737 - val_loss: 0.0085 - val_categorical_accuracy: 0.9554
Epoch 21/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0055 - categorical_accuracy: 0.9745 - val_loss: 0.0085 - val_categorical_accuracy: 0.9540
Epoch 22/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0055 - categorical_accuracy: 0.9741 - val_loss: 0.0085 - val_categorical_accuracy: 0.9533

Epoch 00022: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 23/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0051 - categorical_accuracy: 0.9765 - val_loss: 0.0071 - val_categorical_accuracy: 0.9613
Epoch 24/65535
420/420 [==============================] - 224s 532ms/step - loss: 0.0051 - categorical_accuracy: 0.9768 - val_loss: 0.0071 - val_categorical_accuracy: 0.9619
Epoch 25/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0049 - categorical_accuracy: 0.9777 - val_loss: 0.0071 - val_categorical_accuracy: 0.9616
Epoch 26/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0049 - categorical_accuracy: 0.9773 - val_loss: 0.0071 - val_categorical_accuracy: 0.9615
Epoch 27/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0049 - categorical_accuracy: 0.9774 - val_loss: 0.0071 - val_categorical_accuracy: 0.9613
Epoch 28/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0049 - categorical_accuracy: 0.9779 - val_loss: 0.0071 - val_categorical_accuracy: 0.9613
Epoch 29/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0048 - categorical_accuracy: 0.9780 - val_loss: 0.0071 - val_categorical_accuracy: 0.9613

Epoch 00029: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 30/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0048 - categorical_accuracy: 0.9785 - val_loss: 0.0069 - val_categorical_accuracy: 0.9631
Epoch 31/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0048 - categorical_accuracy: 0.9782 - val_loss: 0.0069 - val_categorical_accuracy: 0.9624
Epoch 32/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0048 - categorical_accuracy: 0.9780 - val_loss: 0.0069 - val_categorical_accuracy: 0.9624
Epoch 33/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0048 - categorical_accuracy: 0.9787 - val_loss: 0.0069 - val_categorical_accuracy: 0.9624
Epoch 34/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0048 - categorical_accuracy: 0.9779 - val_loss: 0.0069 - val_categorical_accuracy: 0.9625
Epoch 35/65535
420/420 [==============================] - 228s 542ms/step - loss: 0.0047 - categorical_accuracy: 0.9789 - val_loss: 0.0069 - val_categorical_accuracy: 0.9624
Epoch 36/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0047 - categorical_accuracy: 0.9791 - val_loss: 0.0069 - val_categorical_accuracy: 0.9627
Epoch 37/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0047 - categorical_accuracy: 0.9790 - val_loss: 0.0069 - val_categorical_accuracy: 0.9626
Epoch 38/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0047 - categorical_accuracy: 0.9789 - val_loss: 0.0069 - val_categorical_accuracy: 0.9623
Epoch 39/65535
420/420 [==============================] - 221s 525ms/step - loss: 0.0046 - categorical_accuracy: 0.9791 - val_loss: 0.0069 - val_categorical_accuracy: 0.9625
Epoch 40/65535
420/420 [==============================] - 221s 525ms/step - loss: 0.0047 - categorical_accuracy: 0.9790 - val_loss: 0.0069 - val_categorical_accuracy: 0.9624
Epoch 41/65535
420/420 [==============================] - 222s 527ms/step - loss: 0.0047 - categorical_accuracy: 0.9789 - val_loss: 0.0069 - val_categorical_accuracy: 0.9630
Epoch 42/65535
420/420 [==============================] - 220s 525ms/step - loss: 0.0047 - categorical_accuracy: 0.9789 - val_loss: 0.0069 - val_categorical_accuracy: 0.9627
Epoch 43/65535
420/420 [==============================] - 220s 525ms/step - loss: 0.0046 - categorical_accuracy: 0.9791 - val_loss: 0.0069 - val_categorical_accuracy: 0.9622
Epoch 44/65535
420/420 [==============================] - 220s 524ms/step - loss: 0.0046 - categorical_accuracy: 0.9791 - val_loss: 0.0069 - val_categorical_accuracy: 0.9630
Epoch 45/65535
420/420 [==============================] - 222s 527ms/step - loss: 0.0046 - categorical_accuracy: 0.9792 - val_loss: 0.0069 - val_categorical_accuracy: 0.9620
Epoch 46/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0046 - categorical_accuracy: 0.9794 - val_loss: 0.0069 - val_categorical_accuracy: 0.9623
Epoch 47/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0046 - categorical_accuracy: 0.9787 - val_loss: 0.0069 - val_categorical_accuracy: 0.9627
Epoch 48/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0046 - categorical_accuracy: 0.9790 - val_loss: 0.0069 - val_categorical_accuracy: 0.9628
Epoch 49/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0046 - categorical_accuracy: 0.9791 - val_loss: 0.0069 - val_categorical_accuracy: 0.9623
Epoch 50/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0046 - categorical_accuracy: 0.9795 - val_loss: 0.0069 - val_categorical_accuracy: 0.9628
Epoch 51/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0046 - categorical_accuracy: 0.9788 - val_loss: 0.0069 - val_categorical_accuracy: 0.9629
Epoch 52/65535
420/420 [==============================] - 220s 524ms/step - loss: 0.0046 - categorical_accuracy: 0.9796 - val_loss: 0.0069 - val_categorical_accuracy: 0.9628
Epoch 53/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0046 - categorical_accuracy: 0.9791 - val_loss: 0.0069 - val_categorical_accuracy: 0.9630
Epoch 54/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0045 - categorical_accuracy: 0.9798 - val_loss: 0.0069 - val_categorical_accuracy: 0.9627
Epoch 55/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0045 - categorical_accuracy: 0.9794 - val_loss: 0.0069 - val_categorical_accuracy: 0.9625
Epoch 56/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0045 - categorical_accuracy: 0.9795 - val_loss: 0.0069 - val_categorical_accuracy: 0.9630
Epoch 57/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0045 - categorical_accuracy: 0.9796 - val_loss: 0.0069 - val_categorical_accuracy: 0.9625
Epoch 58/65535
420/420 [==============================] - 221s 525ms/step - loss: 0.0045 - categorical_accuracy: 0.9797 - val_loss: 0.0069 - val_categorical_accuracy: 0.9629
Epoch 59/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0045 - categorical_accuracy: 0.9801 - val_loss: 0.0069 - val_categorical_accuracy: 0.9622
Epoch 60/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0045 - categorical_accuracy: 0.9797 - val_loss: 0.0069 - val_categorical_accuracy: 0.9631
Epoch 61/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0044 - categorical_accuracy: 0.9805 - val_loss: 0.0069 - val_categorical_accuracy: 0.9632
Epoch 62/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0044 - categorical_accuracy: 0.9799 - val_loss: 0.0069 - val_categorical_accuracy: 0.9624
Epoch 63/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0044 - categorical_accuracy: 0.9800 - val_loss: 0.0069 - val_categorical_accuracy: 0.9627
Epoch 64/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0044 - categorical_accuracy: 0.9803 - val_loss: 0.0069 - val_categorical_accuracy: 0.9625
Epoch 65/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0044 - categorical_accuracy: 0.9802 - val_loss: 0.0068 - val_categorical_accuracy: 0.9633
Epoch 66/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0044 - categorical_accuracy: 0.9802 - val_loss: 0.0069 - val_categorical_accuracy: 0.9627
Epoch 67/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0044 - categorical_accuracy: 0.9805 - val_loss: 0.0069 - val_categorical_accuracy: 0.9624
Epoch 68/65535
420/420 [==============================] - 220s 525ms/step - loss: 0.0044 - categorical_accuracy: 0.9805 - val_loss: 0.0069 - val_categorical_accuracy: 0.9627
Epoch 69/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0044 - categorical_accuracy: 0.9805 - val_loss: 0.0069 - val_categorical_accuracy: 0.9627
Epoch 70/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0044 - categorical_accuracy: 0.9805 - val_loss: 0.0069 - val_categorical_accuracy: 0.9632
Epoch 71/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0043 - categorical_accuracy: 0.9812 - val_loss: 0.0069 - val_categorical_accuracy: 0.9627
Epoch 72/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0043 - categorical_accuracy: 0.9806 - val_loss: 0.0068 - val_categorical_accuracy: 0.9634
Epoch 73/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0044 - categorical_accuracy: 0.9807 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 74/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0043 - categorical_accuracy: 0.9808 - val_loss: 0.0068 - val_categorical_accuracy: 0.9632
Epoch 75/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0043 - categorical_accuracy: 0.9810 - val_loss: 0.0068 - val_categorical_accuracy: 0.9629
Epoch 76/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0043 - categorical_accuracy: 0.9810 - val_loss: 0.0068 - val_categorical_accuracy: 0.9633
Epoch 77/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0043 - categorical_accuracy: 0.9808 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 78/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0043 - categorical_accuracy: 0.9813 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 79/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0043 - categorical_accuracy: 0.9804 - val_loss: 0.0068 - val_categorical_accuracy: 0.9627
Epoch 80/65535
420/420 [==============================] - 220s 523ms/step - loss: 0.0043 - categorical_accuracy: 0.9806 - val_loss: 0.0068 - val_categorical_accuracy: 0.9625
Epoch 81/65535
420/420 [==============================] - 221s 525ms/step - loss: 0.0042 - categorical_accuracy: 0.9816 - val_loss: 0.0069 - val_categorical_accuracy: 0.9630
Epoch 82/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0043 - categorical_accuracy: 0.9807 - val_loss: 0.0068 - val_categorical_accuracy: 0.9626
Epoch 83/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0042 - categorical_accuracy: 0.9812 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 84/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0042 - categorical_accuracy: 0.9813 - val_loss: 0.0068 - val_categorical_accuracy: 0.9624
Epoch 85/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0042 - categorical_accuracy: 0.9815 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 86/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0042 - categorical_accuracy: 0.9809 - val_loss: 0.0068 - val_categorical_accuracy: 0.9631
Epoch 87/65535
420/420 [==============================] - 221s 525ms/step - loss: 0.0042 - categorical_accuracy: 0.9816 - val_loss: 0.0068 - val_categorical_accuracy: 0.9625
Epoch 88/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0042 - categorical_accuracy: 0.9816 - val_loss: 0.0068 - val_categorical_accuracy: 0.9628
Epoch 89/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0042 - categorical_accuracy: 0.9815 - val_loss: 0.0068 - val_categorical_accuracy: 0.9635
Epoch 90/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0042 - categorical_accuracy: 0.9813 - val_loss: 0.0068 - val_categorical_accuracy: 0.9633
Epoch 91/65535
420/420 [==============================] - 223s 532ms/step - loss: 0.0042 - categorical_accuracy: 0.9815 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 92/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0041 - categorical_accuracy: 0.9819 - val_loss: 0.0068 - val_categorical_accuracy: 0.9626
Epoch 93/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0041 - categorical_accuracy: 0.9820 - val_loss: 0.0068 - val_categorical_accuracy: 0.9632
Epoch 94/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0042 - categorical_accuracy: 0.9813 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 95/65535
420/420 [==============================] - 222s 527ms/step - loss: 0.0041 - categorical_accuracy: 0.9816 - val_loss: 0.0068 - val_categorical_accuracy: 0.9636
Epoch 96/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0041 - categorical_accuracy: 0.9819 - val_loss: 0.0068 - val_categorical_accuracy: 0.9627
Epoch 97/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0041 - categorical_accuracy: 0.9820 - val_loss: 0.0068 - val_categorical_accuracy: 0.9631
Epoch 98/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0041 - categorical_accuracy: 0.9818 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 99/65535
420/420 [==============================] - 223s 530ms/step - loss: 0.0041 - categorical_accuracy: 0.9820 - val_loss: 0.0068 - val_categorical_accuracy: 0.9624
Epoch 100/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0041 - categorical_accuracy: 0.9817 - val_loss: 0.0068 - val_categorical_accuracy: 0.9627
Epoch 101/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0041 - categorical_accuracy: 0.9819 - val_loss: 0.0068 - val_categorical_accuracy: 0.9627
Epoch 102/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.0041 - categorical_accuracy: 0.9823 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 103/65535
420/420 [==============================] - 225s 536ms/step - loss: 0.0041 - categorical_accuracy: 0.9820 - val_loss: 0.0068 - val_categorical_accuracy: 0.9626
Epoch 104/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0040 - categorical_accuracy: 0.9822 - val_loss: 0.0068 - val_categorical_accuracy: 0.9632
Epoch 105/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0041 - categorical_accuracy: 0.9815 - val_loss: 0.0068 - val_categorical_accuracy: 0.9638
Epoch 106/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0040 - categorical_accuracy: 0.9823 - val_loss: 0.0068 - val_categorical_accuracy: 0.9633
Epoch 107/65535
420/420 [==============================] - 222s 530ms/step - loss: 0.0041 - categorical_accuracy: 0.9818 - val_loss: 0.0068 - val_categorical_accuracy: 0.9633
Epoch 108/65535
420/420 [==============================] - 220s 523ms/step - loss: 0.0040 - categorical_accuracy: 0.9825 - val_loss: 0.0068 - val_categorical_accuracy: 0.9636
Epoch 109/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0040 - categorical_accuracy: 0.9821 - val_loss: 0.0068 - val_categorical_accuracy: 0.9633
Epoch 110/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0040 - categorical_accuracy: 0.9830 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 111/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0040 - categorical_accuracy: 0.9823 - val_loss: 0.0068 - val_categorical_accuracy: 0.9629
Epoch 112/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0040 - categorical_accuracy: 0.9822 - val_loss: 0.0068 - val_categorical_accuracy: 0.9635
Epoch 113/65535
420/420 [==============================] - 220s 525ms/step - loss: 0.0040 - categorical_accuracy: 0.9822 - val_loss: 0.0068 - val_categorical_accuracy: 0.9625
Epoch 114/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0040 - categorical_accuracy: 0.9824 - val_loss: 0.0068 - val_categorical_accuracy: 0.9627
Epoch 115/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0040 - categorical_accuracy: 0.9828 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 116/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0040 - categorical_accuracy: 0.9827 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 117/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0039 - categorical_accuracy: 0.9826 - val_loss: 0.0068 - val_categorical_accuracy: 0.9632
Epoch 118/65535
420/420 [==============================] - 221s 526ms/step - loss: 0.0040 - categorical_accuracy: 0.9827 - val_loss: 0.0068 - val_categorical_accuracy: 0.9633
Epoch 119/65535
420/420 [==============================] - 222s 529ms/step - loss: 0.0039 - categorical_accuracy: 0.9827 - val_loss: 0.0068 - val_categorical_accuracy: 0.9630
Epoch 00119: early stopping
========= generating oof predictions 12:39:28 =========
========= generating test set predictions 12:39:40 =========
========= fitting 4 th model 12:42:00 =========
Epoch 1/65535
420/420 [==============================] - 224s 534ms/step - loss: 0.1100 - categorical_accuracy: 0.3516 - val_loss: 0.0860 - val_categorical_accuracy: 0.4394
Epoch 2/65535
420/420 [==============================] - 232s 551ms/step - loss: 0.0549 - categorical_accuracy: 0.7719 - val_loss: 0.0424 - val_categorical_accuracy: 0.8014
Epoch 3/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0311 - categorical_accuracy: 0.8746 - val_loss: 0.0288 - val_categorical_accuracy: 0.8515
Epoch 4/65535
420/420 [==============================] - 232s 551ms/step - loss: 0.0222 - categorical_accuracy: 0.9041 - val_loss: 0.0247 - val_categorical_accuracy: 0.8649
Epoch 5/65535
420/420 [==============================] - 231s 549ms/step - loss: 0.0179 - categorical_accuracy: 0.9195 - val_loss: 0.0209 - val_categorical_accuracy: 0.8913
Epoch 6/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0153 - categorical_accuracy: 0.9296 - val_loss: 0.0185 - val_categorical_accuracy: 0.9017
Epoch 7/65535
420/420 [==============================] - 231s 551ms/step - loss: 0.0136 - categorical_accuracy: 0.9369 - val_loss: 0.0197 - val_categorical_accuracy: 0.8942
Epoch 8/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0123 - categorical_accuracy: 0.9424 - val_loss: 0.0212 - val_categorical_accuracy: 0.8796
Epoch 9/65535
420/420 [==============================] - 233s 555ms/step - loss: 0.0112 - categorical_accuracy: 0.9470 - val_loss: 0.0190 - val_categorical_accuracy: 0.8968
Epoch 10/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0105 - categorical_accuracy: 0.9488 - val_loss: 0.0254 - val_categorical_accuracy: 0.8520
Epoch 11/65535
420/420 [==============================] - 233s 555ms/step - loss: 0.0097 - categorical_accuracy: 0.9521 - val_loss: 0.0171 - val_categorical_accuracy: 0.9064
Epoch 12/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0092 - categorical_accuracy: 0.9553 - val_loss: 0.0267 - val_categorical_accuracy: 0.8466
Epoch 13/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0086 - categorical_accuracy: 0.9574 - val_loss: 0.0298 - val_categorical_accuracy: 0.8294
Epoch 14/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0082 - categorical_accuracy: 0.9599 - val_loss: 0.0227 - val_categorical_accuracy: 0.8714
Epoch 15/65535
420/420 [==============================] - 231s 551ms/step - loss: 0.0081 - categorical_accuracy: 0.9603 - val_loss: 0.0504 - val_categorical_accuracy: 0.7488
Epoch 16/65535
420/420 [==============================] - 232s 554ms/step - loss: 0.0076 - categorical_accuracy: 0.9623 - val_loss: 0.0200 - val_categorical_accuracy: 0.8887
Epoch 17/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0071 - categorical_accuracy: 0.9650 - val_loss: 0.0117 - val_categorical_accuracy: 0.9347
Epoch 18/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0067 - categorical_accuracy: 0.9669 - val_loss: 0.0149 - val_categorical_accuracy: 0.9144
Epoch 19/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0065 - categorical_accuracy: 0.9679 - val_loss: 0.0174 - val_categorical_accuracy: 0.9009
Epoch 20/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0062 - categorical_accuracy: 0.9691 - val_loss: 0.0120 - val_categorical_accuracy: 0.9351
Epoch 21/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0061 - categorical_accuracy: 0.9698 - val_loss: 0.0251 - val_categorical_accuracy: 0.8652
Epoch 22/65535
420/420 [==============================] - 233s 554ms/step - loss: 0.0058 - categorical_accuracy: 0.9710 - val_loss: 0.0192 - val_categorical_accuracy: 0.8910
Epoch 23/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0056 - categorical_accuracy: 0.9722 - val_loss: 0.0169 - val_categorical_accuracy: 0.9137

Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 24/65535
420/420 [==============================] - 233s 554ms/step - loss: 0.0046 - categorical_accuracy: 0.9769 - val_loss: 0.0075 - val_categorical_accuracy: 0.9586
Epoch 25/65535
420/420 [==============================] - 233s 554ms/step - loss: 0.0042 - categorical_accuracy: 0.9796 - val_loss: 0.0073 - val_categorical_accuracy: 0.9594
Epoch 26/65535
420/420 [==============================] - 233s 554ms/step - loss: 0.0041 - categorical_accuracy: 0.9802 - val_loss: 0.0074 - val_categorical_accuracy: 0.9591
Epoch 27/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0040 - categorical_accuracy: 0.9809 - val_loss: 0.0076 - val_categorical_accuracy: 0.9582
Epoch 28/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0039 - categorical_accuracy: 0.9813 - val_loss: 0.0076 - val_categorical_accuracy: 0.9582
Epoch 29/65535
420/420 [==============================] - 233s 555ms/step - loss: 0.0038 - categorical_accuracy: 0.9817 - val_loss: 0.0077 - val_categorical_accuracy: 0.9568
Epoch 30/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0037 - categorical_accuracy: 0.9821 - val_loss: 0.0073 - val_categorical_accuracy: 0.9592
Epoch 31/65535
420/420 [==============================] - 233s 554ms/step - loss: 0.0037 - categorical_accuracy: 0.9826 - val_loss: 0.0073 - val_categorical_accuracy: 0.9603

Epoch 00031: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 32/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0035 - categorical_accuracy: 0.9840 - val_loss: 0.0064 - val_categorical_accuracy: 0.9650
Epoch 33/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0034 - categorical_accuracy: 0.9845 - val_loss: 0.0064 - val_categorical_accuracy: 0.9650
Epoch 34/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0034 - categorical_accuracy: 0.9846 - val_loss: 0.0064 - val_categorical_accuracy: 0.9649
Epoch 35/65535
420/420 [==============================] - 231s 549ms/step - loss: 0.0033 - categorical_accuracy: 0.9850 - val_loss: 0.0064 - val_categorical_accuracy: 0.9649
Epoch 36/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0033 - categorical_accuracy: 0.9848 - val_loss: 0.0064 - val_categorical_accuracy: 0.9652
Epoch 37/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0033 - categorical_accuracy: 0.9851 - val_loss: 0.0064 - val_categorical_accuracy: 0.9652
Epoch 38/65535
420/420 [==============================] - 232s 551ms/step - loss: 0.0033 - categorical_accuracy: 0.9849 - val_loss: 0.0064 - val_categorical_accuracy: 0.9653

Epoch 00038: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 39/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0032 - categorical_accuracy: 0.9850 - val_loss: 0.0063 - val_categorical_accuracy: 0.9659
Epoch 40/65535
420/420 [==============================] - 232s 552ms/step - loss: 0.0032 - categorical_accuracy: 0.9853 - val_loss: 0.0063 - val_categorical_accuracy: 0.9656
Epoch 41/65535
420/420 [==============================] - 233s 554ms/step - loss: 0.0032 - categorical_accuracy: 0.9858 - val_loss: 0.0063 - val_categorical_accuracy: 0.9654
Epoch 42/65535
420/420 [==============================] - 231s 550ms/step - loss: 0.0032 - categorical_accuracy: 0.9854 - val_loss: 0.0063 - val_categorical_accuracy: 0.9653
Epoch 43/65535
420/420 [==============================] - 231s 551ms/step - loss: 0.0032 - categorical_accuracy: 0.9858 - val_loss: 0.0063 - val_categorical_accuracy: 0.9655
Epoch 44/65535
420/420 [==============================] - 234s 557ms/step - loss: 0.0032 - categorical_accuracy: 0.9854 - val_loss: 0.0063 - val_categorical_accuracy: 0.9656
Epoch 45/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0031 - categorical_accuracy: 0.9859 - val_loss: 0.0063 - val_categorical_accuracy: 0.9656
Epoch 46/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0032 - categorical_accuracy: 0.9858 - val_loss: 0.0063 - val_categorical_accuracy: 0.9659
Epoch 47/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0031 - categorical_accuracy: 0.9856 - val_loss: 0.0063 - val_categorical_accuracy: 0.9656
Epoch 48/65535
420/420 [==============================] - 233s 555ms/step - loss: 0.0031 - categorical_accuracy: 0.9858 - val_loss: 0.0063 - val_categorical_accuracy: 0.9662
Epoch 49/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0031 - categorical_accuracy: 0.9857 - val_loss: 0.0063 - val_categorical_accuracy: 0.9657
Epoch 50/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0031 - categorical_accuracy: 0.9858 - val_loss: 0.0063 - val_categorical_accuracy: 0.9656
Epoch 51/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0031 - categorical_accuracy: 0.9858 - val_loss: 0.0063 - val_categorical_accuracy: 0.9656
Epoch 52/65535
420/420 [==============================] - 233s 554ms/step - loss: 0.0031 - categorical_accuracy: 0.9860 - val_loss: 0.0063 - val_categorical_accuracy: 0.9657
Epoch 53/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0031 - categorical_accuracy: 0.9860 - val_loss: 0.0063 - val_categorical_accuracy: 0.9660
Epoch 54/65535
420/420 [==============================] - 232s 553ms/step - loss: 0.0031 - categorical_accuracy: 0.9861 - val_loss: 0.0063 - val_categorical_accuracy: 0.9659
Epoch 00054: early stopping
========= generating oof predictions 16:10:49 =========
========= generating test set predictions 16:11:00 =========
========= fitting 5 th model 16:13:21 =========
Epoch 1/65535
420/420 [==============================] - 205s 489ms/step - loss: 0.1091 - categorical_accuracy: 0.3550 - val_loss: 0.0913 - val_categorical_accuracy: 0.4203
Epoch 2/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0555 - categorical_accuracy: 0.7712 - val_loss: 0.0437 - val_categorical_accuracy: 0.7617
Epoch 3/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0315 - categorical_accuracy: 0.8728 - val_loss: 0.0349 - val_categorical_accuracy: 0.8059
Epoch 4/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0223 - categorical_accuracy: 0.9056 - val_loss: 0.0238 - val_categorical_accuracy: 0.8763
Epoch 5/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0180 - categorical_accuracy: 0.9198 - val_loss: 0.0205 - val_categorical_accuracy: 0.8888
Epoch 6/65535
420/420 [==============================] - 207s 494ms/step - loss: 0.0155 - categorical_accuracy: 0.9284 - val_loss: 0.0156 - val_categorical_accuracy: 0.9213
Epoch 7/65535
420/420 [==============================] - 208s 494ms/step - loss: 0.0136 - categorical_accuracy: 0.9364 - val_loss: 0.0163 - val_categorical_accuracy: 0.9133
Epoch 8/65535
420/420 [==============================] - 208s 496ms/step - loss: 0.0121 - categorical_accuracy: 0.9426 - val_loss: 0.0178 - val_categorical_accuracy: 0.9029
Epoch 9/65535
420/420 [==============================] - 208s 496ms/step - loss: 0.0111 - categorical_accuracy: 0.9468 - val_loss: 0.0154 - val_categorical_accuracy: 0.9155
Epoch 10/65535
420/420 [==============================] - 208s 496ms/step - loss: 0.0104 - categorical_accuracy: 0.9503 - val_loss: 0.0185 - val_categorical_accuracy: 0.8974
Epoch 11/65535
420/420 [==============================] - 208s 496ms/step - loss: 0.0097 - categorical_accuracy: 0.9533 - val_loss: 0.0198 - val_categorical_accuracy: 0.8906
Epoch 12/65535
420/420 [==============================] - 209s 497ms/step - loss: 0.0091 - categorical_accuracy: 0.9546 - val_loss: 0.0162 - val_categorical_accuracy: 0.9109
Epoch 13/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0086 - categorical_accuracy: 0.9580 - val_loss: 0.0178 - val_categorical_accuracy: 0.8996
Epoch 14/65535
420/420 [==============================] - 207s 494ms/step - loss: 0.0082 - categorical_accuracy: 0.9598 - val_loss: 0.0199 - val_categorical_accuracy: 0.8894
Epoch 15/65535
420/420 [==============================] - 208s 494ms/step - loss: 0.0077 - categorical_accuracy: 0.9625 - val_loss: 0.0187 - val_categorical_accuracy: 0.8976

Epoch 00015: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 16/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0062 - categorical_accuracy: 0.9697 - val_loss: 0.0086 - val_categorical_accuracy: 0.9531
Epoch 17/65535
420/420 [==============================] - 216s 515ms/step - loss: 0.0058 - categorical_accuracy: 0.9722 - val_loss: 0.0089 - val_categorical_accuracy: 0.9524
Epoch 18/65535
420/420 [==============================] - 215s 511ms/step - loss: 0.0057 - categorical_accuracy: 0.9735 - val_loss: 0.0085 - val_categorical_accuracy: 0.9530
Epoch 19/65535
420/420 [==============================] - 208s 496ms/step - loss: 0.0056 - categorical_accuracy: 0.9738 - val_loss: 0.0085 - val_categorical_accuracy: 0.9533
Epoch 20/65535
420/420 [==============================] - 209s 498ms/step - loss: 0.0055 - categorical_accuracy: 0.9747 - val_loss: 0.0085 - val_categorical_accuracy: 0.9530
Epoch 21/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0054 - categorical_accuracy: 0.9744 - val_loss: 0.0090 - val_categorical_accuracy: 0.9513
Epoch 22/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0053 - categorical_accuracy: 0.9755 - val_loss: 0.0088 - val_categorical_accuracy: 0.9518

Epoch 00022: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 23/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0049 - categorical_accuracy: 0.9773 - val_loss: 0.0076 - val_categorical_accuracy: 0.9596
Epoch 24/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0048 - categorical_accuracy: 0.9779 - val_loss: 0.0077 - val_categorical_accuracy: 0.9591
Epoch 25/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0048 - categorical_accuracy: 0.9783 - val_loss: 0.0076 - val_categorical_accuracy: 0.9598
Epoch 26/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0047 - categorical_accuracy: 0.9784 - val_loss: 0.0075 - val_categorical_accuracy: 0.9603
Epoch 27/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0047 - categorical_accuracy: 0.9786 - val_loss: 0.0076 - val_categorical_accuracy: 0.9597
Epoch 28/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0047 - categorical_accuracy: 0.9784 - val_loss: 0.0076 - val_categorical_accuracy: 0.9594
Epoch 29/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0047 - categorical_accuracy: 0.9785 - val_loss: 0.0077 - val_categorical_accuracy: 0.9595

Epoch 00029: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 30/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0047 - categorical_accuracy: 0.9787 - val_loss: 0.0075 - val_categorical_accuracy: 0.9606
Epoch 31/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0046 - categorical_accuracy: 0.9796 - val_loss: 0.0075 - val_categorical_accuracy: 0.9606
Epoch 32/65535
420/420 [==============================] - 208s 494ms/step - loss: 0.0046 - categorical_accuracy: 0.9796 - val_loss: 0.0075 - val_categorical_accuracy: 0.9603
Epoch 33/65535
420/420 [==============================] - 208s 494ms/step - loss: 0.0045 - categorical_accuracy: 0.9798 - val_loss: 0.0075 - val_categorical_accuracy: 0.9609
Epoch 34/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0046 - categorical_accuracy: 0.9795 - val_loss: 0.0075 - val_categorical_accuracy: 0.9609
Epoch 35/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0046 - categorical_accuracy: 0.9798 - val_loss: 0.0074 - val_categorical_accuracy: 0.9606
Epoch 36/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0045 - categorical_accuracy: 0.9794 - val_loss: 0.0075 - val_categorical_accuracy: 0.9610
Epoch 37/65535
420/420 [==============================] - 207s 494ms/step - loss: 0.0045 - categorical_accuracy: 0.9800 - val_loss: 0.0074 - val_categorical_accuracy: 0.9607
Epoch 38/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0045 - categorical_accuracy: 0.9795 - val_loss: 0.0075 - val_categorical_accuracy: 0.9610
Epoch 39/65535
420/420 [==============================] - 208s 494ms/step - loss: 0.0045 - categorical_accuracy: 0.9798 - val_loss: 0.0075 - val_categorical_accuracy: 0.9608
Epoch 40/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0045 - categorical_accuracy: 0.9795 - val_loss: 0.0075 - val_categorical_accuracy: 0.9608
Epoch 41/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0045 - categorical_accuracy: 0.9794 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 42/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0045 - categorical_accuracy: 0.9798 - val_loss: 0.0075 - val_categorical_accuracy: 0.9606
Epoch 43/65535
420/420 [==============================] - 207s 494ms/step - loss: 0.0044 - categorical_accuracy: 0.9799 - val_loss: 0.0074 - val_categorical_accuracy: 0.9609
Epoch 44/65535
420/420 [==============================] - 211s 503ms/step - loss: 0.0045 - categorical_accuracy: 0.9801 - val_loss: 0.0074 - val_categorical_accuracy: 0.9609
Epoch 45/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0045 - categorical_accuracy: 0.9798 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 46/65535
420/420 [==============================] - 217s 517ms/step - loss: 0.0044 - categorical_accuracy: 0.9801 - val_loss: 0.0075 - val_categorical_accuracy: 0.9610
Epoch 47/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0044 - categorical_accuracy: 0.9800 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 48/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0044 - categorical_accuracy: 0.9802 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 49/65535
420/420 [==============================] - 208s 494ms/step - loss: 0.0044 - categorical_accuracy: 0.9807 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 50/65535
420/420 [==============================] - 211s 501ms/step - loss: 0.0044 - categorical_accuracy: 0.9809 - val_loss: 0.0074 - val_categorical_accuracy: 0.9610
Epoch 51/65535
420/420 [==============================] - 209s 498ms/step - loss: 0.0044 - categorical_accuracy: 0.9805 - val_loss: 0.0074 - val_categorical_accuracy: 0.9610
Epoch 52/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0044 - categorical_accuracy: 0.9805 - val_loss: 0.0074 - val_categorical_accuracy: 0.9614
Epoch 53/65535
420/420 [==============================] - 208s 496ms/step - loss: 0.0043 - categorical_accuracy: 0.9806 - val_loss: 0.0075 - val_categorical_accuracy: 0.9610
Epoch 54/65535
420/420 [==============================] - 207s 494ms/step - loss: 0.0044 - categorical_accuracy: 0.9804 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 55/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0043 - categorical_accuracy: 0.9806 - val_loss: 0.0074 - val_categorical_accuracy: 0.9609
Epoch 56/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0043 - categorical_accuracy: 0.9809 - val_loss: 0.0074 - val_categorical_accuracy: 0.9608
Epoch 57/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0043 - categorical_accuracy: 0.9805 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 58/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0043 - categorical_accuracy: 0.9811 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 59/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0043 - categorical_accuracy: 0.9807 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 60/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0043 - categorical_accuracy: 0.9810 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 61/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0043 - categorical_accuracy: 0.9811 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 62/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0043 - categorical_accuracy: 0.9810 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 63/65535
420/420 [==============================] - 206s 492ms/step - loss: 0.0043 - categorical_accuracy: 0.9815 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 64/65535
420/420 [==============================] - 207s 494ms/step - loss: 0.0043 - categorical_accuracy: 0.9813 - val_loss: 0.0074 - val_categorical_accuracy: 0.9617
Epoch 65/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0042 - categorical_accuracy: 0.9812 - val_loss: 0.0074 - val_categorical_accuracy: 0.9614
Epoch 66/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0042 - categorical_accuracy: 0.9815 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 67/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0042 - categorical_accuracy: 0.9815 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 68/65535
420/420 [==============================] - 207s 494ms/step - loss: 0.0042 - categorical_accuracy: 0.9811 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 69/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0042 - categorical_accuracy: 0.9814 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 70/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0042 - categorical_accuracy: 0.9810 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 71/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0042 - categorical_accuracy: 0.9815 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 72/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0042 - categorical_accuracy: 0.9812 - val_loss: 0.0074 - val_categorical_accuracy: 0.9614
Epoch 73/65535
420/420 [==============================] - 206s 491ms/step - loss: 0.0042 - categorical_accuracy: 0.9812 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 74/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0042 - categorical_accuracy: 0.9814 - val_loss: 0.0074 - val_categorical_accuracy: 0.9609
Epoch 75/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0042 - categorical_accuracy: 0.9812 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 76/65535
420/420 [==============================] - 206s 492ms/step - loss: 0.0041 - categorical_accuracy: 0.9817 - val_loss: 0.0074 - val_categorical_accuracy: 0.9614
Epoch 77/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0041 - categorical_accuracy: 0.9817 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 78/65535
420/420 [==============================] - 206s 491ms/step - loss: 0.0041 - categorical_accuracy: 0.9811 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 79/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0041 - categorical_accuracy: 0.9818 - val_loss: 0.0074 - val_categorical_accuracy: 0.9617
Epoch 80/65535
420/420 [==============================] - 206s 492ms/step - loss: 0.0041 - categorical_accuracy: 0.9819 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 81/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0041 - categorical_accuracy: 0.9822 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 82/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0041 - categorical_accuracy: 0.9825 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 83/65535
420/420 [==============================] - 206s 492ms/step - loss: 0.0040 - categorical_accuracy: 0.9822 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 84/65535
420/420 [==============================] - 206s 491ms/step - loss: 0.0041 - categorical_accuracy: 0.9816 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 85/65535
420/420 [==============================] - 206s 492ms/step - loss: 0.0040 - categorical_accuracy: 0.9824 - val_loss: 0.0074 - val_categorical_accuracy: 0.9617
Epoch 86/65535
420/420 [==============================] - 208s 496ms/step - loss: 0.0041 - categorical_accuracy: 0.9821 - val_loss: 0.0075 - val_categorical_accuracy: 0.9612
Epoch 87/65535
420/420 [==============================] - 211s 503ms/step - loss: 0.0040 - categorical_accuracy: 0.9826 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 00087: early stopping
========= generating oof predictions 21:15:02 =========
========= generating test set predictions 21:15:14 =========
train loss avg 0.003996933161364087 -- std 0.0005229255600279308, val loss avg 0.006936726585468425 -- std 0.00037533445310203393
train acc avg 0.982412092711896 -- std 0.0021043829853549818, val acc avg 0.962337302365033 -- std 0.0019321622304466957
mean nb epochs 78.6
dump oof predicted probs
dump test set predicted probs
