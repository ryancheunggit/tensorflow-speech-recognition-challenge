en (master *+) python $ python train.py logspectrogram_25_18.75 arm_dscnn t 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (55, 201) ===========
------------- SUMMARY OF MODEL -------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 55, 201)           0
_________________________________________________________________
reshape_1 (Reshape)          (None, 55, 201, 1)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 101, 192)      7104
_________________________________________________________________
batch_normalization_1 (Batch (None, 28, 101, 192)      768
_________________________________________________________________
activation_1 (Activation)    (None, 28, 101, 192)      0
_________________________________________________________________
dropout_1 (Dropout)          (None, 28, 101, 192)      0
_________________________________________________________________
separable_conv2d_1 (Separabl (None, 28, 101, 192)      38784
_________________________________________________________________
batch_normalization_2 (Batch (None, 28, 101, 192)      768
_________________________________________________________________
activation_2 (Activation)    (None, 28, 101, 192)      0
_________________________________________________________________
dropout_2 (Dropout)          (None, 28, 101, 192)      0
_________________________________________________________________
separable_conv2d_2 (Separabl (None, 28, 101, 192)      38784
_________________________________________________________________
batch_normalization_3 (Batch (None, 28, 101, 192)      768
_________________________________________________________________
activation_3 (Activation)    (None, 28, 101, 192)      0
_________________________________________________________________
dropout_3 (Dropout)          (None, 28, 101, 192)      0
_________________________________________________________________
separable_conv2d_3 (Separabl (None, 28, 101, 192)      38784
_________________________________________________________________
batch_normalization_4 (Batch (None, 28, 101, 192)      768
_________________________________________________________________
activation_4 (Activation)    (None, 28, 101, 192)      0
_________________________________________________________________
dropout_4 (Dropout)          (None, 28, 101, 192)      0
_________________________________________________________________
separable_conv2d_4 (Separabl (None, 28, 101, 192)      38784
_________________________________________________________________
batch_normalization_5 (Batch (None, 28, 101, 192)      768
_________________________________________________________________
activation_5 (Activation)    (None, 28, 101, 192)      0
_________________________________________________________________
dropout_5 (Dropout)          (None, 28, 101, 192)      0
_________________________________________________________________
global_average_pooling2d_1 ( (None, 192)               0
_________________________________________________________________
dense_1 (Dense)              (None, 31)                5983
=================================================================
Total params: 172,063
Trainable params: 170,143
Non-trainable params: 1,920
_________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 14:57:37 =========
Epoch 1/65535
420/420 [==============================] - 209s 497ms/step - loss: 0.1030 - categorical_accuracy: 0.3417 - val_loss: 0.0707 - val_categorical_accuracy: 0.5640
Epoch 2/65535
420/420 [==============================] - 209s 499ms/step - loss: 0.0368 - categorical_accuracy: 0.8031 - val_loss: 0.0269 - val_categorical_accuracy: 0.8471
Epoch 3/65535
420/420 [==============================] - 205s 489ms/step - loss: 0.0226 - categorical_accuracy: 0.8792 - val_loss: 0.0211 - val_categorical_accuracy: 0.8856
Epoch 4/65535
420/420 [==============================] - 206s 490ms/step - loss: 0.0179 - categorical_accuracy: 0.9036 - val_loss: 0.0195 - val_categorical_accuracy: 0.8894
Epoch 5/65535
420/420 [==============================] - 210s 500ms/step - loss: 0.0162 - categorical_accuracy: 0.9123 - val_loss: 0.0163 - val_categorical_accuracy: 0.9092
Epoch 6/65535
420/420 [==============================] - 209s 497ms/step - loss: 0.0144 - categorical_accuracy: 0.9221 - val_loss: 0.0160 - val_categorical_accuracy: 0.9149
Epoch 7/65535
420/420 [==============================] - 206s 491ms/step - loss: 0.0133 - categorical_accuracy: 0.9279 - val_loss: 0.0186 - val_categorical_accuracy: 0.9004
Epoch 8/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0123 - categorical_accuracy: 0.9320 - val_loss: 0.0169 - val_categorical_accuracy: 0.9129
Epoch 9/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0116 - categorical_accuracy: 0.9362 - val_loss: 0.0167 - val_categorical_accuracy: 0.9150
Epoch 10/65535
420/420 [==============================] - 215s 511ms/step - loss: 0.0108 - categorical_accuracy: 0.9401 - val_loss: 0.0207 - val_categorical_accuracy: 0.8937
Epoch 11/65535
420/420 [==============================] - 209s 499ms/step - loss: 0.0104 - categorical_accuracy: 0.9431 - val_loss: 0.0169 - val_categorical_accuracy: 0.9113
Epoch 12/65535
420/420 [==============================] - 210s 499ms/step - loss: 0.0102 - categorical_accuracy: 0.9432 - val_loss: 0.0185 - val_categorical_accuracy: 0.9099

Epoch 00012: ReduceLROnPlateau reducing learning rate to 0.0009999999776482583.
Epoch 13/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0073 - categorical_accuracy: 0.9604 - val_loss: 0.0090 - val_categorical_accuracy: 0.9507
Epoch 14/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0066 - categorical_accuracy: 0.9639 - val_loss: 0.0095 - val_categorical_accuracy: 0.9491
Epoch 15/65535
420/420 [==============================] - 209s 497ms/step - loss: 0.0063 - categorical_accuracy: 0.9667 - val_loss: 0.0092 - val_categorical_accuracy: 0.9495
Epoch 16/65535
420/420 [==============================] - 212s 505ms/step - loss: 0.0061 - categorical_accuracy: 0.9668 - val_loss: 0.0092 - val_categorical_accuracy: 0.9498
Epoch 17/65535
420/420 [==============================] - 212s 505ms/step - loss: 0.0059 - categorical_accuracy: 0.9685 - val_loss: 0.0091 - val_categorical_accuracy: 0.9510
Epoch 18/65535
420/420 [==============================] - 208s 494ms/step - loss: 0.0058 - categorical_accuracy: 0.9687 - val_loss: 0.0089 - val_categorical_accuracy: 0.9516
Epoch 19/65535
420/420 [==============================] - 209s 498ms/step - loss: 0.0056 - categorical_accuracy: 0.9703 - val_loss: 0.0091 - val_categorical_accuracy: 0.9510

Epoch 00019: ReduceLROnPlateau reducing learning rate to 0.0001999999862164259.
Epoch 20/65535
420/420 [==============================] - 210s 500ms/step - loss: 0.0050 - categorical_accuracy: 0.9733 - val_loss: 0.0077 - val_categorical_accuracy: 0.9590
Epoch 21/65535
420/420 [==============================] - 210s 501ms/step - loss: 0.0049 - categorical_accuracy: 0.9750 - val_loss: 0.0077 - val_categorical_accuracy: 0.9588
Epoch 22/65535
420/420 [==============================] - 210s 499ms/step - loss: 0.0048 - categorical_accuracy: 0.9752 - val_loss: 0.0078 - val_categorical_accuracy: 0.9585
Epoch 23/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0047 - categorical_accuracy: 0.9757 - val_loss: 0.0076 - val_categorical_accuracy: 0.9593
Epoch 24/65535
420/420 [==============================] - 210s 501ms/step - loss: 0.0047 - categorical_accuracy: 0.9759 - val_loss: 0.0078 - val_categorical_accuracy: 0.9591
Epoch 25/65535
420/420 [==============================] - 207s 492ms/step - loss: 0.0046 - categorical_accuracy: 0.9757 - val_loss: 0.0077 - val_categorical_accuracy: 0.9587
Epoch 26/65535
420/420 [==============================] - 208s 494ms/step - loss: 0.0045 - categorical_accuracy: 0.9770 - val_loss: 0.0078 - val_categorical_accuracy: 0.9588
Epoch 27/65535
420/420 [==============================] - 212s 504ms/step - loss: 0.0046 - categorical_accuracy: 0.9766 - val_loss: 0.0078 - val_categorical_accuracy: 0.9589
Epoch 28/65535
420/420 [==============================] - 212s 506ms/step - loss: 0.0045 - categorical_accuracy: 0.9767 - val_loss: 0.0078 - val_categorical_accuracy: 0.9591
Epoch 29/65535
420/420 [==============================] - 210s 501ms/step - loss: 0.0045 - categorical_accuracy: 0.9767 - val_loss: 0.0078 - val_categorical_accuracy: 0.9588

Epoch 00029: ReduceLROnPlateau reducing learning rate to 3.9999996079131965e-05.
Epoch 30/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0043 - categorical_accuracy: 0.9773 - val_loss: 0.0076 - val_categorical_accuracy: 0.9598
Epoch 31/65535
420/420 [==============================] - 207s 494ms/step - loss: 0.0043 - categorical_accuracy: 0.9782 - val_loss: 0.0076 - val_categorical_accuracy: 0.9600
Epoch 32/65535
420/420 [==============================] - 207s 494ms/step - loss: 0.0043 - categorical_accuracy: 0.9779 - val_loss: 0.0076 - val_categorical_accuracy: 0.9601
Epoch 33/65535
420/420 [==============================] - 210s 499ms/step - loss: 0.0043 - categorical_accuracy: 0.9782 - val_loss: 0.0076 - val_categorical_accuracy: 0.9597
Epoch 34/65535
420/420 [==============================] - 209s 498ms/step - loss: 0.0043 - categorical_accuracy: 0.9779 - val_loss: 0.0076 - val_categorical_accuracy: 0.9601

Epoch 00034: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 35/65535
420/420 [==============================] - 211s 502ms/step - loss: 0.0042 - categorical_accuracy: 0.9783 - val_loss: 0.0076 - val_categorical_accuracy: 0.9600
Epoch 36/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0042 - categorical_accuracy: 0.9784 - val_loss: 0.0076 - val_categorical_accuracy: 0.9599
Epoch 37/65535
420/420 [==============================] - 210s 500ms/step - loss: 0.0042 - categorical_accuracy: 0.9781 - val_loss: 0.0076 - val_categorical_accuracy: 0.9599
Epoch 38/65535
420/420 [==============================] - 210s 501ms/step - loss: 0.0042 - categorical_accuracy: 0.9785 - val_loss: 0.0076 - val_categorical_accuracy: 0.9600
Epoch 39/65535
420/420 [==============================] - 211s 502ms/step - loss: 0.0042 - categorical_accuracy: 0.9786 - val_loss: 0.0076 - val_categorical_accuracy: 0.9593
Epoch 40/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0042 - categorical_accuracy: 0.9788 - val_loss: 0.0076 - val_categorical_accuracy: 0.9597
Epoch 41/65535
420/420 [==============================] - 206s 490ms/step - loss: 0.0042 - categorical_accuracy: 0.9786 - val_loss: 0.0076 - val_categorical_accuracy: 0.9597
Epoch 42/65535
420/420 [==============================] - 205s 488ms/step - loss: 0.0042 - categorical_accuracy: 0.9782 - val_loss: 0.0076 - val_categorical_accuracy: 0.9600
Epoch 43/65535
420/420 [==============================] - 205s 488ms/step - loss: 0.0042 - categorical_accuracy: 0.9783 - val_loss: 0.0075 - val_categorical_accuracy: 0.9597
Epoch 44/65535
420/420 [==============================] - 206s 490ms/step - loss: 0.0042 - categorical_accuracy: 0.9781 - val_loss: 0.0075 - val_categorical_accuracy: 0.9597
Epoch 45/65535
420/420 [==============================] - 210s 499ms/step - loss: 0.0042 - categorical_accuracy: 0.9781 - val_loss: 0.0075 - val_categorical_accuracy: 0.9597
Epoch 46/65535
420/420 [==============================] - 205s 489ms/step - loss: 0.0042 - categorical_accuracy: 0.9782 - val_loss: 0.0075 - val_categorical_accuracy: 0.9596
Epoch 47/65535
420/420 [==============================] - 206s 490ms/step - loss: 0.0042 - categorical_accuracy: 0.9783 - val_loss: 0.0075 - val_categorical_accuracy: 0.9596
Epoch 48/65535
420/420 [==============================] - 206s 490ms/step - loss: 0.0042 - categorical_accuracy: 0.9784 - val_loss: 0.0076 - val_categorical_accuracy: 0.9597
Epoch 49/65535
420/420 [==============================] - 208s 496ms/step - loss: 0.0042 - categorical_accuracy: 0.9785 - val_loss: 0.0076 - val_categorical_accuracy: 0.9596
Epoch 50/65535
420/420 [==============================] - 207s 493ms/step - loss: 0.0042 - categorical_accuracy: 0.9783 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 51/65535
420/420 [==============================] - 209s 499ms/step - loss: 0.0041 - categorical_accuracy: 0.9785 - val_loss: 0.0076 - val_categorical_accuracy: 0.9598
Epoch 52/65535
420/420 [==============================] - 220s 524ms/step - loss: 0.0042 - categorical_accuracy: 0.9785 - val_loss: 0.0076 - val_categorical_accuracy: 0.9596
Epoch 53/65535
420/420 [==============================] - 220s 524ms/step - loss: 0.0042 - categorical_accuracy: 0.9788 - val_loss: 0.0076 - val_categorical_accuracy: 0.9597
Epoch 54/65535
420/420 [==============================] - 220s 523ms/step - loss: 0.0042 - categorical_accuracy: 0.9791 - val_loss: 0.0076 - val_categorical_accuracy: 0.9597
Epoch 55/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0042 - categorical_accuracy: 0.9782 - val_loss: 0.0076 - val_categorical_accuracy: 0.9596
Epoch 56/65535
420/420 [==============================] - 218s 519ms/step - loss: 0.0041 - categorical_accuracy: 0.9787 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 57/65535
420/420 [==============================] - 220s 524ms/step - loss: 0.0042 - categorical_accuracy: 0.9788 - val_loss: 0.0076 - val_categorical_accuracy: 0.9599
Epoch 58/65535
420/420 [==============================] - 216s 515ms/step - loss: 0.0041 - categorical_accuracy: 0.9790 - val_loss: 0.0076 - val_categorical_accuracy: 0.9599
Epoch 59/65535
420/420 [==============================] - 217s 516ms/step - loss: 0.0042 - categorical_accuracy: 0.9781 - val_loss: 0.0076 - val_categorical_accuracy: 0.9597
Epoch 60/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0041 - categorical_accuracy: 0.9788 - val_loss: 0.0076 - val_categorical_accuracy: 0.9597
Epoch 00060: early stopping
========= generating oof predictions 18:28:04 =========
========= generating test set predictions 18:28:14 =========
========= fitting 2 th model 18:30:16 =========
Epoch 1/65535
420/420 [==============================] - 234s 556ms/step - loss: 0.1022 - categorical_accuracy: 0.3496 - val_loss: 0.0628 - val_categorical_accuracy: 0.6254
Epoch 2/65535
420/420 [==============================] - 216s 515ms/step - loss: 0.0344 - categorical_accuracy: 0.8190 - val_loss: 0.0314 - val_categorical_accuracy: 0.8258
Epoch 3/65535
420/420 [==============================] - 215s 513ms/step - loss: 0.0217 - categorical_accuracy: 0.8841 - val_loss: 0.0352 - val_categorical_accuracy: 0.8116
Epoch 4/65535
420/420 [==============================] - 219s 520ms/step - loss: 0.0176 - categorical_accuracy: 0.9048 - val_loss: 0.0295 - val_categorical_accuracy: 0.8469
Epoch 5/65535
420/420 [==============================] - 217s 518ms/step - loss: 0.0154 - categorical_accuracy: 0.9160 - val_loss: 0.0230 - val_categorical_accuracy: 0.8751
Epoch 6/65535
420/420 [==============================] - 220s 523ms/step - loss: 0.0140 - categorical_accuracy: 0.9244 - val_loss: 0.0200 - val_categorical_accuracy: 0.8951
Epoch 7/65535
420/420 [==============================] - 222s 528ms/step - loss: 0.0129 - categorical_accuracy: 0.9293 - val_loss: 0.0210 - val_categorical_accuracy: 0.8846
Epoch 8/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.0121 - categorical_accuracy: 0.9337 - val_loss: 0.0177 - val_categorical_accuracy: 0.9060
Epoch 9/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0114 - categorical_accuracy: 0.9373 - val_loss: 0.0166 - val_categorical_accuracy: 0.9119
Epoch 10/65535
420/420 [==============================] - 215s 513ms/step - loss: 0.0109 - categorical_accuracy: 0.9406 - val_loss: 0.0185 - val_categorical_accuracy: 0.9057
Epoch 11/65535
420/420 [==============================] - 217s 516ms/step - loss: 0.0103 - categorical_accuracy: 0.9428 - val_loss: 0.0214 - val_categorical_accuracy: 0.8923
Epoch 12/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0098 - categorical_accuracy: 0.9461 - val_loss: 0.0212 - val_categorical_accuracy: 0.9039
Epoch 13/65535
420/420 [==============================] - 213s 507ms/step - loss: 0.0098 - categorical_accuracy: 0.9455 - val_loss: 0.0189 - val_categorical_accuracy: 0.9081
Epoch 14/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0093 - categorical_accuracy: 0.9488 - val_loss: 0.0421 - val_categorical_accuracy: 0.8484
Epoch 15/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0087 - categorical_accuracy: 0.9519 - val_loss: 0.0147 - val_categorical_accuracy: 0.9273
Epoch 16/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0089 - categorical_accuracy: 0.9500 - val_loss: 0.0164 - val_categorical_accuracy: 0.9195
Epoch 17/65535
420/420 [==============================] - 204s 487ms/step - loss: 0.0084 - categorical_accuracy: 0.9540 - val_loss: 0.0166 - val_categorical_accuracy: 0.9157
Epoch 18/65535
420/420 [==============================] - 201s 479ms/step - loss: 0.0085 - categorical_accuracy: 0.9533 - val_loss: 0.0214 - val_categorical_accuracy: 0.9060
Epoch 19/65535
420/420 [==============================] - 203s 482ms/step - loss: 0.0080 - categorical_accuracy: 0.9559 - val_loss: 0.0172 - val_categorical_accuracy: 0.9145
Epoch 20/65535
420/420 [==============================] - 208s 495ms/step - loss: 0.0078 - categorical_accuracy: 0.9565 - val_loss: 0.0239 - val_categorical_accuracy: 0.8875
Epoch 21/65535
420/420 [==============================] - 202s 481ms/step - loss: 0.0077 - categorical_accuracy: 0.9576 - val_loss: 0.0150 - val_categorical_accuracy: 0.9291

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.0009999999776482583.
Epoch 22/65535
420/420 [==============================] - 199s 475ms/step - loss: 0.0055 - categorical_accuracy: 0.9698 - val_loss: 0.0087 - val_categorical_accuracy: 0.9544
Epoch 23/65535
420/420 [==============================] - 201s 479ms/step - loss: 0.0050 - categorical_accuracy: 0.9728 - val_loss: 0.0089 - val_categorical_accuracy: 0.9544
Epoch 24/65535
420/420 [==============================] - 199s 475ms/step - loss: 0.0048 - categorical_accuracy: 0.9746 - val_loss: 0.0083 - val_categorical_accuracy: 0.9562
Epoch 25/65535
420/420 [==============================] - 201s 478ms/step - loss: 0.0045 - categorical_accuracy: 0.9759 - val_loss: 0.0085 - val_categorical_accuracy: 0.9546
Epoch 26/65535
420/420 [==============================] - 199s 473ms/step - loss: 0.0044 - categorical_accuracy: 0.9766 - val_loss: 0.0090 - val_categorical_accuracy: 0.9525
Epoch 27/65535
420/420 [==============================] - 205s 487ms/step - loss: 0.0044 - categorical_accuracy: 0.9763 - val_loss: 0.0093 - val_categorical_accuracy: 0.9532
Epoch 28/65535
420/420 [==============================] - 202s 482ms/step - loss: 0.0042 - categorical_accuracy: 0.9776 - val_loss: 0.0089 - val_categorical_accuracy: 0.9543
Epoch 29/65535
420/420 [==============================] - 200s 477ms/step - loss: 0.0040 - categorical_accuracy: 0.9789 - val_loss: 0.0089 - val_categorical_accuracy: 0.9541
Epoch 30/65535
420/420 [==============================] - 199s 474ms/step - loss: 0.0040 - categorical_accuracy: 0.9787 - val_loss: 0.0097 - val_categorical_accuracy: 0.9518

Epoch 00030: ReduceLROnPlateau reducing learning rate to 0.0001999999862164259.
Epoch 31/65535
420/420 [==============================] - 200s 476ms/step - loss: 0.0037 - categorical_accuracy: 0.9803 - val_loss: 0.0083 - val_categorical_accuracy: 0.9571
Epoch 32/65535
420/420 [==============================] - 200s 476ms/step - loss: 0.0035 - categorical_accuracy: 0.9815 - val_loss: 0.0082 - val_categorical_accuracy: 0.9582
Epoch 33/65535
420/420 [==============================] - 205s 487ms/step - loss: 0.0034 - categorical_accuracy: 0.9823 - val_loss: 0.0085 - val_categorical_accuracy: 0.9574
Epoch 34/65535
420/420 [==============================] - 199s 475ms/step - loss: 0.0034 - categorical_accuracy: 0.9820 - val_loss: 0.0083 - val_categorical_accuracy: 0.9580
Epoch 35/65535
420/420 [==============================] - 198s 471ms/step - loss: 0.0034 - categorical_accuracy: 0.9816 - val_loss: 0.0084 - val_categorical_accuracy: 0.9574

Epoch 00035: ReduceLROnPlateau reducing learning rate to 3.9999996079131965e-05.
Epoch 36/65535
420/420 [==============================] - 197s 470ms/step - loss: 0.0033 - categorical_accuracy: 0.9821 - val_loss: 0.0082 - val_categorical_accuracy: 0.9585
Epoch 37/65535
420/420 [==============================] - 200s 476ms/step - loss: 0.0033 - categorical_accuracy: 0.9833 - val_loss: 0.0082 - val_categorical_accuracy: 0.9585
Epoch 38/65535
420/420 [==============================] - 203s 483ms/step - loss: 0.0033 - categorical_accuracy: 0.9827 - val_loss: 0.0082 - val_categorical_accuracy: 0.9582
Epoch 39/65535
420/420 [==============================] - 202s 480ms/step - loss: 0.0032 - categorical_accuracy: 0.9834 - val_loss: 0.0082 - val_categorical_accuracy: 0.9580
Epoch 40/65535
420/420 [==============================] - 203s 483ms/step - loss: 0.0032 - categorical_accuracy: 0.9835 - val_loss: 0.0082 - val_categorical_accuracy: 0.9581
Epoch 41/65535
420/420 [==============================] - 202s 481ms/step - loss: 0.0032 - categorical_accuracy: 0.9832 - val_loss: 0.0082 - val_categorical_accuracy: 0.9583
Epoch 42/65535
420/420 [==============================] - 205s 487ms/step - loss: 0.0033 - categorical_accuracy: 0.9831 - val_loss: 0.0082 - val_categorical_accuracy: 0.9579
Epoch 43/65535
420/420 [==============================] - 204s 486ms/step - loss: 0.0032 - categorical_accuracy: 0.9834 - val_loss: 0.0082 - val_categorical_accuracy: 0.9580

Epoch 00043: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 44/65535
420/420 [==============================] - 209s 497ms/step - loss: 0.0033 - categorical_accuracy: 0.9832 - val_loss: 0.0082 - val_categorical_accuracy: 0.9582
Epoch 45/65535
420/420 [==============================] - 202s 482ms/step - loss: 0.0032 - categorical_accuracy: 0.9833 - val_loss: 0.0082 - val_categorical_accuracy: 0.9585
Epoch 46/65535
420/420 [==============================] - 198s 471ms/step - loss: 0.0033 - categorical_accuracy: 0.9836 - val_loss: 0.0082 - val_categorical_accuracy: 0.9582
Epoch 47/65535
420/420 [==============================] - 204s 486ms/step - loss: 0.0032 - categorical_accuracy: 0.9836 - val_loss: 0.0082 - val_categorical_accuracy: 0.9583
Epoch 48/65535
420/420 [==============================] - 201s 478ms/step - loss: 0.0032 - categorical_accuracy: 0.9833 - val_loss: 0.0082 - val_categorical_accuracy: 0.9584
Epoch 49/65535
420/420 [==============================] - 198s 472ms/step - loss: 0.0032 - categorical_accuracy: 0.9832 - val_loss: 0.0082 - val_categorical_accuracy: 0.9583
Epoch 50/65535
420/420 [==============================] - 199s 473ms/step - loss: 0.0032 - categorical_accuracy: 0.9831 - val_loss: 0.0082 - val_categorical_accuracy: 0.9583
Epoch 51/65535
420/420 [==============================] - 205s 487ms/step - loss: 0.0032 - categorical_accuracy: 0.9836 - val_loss: 0.0082 - val_categorical_accuracy: 0.9582
Epoch 52/65535
420/420 [==============================] - 210s 499ms/step - loss: 0.0032 - categorical_accuracy: 0.9837 - val_loss: 0.0082 - val_categorical_accuracy: 0.9584
Epoch 00052: early stopping
========= generating oof predictions 21:29:29 =========
========= generating test set predictions 21:29:39 =========
========= fitting 3 th model 21:31:38 =========
Epoch 1/65535
420/420 [==============================] - 224s 533ms/step - loss: 0.1072 - categorical_accuracy: 0.3131 - val_loss: 0.0630 - val_categorical_accuracy: 0.6048
Epoch 2/65535
420/420 [==============================] - 197s 469ms/step - loss: 0.0367 - categorical_accuracy: 0.8047 - val_loss: 0.0316 - val_categorical_accuracy: 0.8227
Epoch 3/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0223 - categorical_accuracy: 0.8801 - val_loss: 0.0235 - val_categorical_accuracy: 0.8716
Epoch 4/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0183 - categorical_accuracy: 0.9018 - val_loss: 0.0210 - val_categorical_accuracy: 0.8856
Epoch 5/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0157 - categorical_accuracy: 0.9151 - val_loss: 0.0252 - val_categorical_accuracy: 0.8704
Epoch 6/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0144 - categorical_accuracy: 0.9215 - val_loss: 0.0168 - val_categorical_accuracy: 0.9110
Epoch 7/65535
420/420 [==============================] - 196s 466ms/step - loss: 0.0132 - categorical_accuracy: 0.9283 - val_loss: 0.0189 - val_categorical_accuracy: 0.8950
Epoch 8/65535
420/420 [==============================] - 199s 473ms/step - loss: 0.0122 - categorical_accuracy: 0.9327 - val_loss: 0.0178 - val_categorical_accuracy: 0.9082
Epoch 9/65535
420/420 [==============================] - 194s 463ms/step - loss: 0.0116 - categorical_accuracy: 0.9364 - val_loss: 0.0184 - val_categorical_accuracy: 0.9015
Epoch 10/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0112 - categorical_accuracy: 0.9382 - val_loss: 0.0190 - val_categorical_accuracy: 0.9066
Epoch 11/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0106 - categorical_accuracy: 0.9409 - val_loss: 0.0141 - val_categorical_accuracy: 0.9279
Epoch 12/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0101 - categorical_accuracy: 0.9440 - val_loss: 0.0138 - val_categorical_accuracy: 0.9296
Epoch 13/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0099 - categorical_accuracy: 0.9456 - val_loss: 0.0166 - val_categorical_accuracy: 0.9145
Epoch 14/65535
420/420 [==============================] - 200s 476ms/step - loss: 0.0093 - categorical_accuracy: 0.9490 - val_loss: 0.0148 - val_categorical_accuracy: 0.9230
Epoch 15/65535
420/420 [==============================] - 199s 474ms/step - loss: 0.0091 - categorical_accuracy: 0.9496 - val_loss: 0.0171 - val_categorical_accuracy: 0.9156
Epoch 16/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0089 - categorical_accuracy: 0.9504 - val_loss: 0.0178 - val_categorical_accuracy: 0.9198
Epoch 17/65535
420/420 [==============================] - 197s 470ms/step - loss: 0.0089 - categorical_accuracy: 0.9504 - val_loss: 0.0143 - val_categorical_accuracy: 0.9264
Epoch 18/65535
420/420 [==============================] - 196s 466ms/step - loss: 0.0082 - categorical_accuracy: 0.9538 - val_loss: 0.0160 - val_categorical_accuracy: 0.9231

Epoch 00018: ReduceLROnPlateau reducing learning rate to 0.0009999999776482583.
Epoch 19/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0059 - categorical_accuracy: 0.9676 - val_loss: 0.0086 - val_categorical_accuracy: 0.9542
Epoch 20/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0054 - categorical_accuracy: 0.9703 - val_loss: 0.0081 - val_categorical_accuracy: 0.9560
Epoch 21/65535
420/420 [==============================] - 199s 473ms/step - loss: 0.0052 - categorical_accuracy: 0.9716 - val_loss: 0.0082 - val_categorical_accuracy: 0.9569
Epoch 22/65535
420/420 [==============================] - 196s 468ms/step - loss: 0.0050 - categorical_accuracy: 0.9728 - val_loss: 0.0084 - val_categorical_accuracy: 0.9547
Epoch 23/65535
420/420 [==============================] - 197s 469ms/step - loss: 0.0049 - categorical_accuracy: 0.9730 - val_loss: 0.0086 - val_categorical_accuracy: 0.9551
Epoch 24/65535
420/420 [==============================] - 195s 463ms/step - loss: 0.0047 - categorical_accuracy: 0.9746 - val_loss: 0.0085 - val_categorical_accuracy: 0.9550
Epoch 25/65535
420/420 [==============================] - 193s 460ms/step - loss: 0.0047 - categorical_accuracy: 0.9745 - val_loss: 0.0090 - val_categorical_accuracy: 0.9533
Epoch 26/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0045 - categorical_accuracy: 0.9753 - val_loss: 0.0085 - val_categorical_accuracy: 0.9550

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.0001999999862164259.
Epoch 27/65535
420/420 [==============================] - 194s 463ms/step - loss: 0.0041 - categorical_accuracy: 0.9778 - val_loss: 0.0077 - val_categorical_accuracy: 0.9591
Epoch 28/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0040 - categorical_accuracy: 0.9792 - val_loss: 0.0076 - val_categorical_accuracy: 0.9596
Epoch 29/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0039 - categorical_accuracy: 0.9793 - val_loss: 0.0076 - val_categorical_accuracy: 0.9589
Epoch 30/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0038 - categorical_accuracy: 0.9797 - val_loss: 0.0076 - val_categorical_accuracy: 0.9598
Epoch 31/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0038 - categorical_accuracy: 0.9794 - val_loss: 0.0077 - val_categorical_accuracy: 0.9580
Epoch 32/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0038 - categorical_accuracy: 0.9799 - val_loss: 0.0077 - val_categorical_accuracy: 0.9587
Epoch 33/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0037 - categorical_accuracy: 0.9804 - val_loss: 0.0075 - val_categorical_accuracy: 0.9595
Epoch 34/65535
420/420 [==============================] - 193s 461ms/step - loss: 0.0037 - categorical_accuracy: 0.9810 - val_loss: 0.0077 - val_categorical_accuracy: 0.9592

Epoch 00034: ReduceLROnPlateau reducing learning rate to 3.9999996079131965e-05.
Epoch 35/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0036 - categorical_accuracy: 0.9805 - val_loss: 0.0075 - val_categorical_accuracy: 0.9603
Epoch 36/65535
420/420 [==============================] - 199s 473ms/step - loss: 0.0036 - categorical_accuracy: 0.9812 - val_loss: 0.0075 - val_categorical_accuracy: 0.9606
Epoch 37/65535
420/420 [==============================] - 191s 455ms/step - loss: 0.0036 - categorical_accuracy: 0.9815 - val_loss: 0.0074 - val_categorical_accuracy: 0.9604
Epoch 38/65535
420/420 [==============================] - 193s 460ms/step - loss: 0.0036 - categorical_accuracy: 0.9812 - val_loss: 0.0074 - val_categorical_accuracy: 0.9608
Epoch 39/65535
420/420 [==============================] - 193s 460ms/step - loss: 0.0036 - categorical_accuracy: 0.9810 - val_loss: 0.0074 - val_categorical_accuracy: 0.9608
Epoch 40/65535
420/420 [==============================] - 193s 459ms/step - loss: 0.0035 - categorical_accuracy: 0.9809 - val_loss: 0.0074 - val_categorical_accuracy: 0.9607
Epoch 41/65535
420/420 [==============================] - 193s 460ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0074 - val_categorical_accuracy: 0.9608

Epoch 00041: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 42/65535
420/420 [==============================] - 191s 456ms/step - loss: 0.0035 - categorical_accuracy: 0.9818 - val_loss: 0.0074 - val_categorical_accuracy: 0.9609
Epoch 43/65535
420/420 [==============================] - 192s 456ms/step - loss: 0.0035 - categorical_accuracy: 0.9818 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 44/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0035 - categorical_accuracy: 0.9820 - val_loss: 0.0074 - val_categorical_accuracy: 0.9610
Epoch 45/65535
420/420 [==============================] - 193s 459ms/step - loss: 0.0035 - categorical_accuracy: 0.9814 - val_loss: 0.0074 - val_categorical_accuracy: 0.9610
Epoch 46/65535
420/420 [==============================] - 192s 457ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0074 - val_categorical_accuracy: 0.9610
Epoch 47/65535
420/420 [==============================] - 192s 458ms/step - loss: 0.0035 - categorical_accuracy: 0.9814 - val_loss: 0.0074 - val_categorical_accuracy: 0.9610
Epoch 48/65535
420/420 [==============================] - 192s 456ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0074 - val_categorical_accuracy: 0.9610
Epoch 49/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0035 - categorical_accuracy: 0.9815 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 50/65535
420/420 [==============================] - 192s 458ms/step - loss: 0.0035 - categorical_accuracy: 0.9822 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 51/65535
420/420 [==============================] - 192s 457ms/step - loss: 0.0035 - categorical_accuracy: 0.9814 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 52/65535
420/420 [==============================] - 192s 456ms/step - loss: 0.0035 - categorical_accuracy: 0.9817 - val_loss: 0.0074 - val_categorical_accuracy: 0.9614
Epoch 53/65535
420/420 [==============================] - 192s 456ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 54/65535
420/420 [==============================] - 192s 458ms/step - loss: 0.0035 - categorical_accuracy: 0.9815 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 55/65535
420/420 [==============================] - 192s 458ms/step - loss: 0.0035 - categorical_accuracy: 0.9818 - val_loss: 0.0074 - val_categorical_accuracy: 0.9610
Epoch 56/65535
420/420 [==============================] - 193s 460ms/step - loss: 0.0035 - categorical_accuracy: 0.9814 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 57/65535
420/420 [==============================] - 193s 458ms/step - loss: 0.0035 - categorical_accuracy: 0.9821 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 58/65535
420/420 [==============================] - 191s 455ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 59/65535
420/420 [==============================] - 193s 458ms/step - loss: 0.0035 - categorical_accuracy: 0.9817 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 60/65535
420/420 [==============================] - 192s 457ms/step - loss: 0.0035 - categorical_accuracy: 0.9824 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 61/65535
420/420 [==============================] - 192s 456ms/step - loss: 0.0034 - categorical_accuracy: 0.9821 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 62/65535
420/420 [==============================] - 192s 457ms/step - loss: 0.0035 - categorical_accuracy: 0.9822 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 63/65535
420/420 [==============================] - 192s 457ms/step - loss: 0.0035 - categorical_accuracy: 0.9821 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 64/65535
420/420 [==============================] - 191s 456ms/step - loss: 0.0035 - categorical_accuracy: 0.9815 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 65/65535
420/420 [==============================] - 192s 456ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 66/65535
420/420 [==============================] - 192s 457ms/step - loss: 0.0034 - categorical_accuracy: 0.9820 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 67/65535
420/420 [==============================] - 191s 455ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 68/65535
420/420 [==============================] - 192s 457ms/step - loss: 0.0035 - categorical_accuracy: 0.9820 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 69/65535
420/420 [==============================] - 191s 455ms/step - loss: 0.0035 - categorical_accuracy: 0.9821 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 70/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0035 - categorical_accuracy: 0.9816 - val_loss: 0.0074 - val_categorical_accuracy: 0.9610
Epoch 71/65535
420/420 [==============================] - 194s 463ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0074 - val_categorical_accuracy: 0.9611
Epoch 72/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0034 - categorical_accuracy: 0.9821 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 73/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0034 - categorical_accuracy: 0.9820 - val_loss: 0.0074 - val_categorical_accuracy: 0.9614
Epoch 74/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0034 - categorical_accuracy: 0.9818 - val_loss: 0.0074 - val_categorical_accuracy: 0.9614
Epoch 75/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0034 - categorical_accuracy: 0.9826 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 76/65535
420/420 [==============================] - 194s 463ms/step - loss: 0.0034 - categorical_accuracy: 0.9820 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 77/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0034 - categorical_accuracy: 0.9824 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 78/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0035 - categorical_accuracy: 0.9820 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 79/65535
420/420 [==============================] - 193s 460ms/step - loss: 0.0034 - categorical_accuracy: 0.9821 - val_loss: 0.0074 - val_categorical_accuracy: 0.9615
Epoch 80/65535
420/420 [==============================] - 193s 460ms/step - loss: 0.0034 - categorical_accuracy: 0.9823 - val_loss: 0.0074 - val_categorical_accuracy: 0.9614
Epoch 81/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0034 - categorical_accuracy: 0.9822 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 82/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0034 - categorical_accuracy: 0.9824 - val_loss: 0.0074 - val_categorical_accuracy: 0.9613
Epoch 83/65535
420/420 [==============================] - 194s 463ms/step - loss: 0.0034 - categorical_accuracy: 0.9818 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 84/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0034 - categorical_accuracy: 0.9821 - val_loss: 0.0074 - val_categorical_accuracy: 0.9612
Epoch 00084: early stopping
========= generating oof predictions 02:03:35 =========
========= generating test set predictions 02:03:45 =========
========= fitting 4 th model 02:05:40 =========
Epoch 1/65535
420/420 [==============================] - 200s 477ms/step - loss: 0.1095 - categorical_accuracy: 0.2967 - val_loss: 0.1137 - val_categorical_accuracy: 0.4064
Epoch 2/65535
420/420 [==============================] - 218s 519ms/step - loss: 0.0391 - categorical_accuracy: 0.7935 - val_loss: 0.0305 - val_categorical_accuracy: 0.8309
Epoch 3/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0229 - categorical_accuracy: 0.8773 - val_loss: 0.0360 - val_categorical_accuracy: 0.8026
Epoch 4/65535
420/420 [==============================] - 218s 518ms/step - loss: 0.0184 - categorical_accuracy: 0.9016 - val_loss: 0.0229 - val_categorical_accuracy: 0.8852
Epoch 5/65535
420/420 [==============================] - 217s 516ms/step - loss: 0.0158 - categorical_accuracy: 0.9152 - val_loss: 0.0284 - val_categorical_accuracy: 0.8693
Epoch 6/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0145 - categorical_accuracy: 0.9209 - val_loss: 0.0259 - val_categorical_accuracy: 0.8708
Epoch 7/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0133 - categorical_accuracy: 0.9268 - val_loss: 0.0217 - val_categorical_accuracy: 0.8837
Epoch 8/65535
420/420 [==============================] - 218s 519ms/step - loss: 0.0124 - categorical_accuracy: 0.9315 - val_loss: 0.0156 - val_categorical_accuracy: 0.9143
Epoch 9/65535
420/420 [==============================] - 218s 519ms/step - loss: 0.0117 - categorical_accuracy: 0.9360 - val_loss: 0.0168 - val_categorical_accuracy: 0.9120
Epoch 10/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0112 - categorical_accuracy: 0.9381 - val_loss: 0.0157 - val_categorical_accuracy: 0.9164
Epoch 11/65535
420/420 [==============================] - 218s 518ms/step - loss: 0.0108 - categorical_accuracy: 0.9401 - val_loss: 0.0182 - val_categorical_accuracy: 0.9059
Epoch 12/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0102 - categorical_accuracy: 0.9437 - val_loss: 0.0157 - val_categorical_accuracy: 0.9180
Epoch 13/65535
420/420 [==============================] - 219s 520ms/step - loss: 0.0097 - categorical_accuracy: 0.9456 - val_loss: 0.0186 - val_categorical_accuracy: 0.9059
Epoch 14/65535
420/420 [==============================] - 218s 518ms/step - loss: 0.0094 - categorical_accuracy: 0.9482 - val_loss: 0.0176 - val_categorical_accuracy: 0.9128

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.0009999999776482583.
Epoch 15/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0067 - categorical_accuracy: 0.9635 - val_loss: 0.0087 - val_categorical_accuracy: 0.9527
Epoch 16/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0061 - categorical_accuracy: 0.9668 - val_loss: 0.0085 - val_categorical_accuracy: 0.9545
Epoch 17/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0058 - categorical_accuracy: 0.9680 - val_loss: 0.0084 - val_categorical_accuracy: 0.9536
Epoch 18/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0057 - categorical_accuracy: 0.9693 - val_loss: 0.0087 - val_categorical_accuracy: 0.9524
Epoch 19/65535
420/420 [==============================] - 217s 517ms/step - loss: 0.0054 - categorical_accuracy: 0.9706 - val_loss: 0.0090 - val_categorical_accuracy: 0.9507
Epoch 20/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0053 - categorical_accuracy: 0.9713 - val_loss: 0.0085 - val_categorical_accuracy: 0.9549
Epoch 21/65535
420/420 [==============================] - 217s 517ms/step - loss: 0.0052 - categorical_accuracy: 0.9719 - val_loss: 0.0087 - val_categorical_accuracy: 0.9530
Epoch 22/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0050 - categorical_accuracy: 0.9730 - val_loss: 0.0087 - val_categorical_accuracy: 0.9545

Epoch 00022: ReduceLROnPlateau reducing learning rate to 0.0001999999862164259.
Epoch 23/65535
420/420 [==============================] - 218s 518ms/step - loss: 0.0045 - categorical_accuracy: 0.9764 - val_loss: 0.0078 - val_categorical_accuracy: 0.9577
Epoch 24/65535
420/420 [==============================] - 219s 520ms/step - loss: 0.0044 - categorical_accuracy: 0.9771 - val_loss: 0.0077 - val_categorical_accuracy: 0.9576
Epoch 25/65535
420/420 [==============================] - 220s 524ms/step - loss: 0.0044 - categorical_accuracy: 0.9768 - val_loss: 0.0077 - val_categorical_accuracy: 0.9578
Epoch 26/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0042 - categorical_accuracy: 0.9780 - val_loss: 0.0078 - val_categorical_accuracy: 0.9577
Epoch 27/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0042 - categorical_accuracy: 0.9774 - val_loss: 0.0078 - val_categorical_accuracy: 0.9582
Epoch 28/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0042 - categorical_accuracy: 0.9779 - val_loss: 0.0078 - val_categorical_accuracy: 0.9574
Epoch 29/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0042 - categorical_accuracy: 0.9781 - val_loss: 0.0077 - val_categorical_accuracy: 0.9577

Epoch 00029: ReduceLROnPlateau reducing learning rate to 3.9999996079131965e-05.
Epoch 30/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0040 - categorical_accuracy: 0.9791 - val_loss: 0.0076 - val_categorical_accuracy: 0.9592
Epoch 31/65535
420/420 [==============================] - 217s 517ms/step - loss: 0.0040 - categorical_accuracy: 0.9797 - val_loss: 0.0076 - val_categorical_accuracy: 0.9589
Epoch 32/65535
420/420 [==============================] - 218s 518ms/step - loss: 0.0040 - categorical_accuracy: 0.9788 - val_loss: 0.0076 - val_categorical_accuracy: 0.9593
Epoch 33/65535
420/420 [==============================] - 217s 518ms/step - loss: 0.0040 - categorical_accuracy: 0.9793 - val_loss: 0.0076 - val_categorical_accuracy: 0.9592
Epoch 34/65535
420/420 [==============================] - 218s 519ms/step - loss: 0.0039 - categorical_accuracy: 0.9795 - val_loss: 0.0076 - val_categorical_accuracy: 0.9588
Epoch 35/65535
420/420 [==============================] - 219s 520ms/step - loss: 0.0039 - categorical_accuracy: 0.9796 - val_loss: 0.0076 - val_categorical_accuracy: 0.9586
Epoch 36/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0039 - categorical_accuracy: 0.9796 - val_loss: 0.0076 - val_categorical_accuracy: 0.9591

Epoch 00036: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 37/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0040 - categorical_accuracy: 0.9792 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 38/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0039 - categorical_accuracy: 0.9800 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 39/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0039 - categorical_accuracy: 0.9797 - val_loss: 0.0076 - val_categorical_accuracy: 0.9593
Epoch 40/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0039 - categorical_accuracy: 0.9798 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 41/65535
420/420 [==============================] - 219s 520ms/step - loss: 0.0039 - categorical_accuracy: 0.9798 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 42/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0039 - categorical_accuracy: 0.9798 - val_loss: 0.0076 - val_categorical_accuracy: 0.9594
Epoch 43/65535
420/420 [==============================] - 219s 520ms/step - loss: 0.0039 - categorical_accuracy: 0.9793 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 44/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0039 - categorical_accuracy: 0.9794 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 45/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0039 - categorical_accuracy: 0.9801 - val_loss: 0.0076 - val_categorical_accuracy: 0.9596
Epoch 46/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0039 - categorical_accuracy: 0.9797 - val_loss: 0.0076 - val_categorical_accuracy: 0.9594
Epoch 47/65535
420/420 [==============================] - 218s 518ms/step - loss: 0.0038 - categorical_accuracy: 0.9800 - val_loss: 0.0076 - val_categorical_accuracy: 0.9595
Epoch 48/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0039 - categorical_accuracy: 0.9798 - val_loss: 0.0076 - val_categorical_accuracy: 0.9592
Epoch 49/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0038 - categorical_accuracy: 0.9809 - val_loss: 0.0076 - val_categorical_accuracy: 0.9593
Epoch 50/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0038 - categorical_accuracy: 0.9802 - val_loss: 0.0076 - val_categorical_accuracy: 0.9592
Epoch 51/65535
420/420 [==============================] - 218s 519ms/step - loss: 0.0039 - categorical_accuracy: 0.9797 - val_loss: 0.0076 - val_categorical_accuracy: 0.9592
Epoch 52/65535
420/420 [==============================] - 220s 524ms/step - loss: 0.0039 - categorical_accuracy: 0.9800 - val_loss: 0.0076 - val_categorical_accuracy: 0.9592
Epoch 53/65535
420/420 [==============================] - 223s 531ms/step - loss: 0.0039 - categorical_accuracy: 0.9798 - val_loss: 0.0076 - val_categorical_accuracy: 0.9592
Epoch 54/65535
420/420 [==============================] - 226s 538ms/step - loss: 0.0039 - categorical_accuracy: 0.9800 - val_loss: 0.0076 - val_categorical_accuracy: 0.9593
Epoch 55/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0039 - categorical_accuracy: 0.9797 - val_loss: 0.0076 - val_categorical_accuracy: 0.9591
Epoch 56/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0038 - categorical_accuracy: 0.9805 - val_loss: 0.0076 - val_categorical_accuracy: 0.9588
Epoch 57/65535
420/420 [==============================] - 217s 518ms/step - loss: 0.0038 - categorical_accuracy: 0.9797 - val_loss: 0.0076 - val_categorical_accuracy: 0.9590
Epoch 58/65535
420/420 [==============================] - 218s 518ms/step - loss: 0.0039 - categorical_accuracy: 0.9794 - val_loss: 0.0076 - val_categorical_accuracy: 0.9590
Epoch 59/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0038 - categorical_accuracy: 0.9802 - val_loss: 0.0076 - val_categorical_accuracy: 0.9592
Epoch 60/65535
420/420 [==============================] - 218s 519ms/step - loss: 0.0039 - categorical_accuracy: 0.9803 - val_loss: 0.0076 - val_categorical_accuracy: 0.9593
Epoch 61/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0039 - categorical_accuracy: 0.9804 - val_loss: 0.0076 - val_categorical_accuracy: 0.9593
Epoch 62/65535
420/420 [==============================] - 218s 520ms/step - loss: 0.0039 - categorical_accuracy: 0.9793 - val_loss: 0.0076 - val_categorical_accuracy: 0.9591
Epoch 63/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0038 - categorical_accuracy: 0.9800 - val_loss: 0.0076 - val_categorical_accuracy: 0.9591
Epoch 64/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0038 - categorical_accuracy: 0.9802 - val_loss: 0.0076 - val_categorical_accuracy: 0.9592
Epoch 00064: early stopping
========= generating oof predictions 05:58:32 =========
========= generating test set predictions 05:58:41 =========
========= fitting 5 th model 06:00:36 =========
Epoch 1/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.1000 - categorical_accuracy: 0.3682 - val_loss: 0.0772 - val_categorical_accuracy: 0.5375
Epoch 2/65535
420/420 [==============================] - 213s 507ms/step - loss: 0.0330 - categorical_accuracy: 0.8278 - val_loss: 0.0269 - val_categorical_accuracy: 0.8493
Epoch 3/65535
420/420 [==============================] - 213s 506ms/step - loss: 0.0213 - categorical_accuracy: 0.8862 - val_loss: 0.0269 - val_categorical_accuracy: 0.8560
Epoch 4/65535
420/420 [==============================] - 212s 506ms/step - loss: 0.0173 - categorical_accuracy: 0.9070 - val_loss: 0.0213 - val_categorical_accuracy: 0.8856
Epoch 5/65535
420/420 [==============================] - 213s 507ms/step - loss: 0.0151 - categorical_accuracy: 0.9186 - val_loss: 0.0240 - val_categorical_accuracy: 0.8765
Epoch 6/65535
420/420 [==============================] - 213s 506ms/step - loss: 0.0139 - categorical_accuracy: 0.9229 - val_loss: 0.0186 - val_categorical_accuracy: 0.9002
Epoch 7/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0128 - categorical_accuracy: 0.9293 - val_loss: 0.0203 - val_categorical_accuracy: 0.8892
Epoch 8/65535
420/420 [==============================] - 212s 505ms/step - loss: 0.0119 - categorical_accuracy: 0.9349 - val_loss: 0.0208 - val_categorical_accuracy: 0.8933
Epoch 9/65535
420/420 [==============================] - 213s 507ms/step - loss: 0.0115 - categorical_accuracy: 0.9360 - val_loss: 0.0234 - val_categorical_accuracy: 0.8829
Epoch 10/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0107 - categorical_accuracy: 0.9406 - val_loss: 0.0181 - val_categorical_accuracy: 0.9075
Epoch 11/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0104 - categorical_accuracy: 0.9427 - val_loss: 0.0149 - val_categorical_accuracy: 0.9207
Epoch 12/65535
420/420 [==============================] - 213s 507ms/step - loss: 0.0097 - categorical_accuracy: 0.9467 - val_loss: 0.0195 - val_categorical_accuracy: 0.9052
Epoch 13/65535
420/420 [==============================] - 213s 507ms/step - loss: 0.0095 - categorical_accuracy: 0.9476 - val_loss: 0.0170 - val_categorical_accuracy: 0.9137
Epoch 14/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0089 - categorical_accuracy: 0.9499 - val_loss: 0.0152 - val_categorical_accuracy: 0.9268
Epoch 15/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0087 - categorical_accuracy: 0.9517 - val_loss: 0.0179 - val_categorical_accuracy: 0.9189
Epoch 16/65535
420/420 [==============================] - 212s 505ms/step - loss: 0.0088 - categorical_accuracy: 0.9501 - val_loss: 0.0158 - val_categorical_accuracy: 0.9226
Epoch 17/65535
420/420 [==============================] - 212s 504ms/step - loss: 0.0083 - categorical_accuracy: 0.9538 - val_loss: 0.0195 - val_categorical_accuracy: 0.9127

Epoch 00017: ReduceLROnPlateau reducing learning rate to 0.0009999999776482583.
Epoch 18/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0058 - categorical_accuracy: 0.9674 - val_loss: 0.0091 - val_categorical_accuracy: 0.9527
Epoch 19/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0053 - categorical_accuracy: 0.9704 - val_loss: 0.0091 - val_categorical_accuracy: 0.9542
Epoch 20/65535
420/420 [==============================] - 213s 507ms/step - loss: 0.0050 - categorical_accuracy: 0.9728 - val_loss: 0.0087 - val_categorical_accuracy: 0.9544
Epoch 21/65535
420/420 [==============================] - 213s 506ms/step - loss: 0.0049 - categorical_accuracy: 0.9732 - val_loss: 0.0090 - val_categorical_accuracy: 0.9533
Epoch 22/65535
420/420 [==============================] - 211s 503ms/step - loss: 0.0048 - categorical_accuracy: 0.9740 - val_loss: 0.0094 - val_categorical_accuracy: 0.9530
Epoch 23/65535
420/420 [==============================] - 212s 504ms/step - loss: 0.0046 - categorical_accuracy: 0.9745 - val_loss: 0.0093 - val_categorical_accuracy: 0.9538
Epoch 24/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0045 - categorical_accuracy: 0.9755 - val_loss: 0.0087 - val_categorical_accuracy: 0.9557
Epoch 25/65535
420/420 [==============================] - 213s 507ms/step - loss: 0.0044 - categorical_accuracy: 0.9754 - val_loss: 0.0095 - val_categorical_accuracy: 0.9524
Epoch 26/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0043 - categorical_accuracy: 0.9769 - val_loss: 0.0090 - val_categorical_accuracy: 0.9552

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.0001999999862164259.
Epoch 27/65535
420/420 [==============================] - 218s 519ms/step - loss: 0.0038 - categorical_accuracy: 0.9793 - val_loss: 0.0082 - val_categorical_accuracy: 0.9591
Epoch 28/65535
420/420 [==============================] - 217s 516ms/step - loss: 0.0037 - categorical_accuracy: 0.9801 - val_loss: 0.0082 - val_categorical_accuracy: 0.9592
Epoch 29/65535
420/420 [==============================] - 212s 505ms/step - loss: 0.0036 - categorical_accuracy: 0.9808 - val_loss: 0.0082 - val_categorical_accuracy: 0.9591
Epoch 30/65535
420/420 [==============================] - 210s 501ms/step - loss: 0.0036 - categorical_accuracy: 0.9807 - val_loss: 0.0082 - val_categorical_accuracy: 0.9582
Epoch 31/65535
420/420 [==============================] - 211s 501ms/step - loss: 0.0036 - categorical_accuracy: 0.9809 - val_loss: 0.0083 - val_categorical_accuracy: 0.9584
Epoch 32/65535
420/420 [==============================] - 216s 514ms/step - loss: 0.0036 - categorical_accuracy: 0.9811 - val_loss: 0.0082 - val_categorical_accuracy: 0.9588
Epoch 33/65535
420/420 [==============================] - 219s 522ms/step - loss: 0.0035 - categorical_accuracy: 0.9816 - val_loss: 0.0083 - val_categorical_accuracy: 0.9582

Epoch 00033: ReduceLROnPlateau reducing learning rate to 3.9999996079131965e-05.
Epoch 34/65535
420/420 [==============================] - 220s 525ms/step - loss: 0.0034 - categorical_accuracy: 0.9821 - val_loss: 0.0081 - val_categorical_accuracy: 0.9588
Epoch 35/65535
420/420 [==============================] - 216s 515ms/step - loss: 0.0034 - categorical_accuracy: 0.9821 - val_loss: 0.0081 - val_categorical_accuracy: 0.9589
Epoch 36/65535
420/420 [==============================] - 216s 515ms/step - loss: 0.0034 - categorical_accuracy: 0.9821 - val_loss: 0.0081 - val_categorical_accuracy: 0.9591
Epoch 37/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0034 - categorical_accuracy: 0.9818 - val_loss: 0.0082 - val_categorical_accuracy: 0.9595
Epoch 38/65535
420/420 [==============================] - 210s 500ms/step - loss: 0.0034 - categorical_accuracy: 0.9822 - val_loss: 0.0081 - val_categorical_accuracy: 0.9591

Epoch 00038: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 39/65535
420/420 [==============================] - 218s 519ms/step - loss: 0.0034 - categorical_accuracy: 0.9822 - val_loss: 0.0081 - val_categorical_accuracy: 0.9594
Epoch 40/65535
420/420 [==============================] - 221s 527ms/step - loss: 0.0033 - categorical_accuracy: 0.9826 - val_loss: 0.0081 - val_categorical_accuracy: 0.9593
Epoch 41/65535
420/420 [==============================] - 219s 521ms/step - loss: 0.0033 - categorical_accuracy: 0.9826 - val_loss: 0.0081 - val_categorical_accuracy: 0.9594
Epoch 42/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0033 - categorical_accuracy: 0.9825 - val_loss: 0.0081 - val_categorical_accuracy: 0.9591
Epoch 43/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0033 - categorical_accuracy: 0.9821 - val_loss: 0.0081 - val_categorical_accuracy: 0.9592
Epoch 44/65535
420/420 [==============================] - 217s 517ms/step - loss: 0.0034 - categorical_accuracy: 0.9823 - val_loss: 0.0081 - val_categorical_accuracy: 0.9594
Epoch 45/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0033 - categorical_accuracy: 0.9827 - val_loss: 0.0081 - val_categorical_accuracy: 0.9591
Epoch 46/65535
420/420 [==============================] - 217s 517ms/step - loss: 0.0033 - categorical_accuracy: 0.9824 - val_loss: 0.0082 - val_categorical_accuracy: 0.9591
Epoch 47/65535
420/420 [==============================] - 225s 536ms/step - loss: 0.0033 - categorical_accuracy: 0.9822 - val_loss: 0.0082 - val_categorical_accuracy: 0.9595
Epoch 48/65535
420/420 [==============================] - 228s 542ms/step - loss: 0.0033 - categorical_accuracy: 0.9826 - val_loss: 0.0082 - val_categorical_accuracy: 0.9593
Epoch 49/65535
420/420 [==============================] - 216s 514ms/step - loss: 0.0033 - categorical_accuracy: 0.9826 - val_loss: 0.0081 - val_categorical_accuracy: 0.9593
Epoch 00049: early stopping
========= generating oof predictions 08:55:37 =========
========= generating test set predictions 08:55:47 =========
train loss avg 0.003578374533224113 -- std 0.00036038304379466774, val loss avg 0.007786433164721653 -- std 0.0003333236299701794
train acc avg 0.9814885734665865 -- std 0.0017661271804385761, val acc avg 0.9595656369226866 -- std 0.0009198944809564199
mean nb epochs 61.8
dump oof predicted probs
dump test set predicted probs
