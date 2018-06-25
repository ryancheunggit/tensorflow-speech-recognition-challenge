ren (master *+) python $ python train.py logmelspectrogram_40_25_10 gg4 2 128
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
conv2d_1 (Conv2D)               (None, 101, 40, 16)  176         reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 101, 40, 16)  64          conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 101, 40, 16)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 101, 40, 16)  2576        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 101, 40, 16)  64          conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 101, 40, 16)  0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 50, 20, 16)   0           activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 50, 20, 32)   5152        max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 50, 20, 32)   128         conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 50, 20, 32)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 50, 20, 32)   10272       activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 50, 20, 32)   128         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 50, 20, 32)   0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 25, 10, 32)   0           activation_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 25, 10, 64)   20544       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 25, 10, 64)   256         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 25, 10, 64)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 25, 10, 64)   41024       activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 25, 10, 64)   256         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 25, 10, 64)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 12, 5, 64)    0           activation_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 12, 5, 128)   82048       max_pooling2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 12, 5, 128)   512         conv2d_7[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 12, 5, 128)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 12, 5, 128)   163968      activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 12, 5, 128)   512         conv2d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 12, 5, 128)   0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 6, 2, 128)    0           activation_8[0][0]
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
Total params: 469,279
Trainable params: 467,295
Non-trainable params: 1,984
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 22:18:39 =========
Epoch 1/65535
420/420 [==============================] - 17s 41ms/step - loss: 0.0746 - categorical_accuracy: 0.5606 - val_loss: 0.0239 - val_categorical_accuracy: 0.8639
Epoch 2/65535
420/420 [==============================] - 16s 37ms/step - loss: 0.0233 - categorical_accuracy: 0.8724 - val_loss: 0.0132 - val_categorical_accuracy: 0.9283
Epoch 3/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0156 - categorical_accuracy: 0.9178 - val_loss: 0.0142 - val_categorical_accuracy: 0.9229
Epoch 4/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0122 - categorical_accuracy: 0.9354 - val_loss: 0.0096 - val_categorical_accuracy: 0.9481
Epoch 5/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0103 - categorical_accuracy: 0.9460 - val_loss: 0.0094 - val_categorical_accuracy: 0.9477
Epoch 6/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0089 - categorical_accuracy: 0.9532 - val_loss: 0.0084 - val_categorical_accuracy: 0.9527
Epoch 7/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0078 - categorical_accuracy: 0.9595 - val_loss: 0.0087 - val_categorical_accuracy: 0.9522
Epoch 8/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0071 - categorical_accuracy: 0.9632 - val_loss: 0.0082 - val_categorical_accuracy: 0.9557
Epoch 9/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0063 - categorical_accuracy: 0.9677 - val_loss: 0.0072 - val_categorical_accuracy: 0.9603
Epoch 10/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0058 - categorical_accuracy: 0.9700 - val_loss: 0.0102 - val_categorical_accuracy: 0.9480
Epoch 11/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0052 - categorical_accuracy: 0.9730 - val_loss: 0.0079 - val_categorical_accuracy: 0.9580
Epoch 12/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0047 - categorical_accuracy: 0.9751 - val_loss: 0.0080 - val_categorical_accuracy: 0.9595
Epoch 13/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0044 - categorical_accuracy: 0.9776 - val_loss: 0.0072 - val_categorical_accuracy: 0.9620
Epoch 14/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0040 - categorical_accuracy: 0.9798 - val_loss: 0.0072 - val_categorical_accuracy: 0.9629
Epoch 15/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0036 - categorical_accuracy: 0.9813 - val_loss: 0.0073 - val_categorical_accuracy: 0.9609

Epoch 00015: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 16/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0027 - categorical_accuracy: 0.9868 - val_loss: 0.0061 - val_categorical_accuracy: 0.9694
Epoch 17/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0023 - categorical_accuracy: 0.9891 - val_loss: 0.0060 - val_categorical_accuracy: 0.9696
Epoch 18/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0021 - categorical_accuracy: 0.9894 - val_loss: 0.0061 - val_categorical_accuracy: 0.9705
Epoch 19/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0020 - categorical_accuracy: 0.9910 - val_loss: 0.0061 - val_categorical_accuracy: 0.9702
Epoch 20/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0018 - categorical_accuracy: 0.9917 - val_loss: 0.0061 - val_categorical_accuracy: 0.9705
Epoch 21/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9921 - val_loss: 0.0062 - val_categorical_accuracy: 0.9697
Epoch 22/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9926 - val_loss: 0.0063 - val_categorical_accuracy: 0.9690

Epoch 00022: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 23/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0015 - categorical_accuracy: 0.9935 - val_loss: 0.0062 - val_categorical_accuracy: 0.9704
Epoch 24/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0015 - categorical_accuracy: 0.9934 - val_loss: 0.0061 - val_categorical_accuracy: 0.9710
Epoch 25/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0014 - categorical_accuracy: 0.9937 - val_loss: 0.0062 - val_categorical_accuracy: 0.9709
Epoch 26/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0014 - categorical_accuracy: 0.9938 - val_loss: 0.0062 - val_categorical_accuracy: 0.9706
Epoch 27/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0014 - categorical_accuracy: 0.9940 - val_loss: 0.0061 - val_categorical_accuracy: 0.9711

Epoch 00027: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 28/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0014 - categorical_accuracy: 0.9939 - val_loss: 0.0061 - val_categorical_accuracy: 0.9711
Epoch 29/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0013 - categorical_accuracy: 0.9941 - val_loss: 0.0061 - val_categorical_accuracy: 0.9712
Epoch 30/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9945 - val_loss: 0.0061 - val_categorical_accuracy: 0.9714
Epoch 31/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0013 - categorical_accuracy: 0.9942 - val_loss: 0.0061 - val_categorical_accuracy: 0.9712
Epoch 32/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0013 - categorical_accuracy: 0.9943 - val_loss: 0.0061 - val_categorical_accuracy: 0.9713

Epoch 00032: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 00032: early stopping
========= generating oof predictions 22:27:22 =========
========= generating test set predictions 22:27:24 =========
========= fitting 2 th model 22:27:43 =========
Epoch 1/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0759 - categorical_accuracy: 0.5498 - val_loss: 0.0529 - val_categorical_accuracy: 0.6830
Epoch 2/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0243 - categorical_accuracy: 0.8667 - val_loss: 0.0203 - val_categorical_accuracy: 0.8880
Epoch 3/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0167 - categorical_accuracy: 0.9109 - val_loss: 0.0137 - val_categorical_accuracy: 0.9256
Epoch 4/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0129 - categorical_accuracy: 0.9316 - val_loss: 0.0108 - val_categorical_accuracy: 0.9397
Epoch 5/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0108 - categorical_accuracy: 0.9433 - val_loss: 0.0095 - val_categorical_accuracy: 0.9494
Epoch 6/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0092 - categorical_accuracy: 0.9524 - val_loss: 0.0087 - val_categorical_accuracy: 0.9532
Epoch 7/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0082 - categorical_accuracy: 0.9571 - val_loss: 0.0101 - val_categorical_accuracy: 0.9438
Epoch 8/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0075 - categorical_accuracy: 0.9612 - val_loss: 0.0081 - val_categorical_accuracy: 0.9550
Epoch 9/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0068 - categorical_accuracy: 0.9648 - val_loss: 0.0127 - val_categorical_accuracy: 0.9355
Epoch 10/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0061 - categorical_accuracy: 0.9676 - val_loss: 0.0085 - val_categorical_accuracy: 0.9527
Epoch 11/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0055 - categorical_accuracy: 0.9715 - val_loss: 0.0070 - val_categorical_accuracy: 0.9599
Epoch 12/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0052 - categorical_accuracy: 0.9724 - val_loss: 0.0094 - val_categorical_accuracy: 0.9497
Epoch 13/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0047 - categorical_accuracy: 0.9754 - val_loss: 0.0082 - val_categorical_accuracy: 0.9573
Epoch 14/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0045 - categorical_accuracy: 0.9766 - val_loss: 0.0075 - val_categorical_accuracy: 0.9608
Epoch 15/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0040 - categorical_accuracy: 0.9796 - val_loss: 0.0086 - val_categorical_accuracy: 0.9565
Epoch 16/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0037 - categorical_accuracy: 0.9802 - val_loss: 0.0079 - val_categorical_accuracy: 0.9607
Epoch 17/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0034 - categorical_accuracy: 0.9819 - val_loss: 0.0077 - val_categorical_accuracy: 0.9628

Epoch 00017: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 18/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0024 - categorical_accuracy: 0.9879 - val_loss: 0.0059 - val_categorical_accuracy: 0.9698
Epoch 19/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0021 - categorical_accuracy: 0.9895 - val_loss: 0.0062 - val_categorical_accuracy: 0.9687
Epoch 20/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0019 - categorical_accuracy: 0.9909 - val_loss: 0.0061 - val_categorical_accuracy: 0.9689
Epoch 21/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9920 - val_loss: 0.0062 - val_categorical_accuracy: 0.9697
Epoch 22/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0016 - categorical_accuracy: 0.9926 - val_loss: 0.0061 - val_categorical_accuracy: 0.9704
Epoch 23/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0015 - categorical_accuracy: 0.9931 - val_loss: 0.0063 - val_categorical_accuracy: 0.9700
Epoch 24/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0015 - categorical_accuracy: 0.9936 - val_loss: 0.0062 - val_categorical_accuracy: 0.9704

Epoch 00024: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 25/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0013 - categorical_accuracy: 0.9942 - val_loss: 0.0061 - val_categorical_accuracy: 0.9709
Epoch 26/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9938 - val_loss: 0.0061 - val_categorical_accuracy: 0.9710
Epoch 27/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9944 - val_loss: 0.0061 - val_categorical_accuracy: 0.9715
Epoch 28/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0012 - categorical_accuracy: 0.9948 - val_loss: 0.0061 - val_categorical_accuracy: 0.9709
Epoch 29/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0012 - categorical_accuracy: 0.9950 - val_loss: 0.0062 - val_categorical_accuracy: 0.9709

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 30/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0012 - categorical_accuracy: 0.9955 - val_loss: 0.0061 - val_categorical_accuracy: 0.9708
Epoch 31/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0011 - categorical_accuracy: 0.9953 - val_loss: 0.0062 - val_categorical_accuracy: 0.9708
Epoch 32/65535
420/420 [==============================] - 16s 38ms/step - loss: 0.0012 - categorical_accuracy: 0.9950 - val_loss: 0.0062 - val_categorical_accuracy: 0.9710
Epoch 33/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0012 - categorical_accuracy: 0.9954 - val_loss: 0.0061 - val_categorical_accuracy: 0.9710
Epoch 00033: early stopping
========= generating oof predictions 22:36:48 =========
========= generating test set predictions 22:36:49 =========
========= fitting 3 th model 22:37:09 =========
Epoch 1/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0932 - categorical_accuracy: 0.4437 - val_loss: 0.0335 - val_categorical_accuracy: 0.8118
Epoch 2/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0285 - categorical_accuracy: 0.8414 - val_loss: 0.0224 - val_categorical_accuracy: 0.8727
Epoch 3/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0185 - categorical_accuracy: 0.9001 - val_loss: 0.0176 - val_categorical_accuracy: 0.9068
Epoch 4/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0139 - categorical_accuracy: 0.9259 - val_loss: 0.0153 - val_categorical_accuracy: 0.9135
Epoch 5/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0115 - categorical_accuracy: 0.9399 - val_loss: 0.0107 - val_categorical_accuracy: 0.9425
Epoch 6/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0102 - categorical_accuracy: 0.9472 - val_loss: 0.0117 - val_categorical_accuracy: 0.9367
Epoch 7/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0090 - categorical_accuracy: 0.9530 - val_loss: 0.0143 - val_categorical_accuracy: 0.9322
Epoch 8/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0081 - categorical_accuracy: 0.9575 - val_loss: 0.0112 - val_categorical_accuracy: 0.9387
Epoch 9/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0072 - categorical_accuracy: 0.9624 - val_loss: 0.0170 - val_categorical_accuracy: 0.9263
Epoch 10/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0066 - categorical_accuracy: 0.9651 - val_loss: 0.0081 - val_categorical_accuracy: 0.9566
Epoch 11/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0060 - categorical_accuracy: 0.9677 - val_loss: 0.0108 - val_categorical_accuracy: 0.9410
Epoch 12/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0054 - categorical_accuracy: 0.9719 - val_loss: 0.0095 - val_categorical_accuracy: 0.9510
Epoch 13/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0053 - categorical_accuracy: 0.9721 - val_loss: 0.0107 - val_categorical_accuracy: 0.9471
Epoch 14/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0049 - categorical_accuracy: 0.9742 - val_loss: 0.0084 - val_categorical_accuracy: 0.9581
Epoch 15/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0045 - categorical_accuracy: 0.9764 - val_loss: 0.0091 - val_categorical_accuracy: 0.9560
Epoch 16/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0043 - categorical_accuracy: 0.9770 - val_loss: 0.0098 - val_categorical_accuracy: 0.9515

Epoch 00016: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 17/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0031 - categorical_accuracy: 0.9838 - val_loss: 0.0065 - val_categorical_accuracy: 0.9677
Epoch 18/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0026 - categorical_accuracy: 0.9871 - val_loss: 0.0068 - val_categorical_accuracy: 0.9670
Epoch 19/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0024 - categorical_accuracy: 0.9881 - val_loss: 0.0069 - val_categorical_accuracy: 0.9673
Epoch 20/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0022 - categorical_accuracy: 0.9895 - val_loss: 0.0070 - val_categorical_accuracy: 0.9671
Epoch 21/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0020 - categorical_accuracy: 0.9903 - val_loss: 0.0069 - val_categorical_accuracy: 0.9673
Epoch 22/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0020 - categorical_accuracy: 0.9910 - val_loss: 0.0069 - val_categorical_accuracy: 0.9678
Epoch 23/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0019 - categorical_accuracy: 0.9907 - val_loss: 0.0073 - val_categorical_accuracy: 0.9660

Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 24/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9922 - val_loss: 0.0067 - val_categorical_accuracy: 0.9691
Epoch 25/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9924 - val_loss: 0.0068 - val_categorical_accuracy: 0.9689
Epoch 26/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0016 - categorical_accuracy: 0.9928 - val_loss: 0.0068 - val_categorical_accuracy: 0.9695
Epoch 27/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0016 - categorical_accuracy: 0.9928 - val_loss: 0.0067 - val_categorical_accuracy: 0.9692
Epoch 28/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0016 - categorical_accuracy: 0.9928 - val_loss: 0.0067 - val_categorical_accuracy: 0.9689

Epoch 00028: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 29/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0016 - categorical_accuracy: 0.9927 - val_loss: 0.0067 - val_categorical_accuracy: 0.9691
Epoch 30/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0015 - categorical_accuracy: 0.9936 - val_loss: 0.0067 - val_categorical_accuracy: 0.9692
Epoch 31/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0015 - categorical_accuracy: 0.9935 - val_loss: 0.0067 - val_categorical_accuracy: 0.9691
Epoch 32/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0015 - categorical_accuracy: 0.9932 - val_loss: 0.0067 - val_categorical_accuracy: 0.9692
Epoch 00032: early stopping
========= generating oof predictions 22:46:00 =========
========= generating test set predictions 22:46:01 =========
========= fitting 4 th model 22:46:21 =========
Epoch 1/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0738 - categorical_accuracy: 0.5662 - val_loss: 0.0313 - val_categorical_accuracy: 0.8241
Epoch 2/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0230 - categorical_accuracy: 0.8746 - val_loss: 0.0177 - val_categorical_accuracy: 0.9032
Epoch 3/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0155 - categorical_accuracy: 0.9179 - val_loss: 0.0122 - val_categorical_accuracy: 0.9359
Epoch 4/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0124 - categorical_accuracy: 0.9350 - val_loss: 0.0123 - val_categorical_accuracy: 0.9340
Epoch 5/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0104 - categorical_accuracy: 0.9455 - val_loss: 0.0118 - val_categorical_accuracy: 0.9352
Epoch 6/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0089 - categorical_accuracy: 0.9527 - val_loss: 0.0121 - val_categorical_accuracy: 0.9346
Epoch 7/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0079 - categorical_accuracy: 0.9585 - val_loss: 0.0119 - val_categorical_accuracy: 0.9454
Epoch 8/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0070 - categorical_accuracy: 0.9630 - val_loss: 0.0081 - val_categorical_accuracy: 0.9575
Epoch 9/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0064 - categorical_accuracy: 0.9662 - val_loss: 0.0078 - val_categorical_accuracy: 0.9568
Epoch 10/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0059 - categorical_accuracy: 0.9685 - val_loss: 0.0106 - val_categorical_accuracy: 0.9495
Epoch 11/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0053 - categorical_accuracy: 0.9721 - val_loss: 0.0089 - val_categorical_accuracy: 0.9520
Epoch 12/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0048 - categorical_accuracy: 0.9754 - val_loss: 0.0076 - val_categorical_accuracy: 0.9577
Epoch 13/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0045 - categorical_accuracy: 0.9764 - val_loss: 0.0120 - val_categorical_accuracy: 0.9374
Epoch 14/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0041 - categorical_accuracy: 0.9788 - val_loss: 0.0073 - val_categorical_accuracy: 0.9592
Epoch 15/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0038 - categorical_accuracy: 0.9801 - val_loss: 0.0069 - val_categorical_accuracy: 0.9631
Epoch 16/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0035 - categorical_accuracy: 0.9812 - val_loss: 0.0077 - val_categorical_accuracy: 0.9610
Epoch 17/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0032 - categorical_accuracy: 0.9833 - val_loss: 0.0100 - val_categorical_accuracy: 0.9474
Epoch 18/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0030 - categorical_accuracy: 0.9842 - val_loss: 0.0079 - val_categorical_accuracy: 0.9617
Epoch 19/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0029 - categorical_accuracy: 0.9852 - val_loss: 0.0079 - val_categorical_accuracy: 0.9613
Epoch 20/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0025 - categorical_accuracy: 0.9870 - val_loss: 0.0078 - val_categorical_accuracy: 0.9614
Epoch 21/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0026 - categorical_accuracy: 0.9866 - val_loss: 0.0071 - val_categorical_accuracy: 0.9655

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 22/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0016 - categorical_accuracy: 0.9919 - val_loss: 0.0059 - val_categorical_accuracy: 0.9703
Epoch 23/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9945 - val_loss: 0.0060 - val_categorical_accuracy: 0.9709
Epoch 24/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0011 - categorical_accuracy: 0.9952 - val_loss: 0.0059 - val_categorical_accuracy: 0.9715
Epoch 25/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0011 - categorical_accuracy: 0.9955 - val_loss: 0.0059 - val_categorical_accuracy: 0.9711
Epoch 26/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0010 - categorical_accuracy: 0.9956 - val_loss: 0.0059 - val_categorical_accuracy: 0.9716
Epoch 27/65535
420/420 [==============================] - 16s 38ms/step - loss: 9.5980e-04 - categorical_accuracy: 0.9965 - val_loss: 0.0061 - val_categorical_accuracy: 0.9712
Epoch 28/65535
420/420 [==============================] - 17s 39ms/step - loss: 8.8161e-04 - categorical_accuracy: 0.9966 - val_loss: 0.0060 - val_categorical_accuracy: 0.9715

Epoch 00028: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 29/65535
420/420 [==============================] - 16s 39ms/step - loss: 8.0576e-04 - categorical_accuracy: 0.9972 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 30/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.2939e-04 - categorical_accuracy: 0.9975 - val_loss: 0.0058 - val_categorical_accuracy: 0.9724
Epoch 31/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.3941e-04 - categorical_accuracy: 0.9974 - val_loss: 0.0058 - val_categorical_accuracy: 0.9721
Epoch 32/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.4332e-04 - categorical_accuracy: 0.9972 - val_loss: 0.0058 - val_categorical_accuracy: 0.9722
Epoch 33/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.6786e-04 - categorical_accuracy: 0.9972 - val_loss: 0.0058 - val_categorical_accuracy: 0.9723
Epoch 34/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.9987e-04 - categorical_accuracy: 0.9975 - val_loss: 0.0058 - val_categorical_accuracy: 0.9723
Epoch 35/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.0380e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0058 - val_categorical_accuracy: 0.9728

Epoch 00035: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 36/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.9856e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0058 - val_categorical_accuracy: 0.9727
Epoch 37/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.1060e-04 - categorical_accuracy: 0.9975 - val_loss: 0.0058 - val_categorical_accuracy: 0.9729
Epoch 38/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.1093e-04 - categorical_accuracy: 0.9973 - val_loss: 0.0058 - val_categorical_accuracy: 0.9730
Epoch 39/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.7949e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0058 - val_categorical_accuracy: 0.9727
Epoch 40/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.9200e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0057 - val_categorical_accuracy: 0.9728

Epoch 00040: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 41/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.7810e-04 - categorical_accuracy: 0.9978 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 42/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.1604e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 43/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.7846e-04 - categorical_accuracy: 0.9978 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 44/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.0572e-04 - categorical_accuracy: 0.9975 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 45/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.7210e-04 - categorical_accuracy: 0.9979 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726

Epoch 00045: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 46/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.7011e-04 - categorical_accuracy: 0.9978 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 47/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.0491e-04 - categorical_accuracy: 0.9974 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 48/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.5836e-04 - categorical_accuracy: 0.9978 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 49/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.0472e-04 - categorical_accuracy: 0.9975 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 50/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.8486e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726

Epoch 00050: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 51/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.8128e-04 - categorical_accuracy: 0.9977 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 52/65535
420/420 [==============================] - 16s 39ms/step - loss: 7.0709e-04 - categorical_accuracy: 0.9974 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 53/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.7777e-04 - categorical_accuracy: 0.9976 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 54/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.6392e-04 - categorical_accuracy: 0.9978 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726
Epoch 55/65535
420/420 [==============================] - 16s 39ms/step - loss: 6.5273e-04 - categorical_accuracy: 0.9979 - val_loss: 0.0058 - val_categorical_accuracy: 0.9726

Epoch 00055: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 00055: early stopping
========= generating oof predictions 23:01:26 =========
========= generating test set predictions 23:01:27 =========
========= fitting 5 th model 23:01:47 =========
Epoch 1/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0716 - categorical_accuracy: 0.5750 - val_loss: 0.0267 - val_categorical_accuracy: 0.8487
Epoch 2/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0233 - categorical_accuracy: 0.8715 - val_loss: 0.0194 - val_categorical_accuracy: 0.8911
Epoch 3/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0164 - categorical_accuracy: 0.9146 - val_loss: 0.0148 - val_categorical_accuracy: 0.9187
Epoch 4/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0127 - categorical_accuracy: 0.9347 - val_loss: 0.0110 - val_categorical_accuracy: 0.9392
Epoch 5/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0104 - categorical_accuracy: 0.9450 - val_loss: 0.0114 - val_categorical_accuracy: 0.9380
Epoch 6/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0092 - categorical_accuracy: 0.9524 - val_loss: 0.0104 - val_categorical_accuracy: 0.9450
Epoch 7/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0082 - categorical_accuracy: 0.9567 - val_loss: 0.0093 - val_categorical_accuracy: 0.9509
Epoch 8/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0072 - categorical_accuracy: 0.9628 - val_loss: 0.0097 - val_categorical_accuracy: 0.9490
Epoch 9/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0065 - categorical_accuracy: 0.9665 - val_loss: 0.0098 - val_categorical_accuracy: 0.9484
Epoch 10/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0059 - categorical_accuracy: 0.9693 - val_loss: 0.0084 - val_categorical_accuracy: 0.9550
Epoch 11/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0055 - categorical_accuracy: 0.9713 - val_loss: 0.0087 - val_categorical_accuracy: 0.9562
Epoch 12/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0049 - categorical_accuracy: 0.9742 - val_loss: 0.0097 - val_categorical_accuracy: 0.9530
Epoch 13/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0045 - categorical_accuracy: 0.9763 - val_loss: 0.0101 - val_categorical_accuracy: 0.9483
Epoch 14/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0043 - categorical_accuracy: 0.9781 - val_loss: 0.0094 - val_categorical_accuracy: 0.9525
Epoch 15/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0039 - categorical_accuracy: 0.9794 - val_loss: 0.0090 - val_categorical_accuracy: 0.9538
Epoch 16/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0036 - categorical_accuracy: 0.9812 - val_loss: 0.0093 - val_categorical_accuracy: 0.9542

Epoch 00016: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 17/65535
420/420 [==============================] - 17s 40ms/step - loss: 0.0026 - categorical_accuracy: 0.9873 - val_loss: 0.0067 - val_categorical_accuracy: 0.9667
Epoch 18/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0022 - categorical_accuracy: 0.9890 - val_loss: 0.0067 - val_categorical_accuracy: 0.9682
Epoch 19/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0021 - categorical_accuracy: 0.9901 - val_loss: 0.0068 - val_categorical_accuracy: 0.9679
Epoch 20/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0019 - categorical_accuracy: 0.9911 - val_loss: 0.0068 - val_categorical_accuracy: 0.9677
Epoch 21/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0018 - categorical_accuracy: 0.9910 - val_loss: 0.0069 - val_categorical_accuracy: 0.9666
Epoch 22/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0017 - categorical_accuracy: 0.9920 - val_loss: 0.0068 - val_categorical_accuracy: 0.9671
Epoch 23/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0016 - categorical_accuracy: 0.9926 - val_loss: 0.0070 - val_categorical_accuracy: 0.9676

Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 24/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0015 - categorical_accuracy: 0.9938 - val_loss: 0.0068 - val_categorical_accuracy: 0.9682
Epoch 25/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0014 - categorical_accuracy: 0.9936 - val_loss: 0.0068 - val_categorical_accuracy: 0.9686
Epoch 26/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0014 - categorical_accuracy: 0.9942 - val_loss: 0.0068 - val_categorical_accuracy: 0.9676
Epoch 27/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0014 - categorical_accuracy: 0.9941 - val_loss: 0.0068 - val_categorical_accuracy: 0.9685
Epoch 28/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9945 - val_loss: 0.0068 - val_categorical_accuracy: 0.9680

Epoch 00028: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 29/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9948 - val_loss: 0.0067 - val_categorical_accuracy: 0.9683
Epoch 30/65535
420/420 [==============================] - 16s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9946 - val_loss: 0.0067 - val_categorical_accuracy: 0.9683
Epoch 31/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9948 - val_loss: 0.0067 - val_categorical_accuracy: 0.9681
Epoch 32/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9945 - val_loss: 0.0067 - val_categorical_accuracy: 0.9680
Epoch 33/65535
420/420 [==============================] - 17s 39ms/step - loss: 0.0013 - categorical_accuracy: 0.9942 - val_loss: 0.0067 - val_categorical_accuracy: 0.9679

Epoch 00033: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 00033: early stopping
========= generating oof predictions 23:10:54 =========
========= generating test set predictions 23:10:55 =========
train loss avg 0.0011937239530125736 -- std 0.00028969108747643047, val loss avg 0.0062955202441833755 -- std 0.00037798998693276936
train acc avg 0.9950126877426422 -- std 0.001609162342941081, val acc avg 0.9704111837845393 -- std 0.0016735668862409645
mean nb epochs 37.0
dump oof predicted probs
dump test set predicted probs
