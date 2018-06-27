ren (master *+) python $ python train.py rawwav 1dcnn l 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (16000, 1) ===========
------------- SUMMARY OF MODEL -------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 16000, 1)          0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 16000, 8)          32
_________________________________________________________________
batch_normalization_1 (Batch (None, 16000, 8)          32
_________________________________________________________________
activation_1 (Activation)    (None, 16000, 8)          0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 16000, 8)          200
_________________________________________________________________
batch_normalization_2 (Batch (None, 16000, 8)          32
_________________________________________________________________
activation_2 (Activation)    (None, 16000, 8)          0
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 8000, 8)           0
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 8000, 16)          400
_________________________________________________________________
batch_normalization_3 (Batch (None, 8000, 16)          64
_________________________________________________________________
activation_3 (Activation)    (None, 8000, 16)          0
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 8000, 16)          784
_________________________________________________________________
batch_normalization_4 (Batch (None, 8000, 16)          64
_________________________________________________________________
activation_4 (Activation)    (None, 8000, 16)          0
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 4000, 16)          0
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 4000, 32)          1568
_________________________________________________________________
batch_normalization_5 (Batch (None, 4000, 32)          128
_________________________________________________________________
activation_5 (Activation)    (None, 4000, 32)          0
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 4000, 32)          3104
_________________________________________________________________
batch_normalization_6 (Batch (None, 4000, 32)          128
_________________________________________________________________
activation_6 (Activation)    (None, 4000, 32)          0
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 2000, 32)          0
_________________________________________________________________
dropout_1 (Dropout)          (None, 2000, 32)          0
_________________________________________________________________
conv1d_7 (Conv1D)            (None, 2000, 64)          6208
_________________________________________________________________
batch_normalization_7 (Batch (None, 2000, 64)          256
_________________________________________________________________
activation_7 (Activation)    (None, 2000, 64)          0
_________________________________________________________________
conv1d_8 (Conv1D)            (None, 2000, 64)          12352
_________________________________________________________________
batch_normalization_8 (Batch (None, 2000, 64)          256
_________________________________________________________________
activation_8 (Activation)    (None, 2000, 64)          0
_________________________________________________________________
max_pooling1d_4 (MaxPooling1 (None, 1000, 64)          0
_________________________________________________________________
dropout_2 (Dropout)          (None, 1000, 64)          0
_________________________________________________________________
conv1d_9 (Conv1D)            (None, 1000, 128)         24704
_________________________________________________________________
batch_normalization_9 (Batch (None, 1000, 128)         512
_________________________________________________________________
activation_9 (Activation)    (None, 1000, 128)         0
_________________________________________________________________
conv1d_10 (Conv1D)           (None, 1000, 128)         49280
_________________________________________________________________
batch_normalization_10 (Batc (None, 1000, 128)         512
_________________________________________________________________
activation_10 (Activation)   (None, 1000, 128)         0
_________________________________________________________________
max_pooling1d_5 (MaxPooling1 (None, 500, 128)          0
_________________________________________________________________
dropout_3 (Dropout)          (None, 500, 128)          0
_________________________________________________________________
conv1d_11 (Conv1D)           (None, 500, 256)          98560
_________________________________________________________________
batch_normalization_11 (Batc (None, 500, 256)          1024
_________________________________________________________________
activation_11 (Activation)   (None, 500, 256)          0
_________________________________________________________________
conv1d_12 (Conv1D)           (None, 500, 256)          196864
_________________________________________________________________
batch_normalization_12 (Batc (None, 500, 256)          1024
_________________________________________________________________
activation_12 (Activation)   (None, 500, 256)          0
_________________________________________________________________
conv1d_13 (Conv1D)           (None, 500, 256)          196864
_________________________________________________________________
batch_normalization_13 (Batc (None, 500, 256)          1024
_________________________________________________________________
activation_13 (Activation)   (None, 500, 256)          0
_________________________________________________________________
conv1d_14 (Conv1D)           (None, 500, 256)          196864
_________________________________________________________________
batch_normalization_14 (Batc (None, 500, 256)          1024
_________________________________________________________________
activation_14 (Activation)   (None, 500, 256)          0
_________________________________________________________________
max_pooling1d_6 (MaxPooling1 (None, 250, 256)          0
_________________________________________________________________
dropout_4 (Dropout)          (None, 250, 256)          0
_________________________________________________________________
conv1d_15 (Conv1D)           (None, 250, 512)          393728
_________________________________________________________________
batch_normalization_15 (Batc (None, 250, 512)          2048
_________________________________________________________________
activation_15 (Activation)   (None, 250, 512)          0
_________________________________________________________________
conv1d_16 (Conv1D)           (None, 250, 512)          786944
_________________________________________________________________
batch_normalization_16 (Batc (None, 250, 512)          2048
_________________________________________________________________
activation_16 (Activation)   (None, 250, 512)          0
_________________________________________________________________
conv1d_17 (Conv1D)           (None, 250, 512)          786944
_________________________________________________________________
batch_normalization_17 (Batc (None, 250, 512)          2048
_________________________________________________________________
activation_17 (Activation)   (None, 250, 512)          0
_________________________________________________________________
conv1d_18 (Conv1D)           (None, 250, 512)          786944
_________________________________________________________________
batch_normalization_18 (Batc (None, 250, 512)          2048
_________________________________________________________________
activation_18 (Activation)   (None, 250, 512)          0
_________________________________________________________________
max_pooling1d_7 (MaxPooling1 (None, 125, 512)          0
_________________________________________________________________
dropout_5 (Dropout)          (None, 125, 512)          0
_________________________________________________________________
conv1d_19 (Conv1D)           (None, 125, 1024)         1573888
_________________________________________________________________
batch_normalization_19 (Batc (None, 125, 1024)         4096
_________________________________________________________________
activation_19 (Activation)   (None, 125, 1024)         0
_________________________________________________________________
conv1d_20 (Conv1D)           (None, 125, 1024)         3146752
_________________________________________________________________
batch_normalization_20 (Batc (None, 125, 1024)         4096
_________________________________________________________________
activation_20 (Activation)   (None, 125, 1024)         0
_________________________________________________________________
max_pooling1d_8 (MaxPooling1 (None, 62, 1024)          0
_________________________________________________________________
dropout_6 (Dropout)          (None, 62, 1024)          0
_________________________________________________________________
global_average_pooling1d_1 ( (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               524800
_________________________________________________________________
batch_normalization_21 (Batc (None, 512)               2048
_________________________________________________________________
dropout_7 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328
_________________________________________________________________
batch_normalization_22 (Batc (None, 256)               1024
_________________________________________________________________
dropout_8 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_3 (Dense)              (None, 31)                7967
=================================================================
Total params: 8,952,615
Trainable params: 8,939,847
Non-trainable params: 12,768
_________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 21:51:07 =========
Epoch 1/65535
420/420 [==============================] - 188s 448ms/step - loss: 0.1075 - categorical_accuracy: 0.3180 - val_loss: 0.0881 - val_categorical_accuracy: 0.4866
Epoch 2/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0518 - categorical_accuracy: 0.6924 - val_loss: 0.0487 - val_categorical_accuracy: 0.7201
Epoch 3/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0350 - categorical_accuracy: 0.7975 - val_loss: 0.0444 - val_categorical_accuracy: 0.7552
Epoch 4/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0277 - categorical_accuracy: 0.8426 - val_loss: 0.0333 - val_categorical_accuracy: 0.8122
Epoch 5/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0235 - categorical_accuracy: 0.8678 - val_loss: 0.0308 - val_categorical_accuracy: 0.8291
Epoch 6/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0207 - categorical_accuracy: 0.8839 - val_loss: 0.0210 - val_categorical_accuracy: 0.8827
Epoch 7/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0188 - categorical_accuracy: 0.8948 - val_loss: 0.0221 - val_categorical_accuracy: 0.8775
Epoch 8/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0169 - categorical_accuracy: 0.9063 - val_loss: 0.0177 - val_categorical_accuracy: 0.9016
Epoch 9/65535
420/420 [==============================] - 185s 439ms/step - loss: 0.0153 - categorical_accuracy: 0.9144 - val_loss: 0.0163 - val_categorical_accuracy: 0.9102
Epoch 10/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0145 - categorical_accuracy: 0.9204 - val_loss: 0.0172 - val_categorical_accuracy: 0.9055
Epoch 11/65535
420/420 [==============================] - 181s 430ms/step - loss: 0.0137 - categorical_accuracy: 0.9230 - val_loss: 0.0170 - val_categorical_accuracy: 0.9101
Epoch 12/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0131 - categorical_accuracy: 0.9271 - val_loss: 0.0187 - val_categorical_accuracy: 0.8974
Epoch 13/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0124 - categorical_accuracy: 0.9300 - val_loss: 0.0201 - val_categorical_accuracy: 0.8966
Epoch 14/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0115 - categorical_accuracy: 0.9354 - val_loss: 0.0154 - val_categorical_accuracy: 0.9146
Epoch 15/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0111 - categorical_accuracy: 0.9384 - val_loss: 0.0164 - val_categorical_accuracy: 0.9131
Epoch 16/65535
420/420 [==============================] - 181s 431ms/step - loss: 0.0105 - categorical_accuracy: 0.9418 - val_loss: 0.0182 - val_categorical_accuracy: 0.9072
Epoch 17/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0100 - categorical_accuracy: 0.9443 - val_loss: 0.0133 - val_categorical_accuracy: 0.9262
Epoch 18/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0096 - categorical_accuracy: 0.9460 - val_loss: 0.0156 - val_categorical_accuracy: 0.9185
Epoch 19/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0091 - categorical_accuracy: 0.9495 - val_loss: 0.0180 - val_categorical_accuracy: 0.9145
Epoch 20/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0088 - categorical_accuracy: 0.9512 - val_loss: 0.0154 - val_categorical_accuracy: 0.9232
Epoch 21/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0085 - categorical_accuracy: 0.9527 - val_loss: 0.0148 - val_categorical_accuracy: 0.9210
Epoch 22/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0082 - categorical_accuracy: 0.9540 - val_loss: 0.0156 - val_categorical_accuracy: 0.9254
Epoch 23/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0079 - categorical_accuracy: 0.9557 - val_loss: 0.0158 - val_categorical_accuracy: 0.9230

Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 24/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0060 - categorical_accuracy: 0.9662 - val_loss: 0.0086 - val_categorical_accuracy: 0.9536
Epoch 25/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0055 - categorical_accuracy: 0.9705 - val_loss: 0.0083 - val_categorical_accuracy: 0.9558
Epoch 26/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0052 - categorical_accuracy: 0.9718 - val_loss: 0.0082 - val_categorical_accuracy: 0.9552
Epoch 27/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0051 - categorical_accuracy: 0.9733 - val_loss: 0.0082 - val_categorical_accuracy: 0.9559
Epoch 28/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0049 - categorical_accuracy: 0.9738 - val_loss: 0.0082 - val_categorical_accuracy: 0.9560
Epoch 29/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0047 - categorical_accuracy: 0.9750 - val_loss: 0.0081 - val_categorical_accuracy: 0.9550
Epoch 30/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0047 - categorical_accuracy: 0.9742 - val_loss: 0.0083 - val_categorical_accuracy: 0.9569
Epoch 31/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0046 - categorical_accuracy: 0.9750 - val_loss: 0.0080 - val_categorical_accuracy: 0.9565
Epoch 32/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0045 - categorical_accuracy: 0.9760 - val_loss: 0.0083 - val_categorical_accuracy: 0.9558
Epoch 33/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0044 - categorical_accuracy: 0.9762 - val_loss: 0.0080 - val_categorical_accuracy: 0.9570
Epoch 34/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0044 - categorical_accuracy: 0.9766 - val_loss: 0.0091 - val_categorical_accuracy: 0.9478
Epoch 35/65535
420/420 [==============================] - 184s 439ms/step - loss: 0.0042 - categorical_accuracy: 0.9770 - val_loss: 0.0081 - val_categorical_accuracy: 0.9568

Epoch 00035: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 36/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0041 - categorical_accuracy: 0.9788 - val_loss: 0.0081 - val_categorical_accuracy: 0.9587
Epoch 37/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0040 - categorical_accuracy: 0.9788 - val_loss: 0.0081 - val_categorical_accuracy: 0.9585
Epoch 38/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0038 - categorical_accuracy: 0.9797 - val_loss: 0.0080 - val_categorical_accuracy: 0.9588
Epoch 39/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0039 - categorical_accuracy: 0.9797 - val_loss: 0.0084 - val_categorical_accuracy: 0.9585
Epoch 40/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0038 - categorical_accuracy: 0.9806 - val_loss: 0.0082 - val_categorical_accuracy: 0.9591

Epoch 00040: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 41/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0038 - categorical_accuracy: 0.9801 - val_loss: 0.0087 - val_categorical_accuracy: 0.9585
Epoch 42/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0038 - categorical_accuracy: 0.9798 - val_loss: 0.0088 - val_categorical_accuracy: 0.9583
Epoch 43/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0038 - categorical_accuracy: 0.9794 - val_loss: 0.0087 - val_categorical_accuracy: 0.9589
Epoch 44/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0037 - categorical_accuracy: 0.9801 - val_loss: 0.0087 - val_categorical_accuracy: 0.9588
Epoch 45/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0037 - categorical_accuracy: 0.9802 - val_loss: 0.0088 - val_categorical_accuracy: 0.9591

Epoch 00045: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 46/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0038 - categorical_accuracy: 0.9796 - val_loss: 0.0088 - val_categorical_accuracy: 0.9589
Epoch 00046: early stopping
========= generating oof predictions 00:11:31 =========
========= generating test set predictions 00:11:44 =========
========= fitting 2 th model 00:14:25 =========
Epoch 1/65535
420/420 [==============================] - 184s 439ms/step - loss: 0.1099 - categorical_accuracy: 0.3010 - val_loss: 0.0939 - val_categorical_accuracy: 0.4089
Epoch 2/65535
420/420 [==============================] - 182s 432ms/step - loss: 0.0561 - categorical_accuracy: 0.6649 - val_loss: 0.0588 - val_categorical_accuracy: 0.6686
Epoch 3/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0368 - categorical_accuracy: 0.7883 - val_loss: 0.0483 - val_categorical_accuracy: 0.7379
Epoch 4/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0288 - categorical_accuracy: 0.8378 - val_loss: 0.0429 - val_categorical_accuracy: 0.7688
Epoch 5/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0242 - categorical_accuracy: 0.8646 - val_loss: 0.0307 - val_categorical_accuracy: 0.8304
Epoch 6/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0211 - categorical_accuracy: 0.8809 - val_loss: 0.0271 - val_categorical_accuracy: 0.8465
Epoch 7/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0190 - categorical_accuracy: 0.8953 - val_loss: 0.0176 - val_categorical_accuracy: 0.9040
Epoch 8/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0175 - categorical_accuracy: 0.9025 - val_loss: 0.0200 - val_categorical_accuracy: 0.8913
Epoch 9/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0160 - categorical_accuracy: 0.9114 - val_loss: 0.0167 - val_categorical_accuracy: 0.9011
Epoch 10/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0145 - categorical_accuracy: 0.9201 - val_loss: 0.0198 - val_categorical_accuracy: 0.8931
Epoch 11/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0137 - categorical_accuracy: 0.9254 - val_loss: 0.0163 - val_categorical_accuracy: 0.9052
Epoch 12/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0130 - categorical_accuracy: 0.9285 - val_loss: 0.0181 - val_categorical_accuracy: 0.8993
Epoch 13/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0122 - categorical_accuracy: 0.9330 - val_loss: 0.0136 - val_categorical_accuracy: 0.9239
Epoch 14/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0117 - categorical_accuracy: 0.9357 - val_loss: 0.0210 - val_categorical_accuracy: 0.8880
Epoch 15/65535
420/420 [==============================] - 182s 432ms/step - loss: 0.0109 - categorical_accuracy: 0.9406 - val_loss: 0.0154 - val_categorical_accuracy: 0.9151
Epoch 16/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0104 - categorical_accuracy: 0.9427 - val_loss: 0.0180 - val_categorical_accuracy: 0.8994
Epoch 17/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0101 - categorical_accuracy: 0.9442 - val_loss: 0.0163 - val_categorical_accuracy: 0.9133
Epoch 18/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0098 - categorical_accuracy: 0.9462 - val_loss: 0.0161 - val_categorical_accuracy: 0.9146
Epoch 19/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0094 - categorical_accuracy: 0.9468 - val_loss: 0.0138 - val_categorical_accuracy: 0.9277

Epoch 00019: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 20/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0070 - categorical_accuracy: 0.9613 - val_loss: 0.0094 - val_categorical_accuracy: 0.9482
Epoch 21/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0062 - categorical_accuracy: 0.9666 - val_loss: 0.0090 - val_categorical_accuracy: 0.9501
Epoch 22/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0060 - categorical_accuracy: 0.9681 - val_loss: 0.0090 - val_categorical_accuracy: 0.9519
Epoch 23/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0059 - categorical_accuracy: 0.9688 - val_loss: 0.0089 - val_categorical_accuracy: 0.9512
Epoch 24/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0057 - categorical_accuracy: 0.9705 - val_loss: 0.0090 - val_categorical_accuracy: 0.9500
Epoch 25/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0055 - categorical_accuracy: 0.9702 - val_loss: 0.0087 - val_categorical_accuracy: 0.9516
Epoch 26/65535
420/420 [==============================] - 184s 439ms/step - loss: 0.0055 - categorical_accuracy: 0.9707 - val_loss: 0.0087 - val_categorical_accuracy: 0.9518
Epoch 27/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0053 - categorical_accuracy: 0.9714 - val_loss: 0.0090 - val_categorical_accuracy: 0.9506
Epoch 28/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0052 - categorical_accuracy: 0.9721 - val_loss: 0.0091 - val_categorical_accuracy: 0.9508
Epoch 29/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0051 - categorical_accuracy: 0.9728 - val_loss: 0.0088 - val_categorical_accuracy: 0.9510
Epoch 30/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0050 - categorical_accuracy: 0.9727 - val_loss: 0.0091 - val_categorical_accuracy: 0.9508
Epoch 31/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0050 - categorical_accuracy: 0.9732 - val_loss: 0.0094 - val_categorical_accuracy: 0.9515

Epoch 00031: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 32/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0047 - categorical_accuracy: 0.9752 - val_loss: 0.0084 - val_categorical_accuracy: 0.9539
Epoch 33/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0045 - categorical_accuracy: 0.9760 - val_loss: 0.0083 - val_categorical_accuracy: 0.9549
Epoch 34/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0045 - categorical_accuracy: 0.9760 - val_loss: 0.0082 - val_categorical_accuracy: 0.9551
Epoch 35/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0044 - categorical_accuracy: 0.9770 - val_loss: 0.0083 - val_categorical_accuracy: 0.9543
Epoch 36/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0044 - categorical_accuracy: 0.9769 - val_loss: 0.0083 - val_categorical_accuracy: 0.9540
Epoch 37/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0043 - categorical_accuracy: 0.9773 - val_loss: 0.0082 - val_categorical_accuracy: 0.9541
Epoch 38/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0043 - categorical_accuracy: 0.9774 - val_loss: 0.0083 - val_categorical_accuracy: 0.9545
Epoch 39/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0043 - categorical_accuracy: 0.9775 - val_loss: 0.0082 - val_categorical_accuracy: 0.9544

Epoch 00039: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 40/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0043 - categorical_accuracy: 0.9774 - val_loss: 0.0082 - val_categorical_accuracy: 0.9547
Epoch 41/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0042 - categorical_accuracy: 0.9776 - val_loss: 0.0082 - val_categorical_accuracy: 0.9547
Epoch 42/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0042 - categorical_accuracy: 0.9776 - val_loss: 0.0082 - val_categorical_accuracy: 0.9550
Epoch 43/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0042 - categorical_accuracy: 0.9783 - val_loss: 0.0082 - val_categorical_accuracy: 0.9552
Epoch 44/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0043 - categorical_accuracy: 0.9777 - val_loss: 0.0082 - val_categorical_accuracy: 0.9550
Epoch 45/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0041 - categorical_accuracy: 0.9784 - val_loss: 0.0082 - val_categorical_accuracy: 0.9546
Epoch 46/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0042 - categorical_accuracy: 0.9779 - val_loss: 0.0082 - val_categorical_accuracy: 0.9552
Epoch 47/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0042 - categorical_accuracy: 0.9782 - val_loss: 0.0082 - val_categorical_accuracy: 0.9548
Epoch 48/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0042 - categorical_accuracy: 0.9780 - val_loss: 0.0082 - val_categorical_accuracy: 0.9547
Epoch 49/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0042 - categorical_accuracy: 0.9777 - val_loss: 0.0082 - val_categorical_accuracy: 0.9552

Epoch 00049: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 50/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0041 - categorical_accuracy: 0.9786 - val_loss: 0.0082 - val_categorical_accuracy: 0.9548
Epoch 51/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0042 - categorical_accuracy: 0.9777 - val_loss: 0.0082 - val_categorical_accuracy: 0.9550
Epoch 52/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0042 - categorical_accuracy: 0.9777 - val_loss: 0.0082 - val_categorical_accuracy: 0.9548
Epoch 53/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0042 - categorical_accuracy: 0.9782 - val_loss: 0.0082 - val_categorical_accuracy: 0.9550
Epoch 54/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0042 - categorical_accuracy: 0.9779 - val_loss: 0.0082 - val_categorical_accuracy: 0.9548

Epoch 00054: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 55/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0042 - categorical_accuracy: 0.9782 - val_loss: 0.0082 - val_categorical_accuracy: 0.9547
Epoch 56/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0042 - categorical_accuracy: 0.9777 - val_loss: 0.0082 - val_categorical_accuracy: 0.9547
Epoch 57/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0042 - categorical_accuracy: 0.9778 - val_loss: 0.0082 - val_categorical_accuracy: 0.9550
Epoch 58/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0042 - categorical_accuracy: 0.9783 - val_loss: 0.0082 - val_categorical_accuracy: 0.9546
Epoch 59/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0041 - categorical_accuracy: 0.9787 - val_loss: 0.0082 - val_categorical_accuracy: 0.9548

Epoch 00059: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 00059: early stopping
========= generating oof predictions 03:14:33 =========
========= generating test set predictions 03:14:46 =========
========= fitting 3 th model 03:17:27 =========
Epoch 1/65535
420/420 [==============================] - 185s 440ms/step - loss: 0.1136 - categorical_accuracy: 0.2733 - val_loss: 0.1043 - val_categorical_accuracy: 0.3441
Epoch 2/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0622 - categorical_accuracy: 0.6245 - val_loss: 0.0554 - val_categorical_accuracy: 0.6781
Epoch 3/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0386 - categorical_accuracy: 0.7767 - val_loss: 0.0473 - val_categorical_accuracy: 0.7307
Epoch 4/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0297 - categorical_accuracy: 0.8305 - val_loss: 0.0260 - val_categorical_accuracy: 0.8541
Epoch 5/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0244 - categorical_accuracy: 0.8635 - val_loss: 0.0271 - val_categorical_accuracy: 0.8513
Epoch 6/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0214 - categorical_accuracy: 0.8796 - val_loss: 0.0218 - val_categorical_accuracy: 0.8817
Epoch 7/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0190 - categorical_accuracy: 0.8929 - val_loss: 0.0208 - val_categorical_accuracy: 0.8873
Epoch 8/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0172 - categorical_accuracy: 0.9049 - val_loss: 0.0178 - val_categorical_accuracy: 0.9006
Epoch 9/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0159 - categorical_accuracy: 0.9111 - val_loss: 0.0237 - val_categorical_accuracy: 0.8757
Epoch 10/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0144 - categorical_accuracy: 0.9209 - val_loss: 0.0173 - val_categorical_accuracy: 0.9103
Epoch 11/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0138 - categorical_accuracy: 0.9217 - val_loss: 0.0169 - val_categorical_accuracy: 0.9099
Epoch 12/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0128 - categorical_accuracy: 0.9285 - val_loss: 0.0174 - val_categorical_accuracy: 0.9033
Epoch 13/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0123 - categorical_accuracy: 0.9308 - val_loss: 0.0161 - val_categorical_accuracy: 0.9157
Epoch 14/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0116 - categorical_accuracy: 0.9360 - val_loss: 0.0138 - val_categorical_accuracy: 0.9207
Epoch 15/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0112 - categorical_accuracy: 0.9376 - val_loss: 0.0164 - val_categorical_accuracy: 0.9090
Epoch 16/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0104 - categorical_accuracy: 0.9419 - val_loss: 0.0159 - val_categorical_accuracy: 0.9113
Epoch 17/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0101 - categorical_accuracy: 0.9437 - val_loss: 0.0129 - val_categorical_accuracy: 0.9320
Epoch 18/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0096 - categorical_accuracy: 0.9469 - val_loss: 0.0127 - val_categorical_accuracy: 0.9268
Epoch 19/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0093 - categorical_accuracy: 0.9475 - val_loss: 0.0155 - val_categorical_accuracy: 0.9116
Epoch 20/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0088 - categorical_accuracy: 0.9505 - val_loss: 0.0169 - val_categorical_accuracy: 0.9150
Epoch 21/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0086 - categorical_accuracy: 0.9519 - val_loss: 0.0136 - val_categorical_accuracy: 0.9309
Epoch 22/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0082 - categorical_accuracy: 0.9539 - val_loss: 0.0123 - val_categorical_accuracy: 0.9325
Epoch 23/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0081 - categorical_accuracy: 0.9535 - val_loss: 0.0147 - val_categorical_accuracy: 0.9294
Epoch 24/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0077 - categorical_accuracy: 0.9565 - val_loss: 0.0150 - val_categorical_accuracy: 0.9256
Epoch 25/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0073 - categorical_accuracy: 0.9584 - val_loss: 0.0191 - val_categorical_accuracy: 0.9003
Epoch 26/65535
420/420 [==============================] - 182s 435ms/step - loss: 0.0072 - categorical_accuracy: 0.9591 - val_loss: 0.0147 - val_categorical_accuracy: 0.9211
Epoch 27/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0073 - categorical_accuracy: 0.9605 - val_loss: 0.0126 - val_categorical_accuracy: 0.9298
Epoch 28/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0067 - categorical_accuracy: 0.9620 - val_loss: 0.0128 - val_categorical_accuracy: 0.9269

Epoch 00028: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 29/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0050 - categorical_accuracy: 0.9720 - val_loss: 0.0090 - val_categorical_accuracy: 0.9470
Epoch 30/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0045 - categorical_accuracy: 0.9754 - val_loss: 0.0086 - val_categorical_accuracy: 0.9563
Epoch 31/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0043 - categorical_accuracy: 0.9767 - val_loss: 0.0092 - val_categorical_accuracy: 0.9470
Epoch 32/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0042 - categorical_accuracy: 0.9770 - val_loss: 0.0090 - val_categorical_accuracy: 0.9478
Epoch 33/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0040 - categorical_accuracy: 0.9775 - val_loss: 0.0093 - val_categorical_accuracy: 0.9477
Epoch 34/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0040 - categorical_accuracy: 0.9785 - val_loss: 0.0091 - val_categorical_accuracy: 0.9492
Epoch 35/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0038 - categorical_accuracy: 0.9788 - val_loss: 0.0093 - val_categorical_accuracy: 0.9484
Epoch 36/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0038 - categorical_accuracy: 0.9790 - val_loss: 0.0093 - val_categorical_accuracy: 0.9484

Epoch 00036: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 37/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0036 - categorical_accuracy: 0.9806 - val_loss: 0.0088 - val_categorical_accuracy: 0.9499
Epoch 38/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0036 - categorical_accuracy: 0.9807 - val_loss: 0.0088 - val_categorical_accuracy: 0.9499
Epoch 39/65535
420/420 [==============================] - 183s 437ms/step - loss: 0.0036 - categorical_accuracy: 0.9810 - val_loss: 0.0087 - val_categorical_accuracy: 0.9502
Epoch 40/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0035 - categorical_accuracy: 0.9812 - val_loss: 0.0089 - val_categorical_accuracy: 0.9501
Epoch 41/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0035 - categorical_accuracy: 0.9814 - val_loss: 0.0088 - val_categorical_accuracy: 0.9509

Epoch 00041: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 42/65535
420/420 [==============================] - 185s 440ms/step - loss: 0.0033 - categorical_accuracy: 0.9821 - val_loss: 0.0089 - val_categorical_accuracy: 0.9506
Epoch 43/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0034 - categorical_accuracy: 0.9818 - val_loss: 0.0089 - val_categorical_accuracy: 0.9504
Epoch 44/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0034 - categorical_accuracy: 0.9816 - val_loss: 0.0089 - val_categorical_accuracy: 0.9504
Epoch 45/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0034 - categorical_accuracy: 0.9820 - val_loss: 0.0089 - val_categorical_accuracy: 0.9508
Epoch 00045: early stopping
========= generating oof predictions 05:34:53 =========
========= generating test set predictions 05:35:06 =========
========= fitting 4 th model 05:37:48 =========
Epoch 1/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.1126 - categorical_accuracy: 0.2820 - val_loss: 0.0897 - val_categorical_accuracy: 0.4421
Epoch 2/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0573 - categorical_accuracy: 0.6587 - val_loss: 0.0424 - val_categorical_accuracy: 0.7552
Epoch 3/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0366 - categorical_accuracy: 0.7877 - val_loss: 0.0360 - val_categorical_accuracy: 0.7959
Epoch 4/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0285 - categorical_accuracy: 0.8382 - val_loss: 0.0439 - val_categorical_accuracy: 0.7681
Epoch 5/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0241 - categorical_accuracy: 0.8643 - val_loss: 0.0278 - val_categorical_accuracy: 0.8460
Epoch 6/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0208 - categorical_accuracy: 0.8835 - val_loss: 0.0201 - val_categorical_accuracy: 0.8872
Epoch 7/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0187 - categorical_accuracy: 0.8948 - val_loss: 0.0203 - val_categorical_accuracy: 0.8898
Epoch 8/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0169 - categorical_accuracy: 0.9065 - val_loss: 0.0214 - val_categorical_accuracy: 0.8819
Epoch 9/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0157 - categorical_accuracy: 0.9113 - val_loss: 0.0238 - val_categorical_accuracy: 0.8687
Epoch 10/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0143 - categorical_accuracy: 0.9209 - val_loss: 0.0193 - val_categorical_accuracy: 0.8955
Epoch 11/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0137 - categorical_accuracy: 0.9239 - val_loss: 0.0188 - val_categorical_accuracy: 0.8997
Epoch 12/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0130 - categorical_accuracy: 0.9287 - val_loss: 0.0168 - val_categorical_accuracy: 0.9105
Epoch 13/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0120 - categorical_accuracy: 0.9340 - val_loss: 0.0160 - val_categorical_accuracy: 0.9162
Epoch 14/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0114 - categorical_accuracy: 0.9383 - val_loss: 0.0183 - val_categorical_accuracy: 0.9005
Epoch 15/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0108 - categorical_accuracy: 0.9406 - val_loss: 0.0141 - val_categorical_accuracy: 0.9236
Epoch 16/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0106 - categorical_accuracy: 0.9412 - val_loss: 0.0135 - val_categorical_accuracy: 0.9278
Epoch 17/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0101 - categorical_accuracy: 0.9430 - val_loss: 0.0129 - val_categorical_accuracy: 0.9325
Epoch 18/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0094 - categorical_accuracy: 0.9474 - val_loss: 0.0145 - val_categorical_accuracy: 0.9205
Epoch 19/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0092 - categorical_accuracy: 0.9491 - val_loss: 0.0120 - val_categorical_accuracy: 0.9334
Epoch 20/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0088 - categorical_accuracy: 0.9516 - val_loss: 0.0239 - val_categorical_accuracy: 0.8718
Epoch 21/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0086 - categorical_accuracy: 0.9524 - val_loss: 0.0126 - val_categorical_accuracy: 0.9327
Epoch 22/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0081 - categorical_accuracy: 0.9547 - val_loss: 0.0126 - val_categorical_accuracy: 0.9284
Epoch 23/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0081 - categorical_accuracy: 0.9546 - val_loss: 0.0121 - val_categorical_accuracy: 0.9310
Epoch 24/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0075 - categorical_accuracy: 0.9584 - val_loss: 0.0122 - val_categorical_accuracy: 0.9366
Epoch 25/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0075 - categorical_accuracy: 0.9587 - val_loss: 0.0132 - val_categorical_accuracy: 0.9296

Epoch 00025: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 26/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0056 - categorical_accuracy: 0.9689 - val_loss: 0.0092 - val_categorical_accuracy: 0.9461
Epoch 27/65535
420/420 [==============================] - 183s 436ms/step - loss: 0.0050 - categorical_accuracy: 0.9731 - val_loss: 0.0095 - val_categorical_accuracy: 0.9460
Epoch 28/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0049 - categorical_accuracy: 0.9731 - val_loss: 0.0097 - val_categorical_accuracy: 0.9464
Epoch 29/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0046 - categorical_accuracy: 0.9743 - val_loss: 0.0095 - val_categorical_accuracy: 0.9474
Epoch 30/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0045 - categorical_accuracy: 0.9757 - val_loss: 0.0098 - val_categorical_accuracy: 0.9456
Epoch 31/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0045 - categorical_accuracy: 0.9757 - val_loss: 0.0094 - val_categorical_accuracy: 0.9449
Epoch 32/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0044 - categorical_accuracy: 0.9759 - val_loss: 0.0083 - val_categorical_accuracy: 0.9561
Epoch 33/65535
420/420 [==============================] - 184s 438ms/step - loss: 0.0043 - categorical_accuracy: 0.9764 - val_loss: 0.0096 - val_categorical_accuracy: 0.9459
Epoch 34/65535
420/420 [==============================] - 184s 437ms/step - loss: 0.0042 - categorical_accuracy: 0.9770 - val_loss: 0.0090 - val_categorical_accuracy: 0.9475
Epoch 35/65535
420/420 [==============================] - 182s 434ms/step - loss: 0.0041 - categorical_accuracy: 0.9776 - val_loss: 0.0094 - val_categorical_accuracy: 0.9475
Epoch 36/65535
420/420 [==============================] - 182s 433ms/step - loss: 0.0041 - categorical_accuracy: 0.9777 - val_loss: 0.0094 - val_categorical_accuracy: 0.9475
Epoch 37/65535
420/420 [==============================] - 183s 435ms/step - loss: 0.0040 - categorical_accuracy: 0.9779 - val_loss: 0.0095 - val_categorical_accuracy: 0.9475
Epoch 38/65535
420/420 [==============================] - 188s 448ms/step - loss: 0.0039 - categorical_accuracy: 0.9788 - val_loss: 0.0095 - val_categorical_accuracy: 0.9460

Epoch 00038: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 39/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0037 - categorical_accuracy: 0.9802 - val_loss: 0.0085 - val_categorical_accuracy: 0.9585
Epoch 40/65535
420/420 [==============================] - 195s 463ms/step - loss: 0.0037 - categorical_accuracy: 0.9801 - val_loss: 0.0076 - val_categorical_accuracy: 0.9594
Epoch 41/65535
420/420 [==============================] - 194s 461ms/step - loss: 0.0036 - categorical_accuracy: 0.9803 - val_loss: 0.0082 - val_categorical_accuracy: 0.9589
Epoch 42/65535
420/420 [==============================] - 194s 463ms/step - loss: 0.0036 - categorical_accuracy: 0.9803 - val_loss: 0.0082 - val_categorical_accuracy: 0.9589
Epoch 43/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0036 - categorical_accuracy: 0.9809 - val_loss: 0.0083 - val_categorical_accuracy: 0.9593
Epoch 44/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0035 - categorical_accuracy: 0.9807 - val_loss: 0.0083 - val_categorical_accuracy: 0.9592
Epoch 45/65535
420/420 [==============================] - 196s 466ms/step - loss: 0.0036 - categorical_accuracy: 0.9807 - val_loss: 0.0079 - val_categorical_accuracy: 0.9590
Epoch 46/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0035 - categorical_accuracy: 0.9814 - val_loss: 0.0084 - val_categorical_accuracy: 0.9592

Epoch 00046: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 47/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0035 - categorical_accuracy: 0.9815 - val_loss: 0.0085 - val_categorical_accuracy: 0.9580
Epoch 48/65535
420/420 [==============================] - 194s 463ms/step - loss: 0.0034 - categorical_accuracy: 0.9814 - val_loss: 0.0084 - val_categorical_accuracy: 0.9587
Epoch 49/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0035 - categorical_accuracy: 0.9809 - val_loss: 0.0084 - val_categorical_accuracy: 0.9585
Epoch 50/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0035 - categorical_accuracy: 0.9811 - val_loss: 0.0082 - val_categorical_accuracy: 0.9594
Epoch 51/65535
420/420 [==============================] - 194s 462ms/step - loss: 0.0034 - categorical_accuracy: 0.9816 - val_loss: 0.0082 - val_categorical_accuracy: 0.9592

Epoch 00051: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 52/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0034 - categorical_accuracy: 0.9815 - val_loss: 0.0082 - val_categorical_accuracy: 0.9589
Epoch 53/65535
420/420 [==============================] - 195s 464ms/step - loss: 0.0034 - categorical_accuracy: 0.9821 - val_loss: 0.0082 - val_categorical_accuracy: 0.9591
Epoch 54/65535
420/420 [==============================] - 195s 463ms/step - loss: 0.0035 - categorical_accuracy: 0.9815 - val_loss: 0.0082 - val_categorical_accuracy: 0.9590
Epoch 55/65535
420/420 [==============================] - 196s 465ms/step - loss: 0.0034 - categorical_accuracy: 0.9815 - val_loss: 0.0082 - val_categorical_accuracy: 0.9590
Epoch 00055: early stopping
========= generating oof predictions 08:29:15 =========
========= generating test set predictions 08:29:29 =========
========= fitting 5 th model 17:32:20 =========
Epoch 1/65535
420/420 [==============================] - 197s 469ms/step - loss: 0.1093 - categorical_accuracy: 0.3003 - val_loss: 0.0882 - val_categorical_accuracy: 0.4619
Epoch 2/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0566 - categorical_accuracy: 0.6596 - val_loss: 0.0476 - val_categorical_accuracy: 0.7196
Epoch 3/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0366 - categorical_accuracy: 0.7871 - val_loss: 0.0340 - val_categorical_accuracy: 0.8068
Epoch 4/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0284 - categorical_accuracy: 0.8383 - val_loss: 0.0307 - val_categorical_accuracy: 0.8246
Epoch 5/65535
420/420 [==============================] - 195s 463ms/step - loss: 0.0240 - categorical_accuracy: 0.8649 - val_loss: 0.0269 - val_categorical_accuracy: 0.8518
Epoch 6/65535
420/420 [==============================] - 194s 463ms/step - loss: 0.0213 - categorical_accuracy: 0.8802 - val_loss: 0.0247 - val_categorical_accuracy: 0.8628
Epoch 7/65535
420/420 [==============================] - 194s 463ms/step - loss: 0.0191 - categorical_accuracy: 0.8929 - val_loss: 0.0207 - val_categorical_accuracy: 0.8834
Epoch 8/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0173 - categorical_accuracy: 0.9039 - val_loss: 0.0204 - val_categorical_accuracy: 0.8868
Epoch 9/65535
420/420 [==============================] - 197s 469ms/step - loss: 0.0160 - categorical_accuracy: 0.9115 - val_loss: 0.0176 - val_categorical_accuracy: 0.9031
Epoch 10/65535
420/420 [==============================] - 197s 470ms/step - loss: 0.0148 - categorical_accuracy: 0.9178 - val_loss: 0.0168 - val_categorical_accuracy: 0.9047
Epoch 11/65535
420/420 [==============================] - 197s 468ms/step - loss: 0.0143 - categorical_accuracy: 0.9204 - val_loss: 0.0174 - val_categorical_accuracy: 0.9023
Epoch 12/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0131 - categorical_accuracy: 0.9279 - val_loss: 0.0164 - val_categorical_accuracy: 0.9120
Epoch 13/65535
420/420 [==============================] - 197s 469ms/step - loss: 0.0125 - categorical_accuracy: 0.9306 - val_loss: 0.0166 - val_categorical_accuracy: 0.9058
Epoch 14/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0117 - categorical_accuracy: 0.9352 - val_loss: 0.0157 - val_categorical_accuracy: 0.9102
Epoch 15/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0111 - categorical_accuracy: 0.9378 - val_loss: 0.0154 - val_categorical_accuracy: 0.9179
Epoch 16/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0105 - categorical_accuracy: 0.9416 - val_loss: 0.0140 - val_categorical_accuracy: 0.9229
Epoch 17/65535
420/420 [==============================] - 197s 469ms/step - loss: 0.0099 - categorical_accuracy: 0.9458 - val_loss: 0.0145 - val_categorical_accuracy: 0.9211
Epoch 18/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0095 - categorical_accuracy: 0.9469 - val_loss: 0.0143 - val_categorical_accuracy: 0.9277
Epoch 19/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0093 - categorical_accuracy: 0.9485 - val_loss: 0.0167 - val_categorical_accuracy: 0.9125
Epoch 20/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0088 - categorical_accuracy: 0.9507 - val_loss: 0.0151 - val_categorical_accuracy: 0.9239
Epoch 21/65535
420/420 [==============================] - 199s 475ms/step - loss: 0.0085 - categorical_accuracy: 0.9523 - val_loss: 0.0154 - val_categorical_accuracy: 0.9224
Epoch 22/65535
420/420 [==============================] - 211s 502ms/step - loss: 0.0084 - categorical_accuracy: 0.9532 - val_loss: 0.0135 - val_categorical_accuracy: 0.9285
Epoch 23/65535
420/420 [==============================] - 213s 507ms/step - loss: 0.0079 - categorical_accuracy: 0.9558 - val_loss: 0.0132 - val_categorical_accuracy: 0.9287
Epoch 24/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0078 - categorical_accuracy: 0.9566 - val_loss: 0.0164 - val_categorical_accuracy: 0.9289
Epoch 25/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0075 - categorical_accuracy: 0.9572 - val_loss: 0.0141 - val_categorical_accuracy: 0.9271
Epoch 26/65535
420/420 [==============================] - 215s 511ms/step - loss: 0.0073 - categorical_accuracy: 0.9582 - val_loss: 0.0156 - val_categorical_accuracy: 0.9286
Epoch 27/65535
420/420 [==============================] - 213s 508ms/step - loss: 0.0069 - categorical_accuracy: 0.9615 - val_loss: 0.0168 - val_categorical_accuracy: 0.9259
Epoch 28/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0068 - categorical_accuracy: 0.9616 - val_loss: 0.0128 - val_categorical_accuracy: 0.9287
Epoch 29/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0066 - categorical_accuracy: 0.9627 - val_loss: 0.0119 - val_categorical_accuracy: 0.9411
Epoch 30/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0065 - categorical_accuracy: 0.9641 - val_loss: 0.0139 - val_categorical_accuracy: 0.9270
Epoch 31/65535
420/420 [==============================] - 214s 509ms/step - loss: 0.0061 - categorical_accuracy: 0.9643 - val_loss: 0.0122 - val_categorical_accuracy: 0.9342
Epoch 32/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0061 - categorical_accuracy: 0.9654 - val_loss: 0.0160 - val_categorical_accuracy: 0.9183
Epoch 33/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0058 - categorical_accuracy: 0.9669 - val_loss: 0.0133 - val_categorical_accuracy: 0.9278
Epoch 34/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0058 - categorical_accuracy: 0.9674 - val_loss: 0.0137 - val_categorical_accuracy: 0.9272
Epoch 35/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0057 - categorical_accuracy: 0.9675 - val_loss: 0.0126 - val_categorical_accuracy: 0.9360
Epoch 00035: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 36/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0043 - categorical_accuracy: 0.9760 - val_loss: 0.0091 - val_categorical_accuracy: 0.9539
Epoch 37/65535
420/420 [==============================] - 215s 511ms/step - loss: 0.0038 - categorical_accuracy: 0.9786 - val_loss: 0.0091 - val_categorical_accuracy: 0.9541
Epoch 38/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0036 - categorical_accuracy: 0.9798 - val_loss: 0.0090 - val_categorical_accuracy: 0.9553
Epoch 39/65535
420/420 [==============================] - 215s 513ms/step - loss: 0.0035 - categorical_accuracy: 0.9811 - val_loss: 0.0098 - val_categorical_accuracy: 0.9560
Epoch 40/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0034 - categorical_accuracy: 0.9810 - val_loss: 0.0093 - val_categorical_accuracy: 0.9559
Epoch 41/65535
420/420 [==============================] - 214s 511ms/step - loss: 0.0034 - categorical_accuracy: 0.9812 - val_loss: 0.0092 - val_categorical_accuracy: 0.9559
Epoch 42/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0033 - categorical_accuracy: 0.9816 - val_loss: 0.0085 - val_categorical_accuracy: 0.9581
Epoch 43/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0033 - categorical_accuracy: 0.9818 - val_loss: 0.0090 - val_categorical_accuracy: 0.9559
Epoch 44/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0032 - categorical_accuracy: 0.9819 - val_loss: 0.0092 - val_categorical_accuracy: 0.9568
Epoch 45/65535
420/420 [==============================] - 216s 513ms/step - loss: 0.0032 - categorical_accuracy: 0.9826 - val_loss: 0.0098 - val_categorical_accuracy: 0.9556
Epoch 46/65535
420/420 [==============================] - 215s 511ms/step - loss: 0.0031 - categorical_accuracy: 0.9830 - val_loss: 0.0096 - val_categorical_accuracy: 0.9573
Epoch 47/65535
420/420 [==============================] - 215s 513ms/step - loss: 0.0030 - categorical_accuracy: 0.9836 - val_loss: 0.0091 - val_categorical_accuracy: 0.9562
Epoch 48/65535
420/420 [==============================] - 215s 511ms/step - loss: 0.0029 - categorical_accuracy: 0.9841 - val_loss: 0.0099 - val_categorical_accuracy: 0.9559
Epoch 00048: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 49/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0029 - categorical_accuracy: 0.9843 - val_loss: 0.0085 - val_categorical_accuracy: 0.9600
Epoch 50/65535
420/420 [==============================] - 215s 511ms/step - loss: 0.0028 - categorical_accuracy: 0.9839 - val_loss: 0.0086 - val_categorical_accuracy: 0.9592
Epoch 51/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0028 - categorical_accuracy: 0.9845 - val_loss: 0.0085 - val_categorical_accuracy: 0.9593
Epoch 52/65535
420/420 [==============================] - 214s 510ms/step - loss: 0.0028 - categorical_accuracy: 0.9844 - val_loss: 0.0084 - val_categorical_accuracy: 0.9598
Epoch 53/65535
420/420 [==============================] - 215s 511ms/step - loss: 0.0028 - categorical_accuracy: 0.9852 - val_loss: 0.0085 - val_categorical_accuracy: 0.9592
Epoch 54/65535
420/420 [==============================] - 215s 512ms/step - loss: 0.0027 - categorical_accuracy: 0.9851 - val_loss: 0.0085 - val_categorical_accuracy: 0.9597
Epoch 55/65535
420/420 [==============================] - 215s 513ms/step - loss: 0.0027 - categorical_accuracy: 0.9852 - val_loss: 0.0084 - val_categorical_accuracy: 0.9605
Epoch 56/65535
420/420 [==============================] - 206s 490ms/step - loss: 0.0027 - categorical_accuracy: 0.9848 - val_loss: 0.0085 - val_categorical_accuracy: 0.9604
Epoch 57/65535
420/420 [==============================] - 200s 476ms/step - loss: 0.0027 - categorical_accuracy: 0.9851 - val_loss: 0.0084 - val_categorical_accuracy: 0.9606
Epoch 58/65535
420/420 [==============================] - 196s 466ms/step - loss: 0.0028 - categorical_accuracy: 0.9849 - val_loss: 0.0087 - val_categorical_accuracy: 0.9594
Epoch 00058: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 59/65535
420/420 [==============================] - 195s 465ms/step - loss: 0.0026 - categorical_accuracy: 0.9854 - val_loss: 0.0085 - val_categorical_accuracy: 0.9601
Epoch 60/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0026 - categorical_accuracy: 0.9856 - val_loss: 0.0085 - val_categorical_accuracy: 0.9600
Epoch 61/65535
420/420 [==============================] - 197s 469ms/step - loss: 0.0027 - categorical_accuracy: 0.9851 - val_loss: 0.0085 - val_categorical_accuracy: 0.9601
Epoch 62/65535
420/420 [==============================] - 197s 469ms/step - loss: 0.0026 - categorical_accuracy: 0.9853 - val_loss: 0.0085 - val_categorical_accuracy: 0.9600
Epoch 63/65535
420/420 [==============================] - 197s 468ms/step - loss: 0.0026 - categorical_accuracy: 0.9851 - val_loss: 0.0085 - val_categorical_accuracy: 0.9607
Epoch 00063: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 64/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0026 - categorical_accuracy: 0.9855 - val_loss: 0.0085 - val_categorical_accuracy: 0.9606
Epoch 65/65535
420/420 [==============================] - 197s 469ms/step - loss: 0.0027 - categorical_accuracy: 0.9852 - val_loss: 0.0085 - val_categorical_accuracy: 0.9606
Epoch 66/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0026 - categorical_accuracy: 0.9856 - val_loss: 0.0085 - val_categorical_accuracy: 0.9606
Epoch 67/65535
420/420 [==============================] - 196s 467ms/step - loss: 0.0026 - categorical_accuracy: 0.9853 - val_loss: 0.0085 - val_categorical_accuracy: 0.9606
Epoch 00067: early stopping
========= generating oof predictions 23:22:11 =========
========= generating test set predictions 23:22:25 =========
train loss avg 0.0034621412723576314 -- std 0.0005090567055871143, val loss avg 0.008507592018460365 -- std 0.000290914056186866
train acc avg 0.9814138306735215 -- std 0.002282310004780581, val acc avg 0.9568247484345903 -- std 0.003594557364268731
mean nb epochs 54.4
dump oof predicted probs
dump test set predicted probs
