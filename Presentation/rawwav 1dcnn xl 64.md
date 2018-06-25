ren (master *+) python $ python train.py rawwav 1dcnn xl 64
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (16000, 1) ===========
------------- SUMMARY OF MODEL -------------
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 16000, 1)     0
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 16000, 4)     36          input_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 16000, 4)     16          conv1d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 16000, 4)     0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 16000, 4)     132         activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 16000, 4)     16          conv1d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 16000, 4)     0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 8000, 4)      0           activation_2[0][0]
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 8000, 8)      264         max_pooling1d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 8000, 8)      32          conv1d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 8000, 8)      0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 8000, 8)      520         activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 8000, 8)      32          conv1d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 8000, 8)      0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)  (None, 4000, 8)      0           activation_4[0][0]
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 4000, 16)     1040        max_pooling1d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 4000, 16)     64          conv1d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 4000, 16)     0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 4000, 16)     2064        activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 4000, 16)     64          conv1d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 4000, 16)     0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)  (None, 2000, 16)     0           activation_6[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 2000, 16)     0           max_pooling1d_3[0][0]
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 2000, 32)     4128        dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 2000, 32)     128         conv1d_7[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 2000, 32)     0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 2000, 32)     8224        activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 2000, 32)     128         conv1d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 2000, 32)     0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
max_pooling1d_4 (MaxPooling1D)  (None, 1000, 32)     0           activation_8[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 1000, 32)     0           max_pooling1d_4[0][0]
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 1000, 64)     16448       dropout_2[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 1000, 64)     256         conv1d_9[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 1000, 64)     0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv1d_10 (Conv1D)              (None, 1000, 64)     32832       activation_9[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 1000, 64)     256         conv1d_10[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 1000, 64)     0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
max_pooling1d_5 (MaxPooling1D)  (None, 500, 64)      0           activation_10[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 500, 64)      0           max_pooling1d_5[0][0]
__________________________________________________________________________________________________
conv1d_11 (Conv1D)              (None, 500, 128)     65664       dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 500, 128)     512         conv1d_11[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 500, 128)     0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
conv1d_12 (Conv1D)              (None, 500, 128)     131200      activation_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 500, 128)     512         conv1d_12[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 500, 128)     0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
conv1d_13 (Conv1D)              (None, 500, 128)     131200      activation_12[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 500, 128)     512         conv1d_13[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 500, 128)     0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
max_pooling1d_6 (MaxPooling1D)  (None, 250, 128)     0           activation_13[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 250, 128)     0           max_pooling1d_6[0][0]
__________________________________________________________________________________________________
conv1d_14 (Conv1D)              (None, 250, 256)     262400      dropout_4[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 250, 256)     1024        conv1d_14[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 250, 256)     0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
conv1d_15 (Conv1D)              (None, 250, 256)     524544      activation_14[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 250, 256)     1024        conv1d_15[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 250, 256)     0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
conv1d_16 (Conv1D)              (None, 250, 256)     524544      activation_15[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 250, 256)     1024        conv1d_16[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 250, 256)     0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
max_pooling1d_7 (MaxPooling1D)  (None, 125, 256)     0           activation_16[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 125, 256)     0           max_pooling1d_7[0][0]
__________________________________________________________________________________________________
conv1d_17 (Conv1D)              (None, 125, 512)     1049088     dropout_5[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 125, 512)     2048        conv1d_17[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 125, 512)     0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
conv1d_18 (Conv1D)              (None, 125, 512)     2097664     activation_17[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 125, 512)     2048        conv1d_18[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 125, 512)     0           batch_normalization_18[0][0]
__________________________________________________________________________________________________
conv1d_19 (Conv1D)              (None, 125, 512)     2097664     activation_18[0][0]
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 125, 512)     2048        conv1d_19[0][0]
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 125, 512)     0           batch_normalization_19[0][0]
__________________________________________________________________________________________________
max_pooling1d_8 (MaxPooling1D)  (None, 62, 512)      0           activation_19[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 62, 512)      0           max_pooling1d_8[0][0]
__________________________________________________________________________________________________
conv1d_20 (Conv1D)              (None, 62, 1024)     4195328     dropout_6[0][0]
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 62, 1024)     4096        conv1d_20[0][0]
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 62, 1024)     0           batch_normalization_20[0][0]
__________________________________________________________________________________________________
conv1d_21 (Conv1D)              (None, 62, 1024)     8389632     activation_20[0][0]
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 62, 1024)     4096        conv1d_21[0][0]
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 62, 1024)     0           batch_normalization_21[0][0]
__________________________________________________________________________________________________
max_pooling1d_9 (MaxPooling1D)  (None, 31, 1024)     0           activation_21[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 31, 1024)     0           max_pooling1d_9[0][0]
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 1024)         0           dropout_7[0][0]
__________________________________________________________________________________________________
global_average_pooling1d_1 (Glo (None, 1024)         0           dropout_7[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 2048)         0           global_max_pooling1d_1[0][0]
                                                                 global_average_pooling1d_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1024)         2098176     concatenate_1[0][0]
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 1024)         4096        dense_1[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 1024)         0           batch_normalization_22[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 512)          524800      dropout_8[0][0]
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 512)          2048        dense_2[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 512)          0           batch_normalization_23[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 31)           15903       dropout_9[0][0]
==================================================================================================
Total params: 22,199,575
Trainable params: 22,186,535
Non-trainable params: 13,040
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 10:30:17 =========
Epoch 1/65535
839/839 [==============================] - 186s 221ms/step - loss: 0.0843 - categorical_accuracy: 0.4779 - val_loss: 0.0517 - val_categorical_accuracy: 0.6934
Epoch 2/65535
839/839 [==============================] - 185s 220ms/step - loss: 0.0279 - categorical_accuracy: 0.8401 - val_loss: 0.0227 - val_categorical_accuracy: 0.8725
Epoch 3/65535
839/839 [==============================] - 182s 217ms/step - loss: 0.0186 - categorical_accuracy: 0.8961 - val_loss: 0.0194 - val_categorical_accuracy: 0.8916
Epoch 4/65535
839/839 [==============================] - 182s 217ms/step - loss: 0.0144 - categorical_accuracy: 0.9220 - val_loss: 0.0147 - val_categorical_accuracy: 0.9204
Epoch 5/65535
839/839 [==============================] - 181s 216ms/step - loss: 0.0119 - categorical_accuracy: 0.9345 - val_loss: 0.0118 - val_categorical_accuracy: 0.9373
Epoch 6/65535
839/839 [==============================] - 183s 218ms/step - loss: 0.0104 - categorical_accuracy: 0.9425 - val_loss: 0.0151 - val_categorical_accuracy: 0.9186
Epoch 7/65535
839/839 [==============================] - 183s 218ms/step - loss: 0.0089 - categorical_accuracy: 0.9515 - val_loss: 0.0141 - val_categorical_accuracy: 0.9271
Epoch 8/65535
839/839 [==============================] - 185s 220ms/step - loss: 0.0081 - categorical_accuracy: 0.9555 - val_loss: 0.0149 - val_categorical_accuracy: 0.9239
Epoch 9/65535
839/839 [==============================] - 180s 214ms/step - loss: 0.0071 - categorical_accuracy: 0.9617 - val_loss: 0.0157 - val_categorical_accuracy: 0.9193
Epoch 10/65535
839/839 [==============================] - 184s 219ms/step - loss: 0.0066 - categorical_accuracy: 0.9636 - val_loss: 0.0126 - val_categorical_accuracy: 0.9376
Epoch 11/65535
839/839 [==============================] - 183s 219ms/step - loss: 0.0061 - categorical_accuracy: 0.9663 - val_loss: 0.0109 - val_categorical_accuracy: 0.9383
Epoch 12/65535
839/839 [==============================] - 182s 217ms/step - loss: 0.0057 - categorical_accuracy: 0.9688 - val_loss: 0.0109 - val_categorical_accuracy: 0.9431
Epoch 13/65535
839/839 [==============================] - 183s 219ms/step - loss: 0.0052 - categorical_accuracy: 0.9713 - val_loss: 0.0140 - val_categorical_accuracy: 0.9323
Epoch 14/65535
839/839 [==============================] - 182s 217ms/step - loss: 0.0047 - categorical_accuracy: 0.9741 - val_loss: 0.0139 - val_categorical_accuracy: 0.9290
Epoch 15/65535
839/839 [==============================] - 181s 216ms/step - loss: 0.0044 - categorical_accuracy: 0.9754 - val_loss: 0.0130 - val_categorical_accuracy: 0.9385
Epoch 16/65535
839/839 [==============================] - 185s 221ms/step - loss: 0.0042 - categorical_accuracy: 0.9766 - val_loss: 0.0127 - val_categorical_accuracy: 0.9431
Epoch 17/65535
839/839 [==============================] - 183s 218ms/step - loss: 0.0039 - categorical_accuracy: 0.9780 - val_loss: 0.0131 - val_categorical_accuracy: 0.9414

Epoch 00017: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 18/65535
839/839 [==============================] - 184s 219ms/step - loss: 0.0025 - categorical_accuracy: 0.9861 - val_loss: 0.0077 - val_categorical_accuracy: 0.9638
Epoch 19/65535
839/839 [==============================] - 183s 218ms/step - loss: 0.0020 - categorical_accuracy: 0.9891 - val_loss: 0.0079 - val_categorical_accuracy: 0.9635
Epoch 20/65535
839/839 [==============================] - 177s 210ms/step - loss: 0.0019 - categorical_accuracy: 0.9902 - val_loss: 0.0072 - val_categorical_accuracy: 0.9650
Epoch 21/65535
839/839 [==============================] - 176s 210ms/step - loss: 0.0017 - categorical_accuracy: 0.9914 - val_loss: 0.0080 - val_categorical_accuracy: 0.9643
Epoch 22/65535
839/839 [==============================] - 186s 221ms/step - loss: 0.0016 - categorical_accuracy: 0.9916 - val_loss: 0.0084 - val_categorical_accuracy: 0.9648
Epoch 23/65535
839/839 [==============================] - 185s 221ms/step - loss: 0.0015 - categorical_accuracy: 0.9922 - val_loss: 0.0080 - val_categorical_accuracy: 0.9652
Epoch 24/65535
839/839 [==============================] - 185s 220ms/step - loss: 0.0014 - categorical_accuracy: 0.9927 - val_loss: 0.0086 - val_categorical_accuracy: 0.9642
Epoch 25/65535
839/839 [==============================] - 183s 218ms/step - loss: 0.0014 - categorical_accuracy: 0.9928 - val_loss: 0.0079 - val_categorical_accuracy: 0.9641
Epoch 26/65535
839/839 [==============================] - 176s 209ms/step - loss: 0.0014 - categorical_accuracy: 0.9933 - val_loss: 0.0093 - val_categorical_accuracy: 0.9632

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 27/65535
839/839 [==============================] - 176s 210ms/step - loss: 0.0012 - categorical_accuracy: 0.9939 - val_loss: 0.0075 - val_categorical_accuracy: 0.9664
Epoch 28/65535
839/839 [==============================] - 176s 210ms/step - loss: 0.0011 - categorical_accuracy: 0.9947 - val_loss: 0.0078 - val_categorical_accuracy: 0.9668
Epoch 29/65535
839/839 [==============================] - 176s 210ms/step - loss: 0.0011 - categorical_accuracy: 0.9943 - val_loss: 0.0073 - val_categorical_accuracy: 0.9667
Epoch 30/65535
839/839 [==============================] - 176s 210ms/step - loss: 0.0011 - categorical_accuracy: 0.9947 - val_loss: 0.0079 - val_categorical_accuracy: 0.9660
Epoch 31/65535
839/839 [==============================] - 176s 210ms/step - loss: 0.0011 - categorical_accuracy: 0.9947 - val_loss: 0.0074 - val_categorical_accuracy: 0.9658

Epoch 00031: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 32/65535
839/839 [==============================] - 177s 210ms/step - loss: 0.0010 - categorical_accuracy: 0.9949 - val_loss: 0.0076 - val_categorical_accuracy: 0.9658
Epoch 33/65535
839/839 [==============================] - 176s 210ms/step - loss: 0.0010 - categorical_accuracy: 0.9951 - val_loss: 0.0076 - val_categorical_accuracy: 0.9658
Epoch 34/65535
839/839 [==============================] - 176s 210ms/step - loss: 0.0010 - categorical_accuracy: 0.9947 - val_loss: 0.0076 - val_categorical_accuracy: 0.9659
Epoch 35/65535
839/839 [==============================] - 176s 210ms/step - loss: 0.0010 - categorical_accuracy: 0.9951 - val_loss: 0.0076 - val_categorical_accuracy: 0.9659
Epoch 00035: early stopping
========= generating oof predictions 12:15:56 =========
========= generating test set predictions 12:16:08 =========
========= fitting 2 th model 12:18:37 =========
Epoch 1/65535
839/839 [==============================] - 179s 213ms/step - loss: 0.0893 - categorical_accuracy: 0.4470 - val_loss: 0.0553 - val_categorical_accuracy: 0.6828
Epoch 2/65535
839/839 [==============================] - 179s 214ms/step - loss: 0.0272 - categorical_accuracy: 0.8456 - val_loss: 0.0281 - val_categorical_accuracy: 0.8418
Epoch 3/65535
839/839 [==============================] - 183s 218ms/step - loss: 0.0179 - categorical_accuracy: 0.9010 - val_loss: 0.0211 - val_categorical_accuracy: 0.8867
Epoch 4/65535
839/839 [==============================] - 186s 221ms/step - loss: 0.0137 - categorical_accuracy: 0.9235 - val_loss: 0.0216 - val_categorical_accuracy: 0.8926
Epoch 5/65535
839/839 [==============================] - 186s 222ms/step - loss: 0.0117 - categorical_accuracy: 0.9362 - val_loss: 0.0155 - val_categorical_accuracy: 0.9189
Epoch 6/65535
839/839 [==============================] - 185s 220ms/step - loss: 0.0100 - categorical_accuracy: 0.9445 - val_loss: 0.0154 - val_categorical_accuracy: 0.9208
Epoch 7/65535
839/839 [==============================] - 194s 231ms/step - loss: 0.0089 - categorical_accuracy: 0.9511 - val_loss: 0.0170 - val_categorical_accuracy: 0.9174
Epoch 8/65535
839/839 [==============================] - 185s 221ms/step - loss: 0.0079 - categorical_accuracy: 0.9565 - val_loss: 0.0136 - val_categorical_accuracy: 0.9267
Epoch 9/65535
839/839 [==============================] - 188s 224ms/step - loss: 0.0072 - categorical_accuracy: 0.9593 - val_loss: 0.0128 - val_categorical_accuracy: 0.9326
Epoch 10/65535
839/839 [==============================] - 188s 223ms/step - loss: 0.0067 - categorical_accuracy: 0.9627 - val_loss: 0.0305 - val_categorical_accuracy: 0.8809
Epoch 11/65535
839/839 [==============================] - 185s 220ms/step - loss: 0.0058 - categorical_accuracy: 0.9674 - val_loss: 0.0140 - val_categorical_accuracy: 0.9300
Epoch 12/65535
839/839 [==============================] - 188s 225ms/step - loss: 0.0053 - categorical_accuracy: 0.9695 - val_loss: 0.0203 - val_categorical_accuracy: 0.9124
Epoch 13/65535
839/839 [==============================] - 188s 224ms/step - loss: 0.0049 - categorical_accuracy: 0.9720 - val_loss: 0.0111 - val_categorical_accuracy: 0.9414
Epoch 14/65535
839/839 [==============================] - 188s 224ms/step - loss: 0.0048 - categorical_accuracy: 0.9733 - val_loss: 0.0244 - val_categorical_accuracy: 0.8990
Epoch 15/65535
839/839 [==============================] - 184s 220ms/step - loss: 0.0042 - categorical_accuracy: 0.9764 - val_loss: 0.0111 - val_categorical_accuracy: 0.9455
Epoch 16/65535
839/839 [==============================] - 186s 222ms/step - loss: 0.0041 - categorical_accuracy: 0.9768 - val_loss: 0.0121 - val_categorical_accuracy: 0.9374
Epoch 17/65535
839/839 [==============================] - 185s 220ms/step - loss: 0.0037 - categorical_accuracy: 0.9790 - val_loss: 0.0107 - val_categorical_accuracy: 0.9445
Epoch 18/65535
839/839 [==============================] - 184s 220ms/step - loss: 0.0037 - categorical_accuracy: 0.9794 - val_loss: 0.0135 - val_categorical_accuracy: 0.9387
Epoch 19/65535
839/839 [==============================] - 185s 220ms/step - loss: 0.0034 - categorical_accuracy: 0.9812 - val_loss: 0.0099 - val_categorical_accuracy: 0.9493
Epoch 20/65535
839/839 [==============================] - 188s 224ms/step - loss: 0.0030 - categorical_accuracy: 0.9835 - val_loss: 0.0148 - val_categorical_accuracy: 0.9351
Epoch 21/65535
839/839 [==============================] - 185s 220ms/step - loss: 0.0032 - categorical_accuracy: 0.9820 - val_loss: 0.0116 - val_categorical_accuracy: 0.9444
Epoch 22/65535
839/839 [==============================] - 187s 223ms/step - loss: 0.0031 - categorical_accuracy: 0.9823 - val_loss: 0.0118 - val_categorical_accuracy: 0.9447
Epoch 23/65535
839/839 [==============================] - 188s 224ms/step - loss: 0.0028 - categorical_accuracy: 0.9848 - val_loss: 0.0102 - val_categorical_accuracy: 0.9493
Epoch 24/65535
839/839 [==============================] - 184s 220ms/step - loss: 0.0026 - categorical_accuracy: 0.9849 - val_loss: 0.0242 - val_categorical_accuracy: 0.9072
Epoch 25/65535
839/839 [==============================] - 190s 226ms/step - loss: 0.0025 - categorical_accuracy: 0.9857 - val_loss: 0.0241 - val_categorical_accuracy: 0.9066

Epoch 00025: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 26/65535
839/839 [==============================] - 191s 228ms/step - loss: 0.0016 - categorical_accuracy: 0.9913 - val_loss: 0.0078 - val_categorical_accuracy: 0.9626
Epoch 27/65535
839/839 [==============================] - 192s 229ms/step - loss: 0.0012 - categorical_accuracy: 0.9935 - val_loss: 0.0078 - val_categorical_accuracy: 0.9633
Epoch 28/65535
839/839 [==============================] - 184s 219ms/step - loss: 0.0011 - categorical_accuracy: 0.9940 - val_loss: 0.0077 - val_categorical_accuracy: 0.9634
Epoch 29/65535
839/839 [==============================] - 183s 218ms/step - loss: 0.0010 - categorical_accuracy: 0.9947 - val_loss: 0.0083 - val_categorical_accuracy: 0.9567
Epoch 30/65535
839/839 [==============================] - 183s 218ms/step - loss: 9.3363e-04 - categorical_accuracy: 0.9951 - val_loss: 0.0076 - val_categorical_accuracy: 0.9644
Epoch 31/65535
839/839 [==============================] - 182s 217ms/step - loss: 8.9567e-04 - categorical_accuracy: 0.9953 - val_loss: 0.0076 - val_categorical_accuracy: 0.9643
Epoch 32/65535
839/839 [==============================] - 185s 220ms/step - loss: 8.4079e-04 - categorical_accuracy: 0.9956 - val_loss: 0.0078 - val_categorical_accuracy: 0.9646
Epoch 33/65535
839/839 [==============================] - 193s 230ms/step - loss: 7.8347e-04 - categorical_accuracy: 0.9960 - val_loss: 0.0089 - val_categorical_accuracy: 0.9634
Epoch 34/65535
839/839 [==============================] - 196s 234ms/step - loss: 7.6265e-04 - categorical_accuracy: 0.9960 - val_loss: 0.0079 - val_categorical_accuracy: 0.9643

Epoch 00034: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 35/65535
839/839 [==============================] - 195s 232ms/step - loss: 6.9146e-04 - categorical_accuracy: 0.9964 - val_loss: 0.0078 - val_categorical_accuracy: 0.9655
Epoch 36/65535
839/839 [==============================] - 184s 220ms/step - loss: 6.6887e-04 - categorical_accuracy: 0.9968 - val_loss: 0.0078 - val_categorical_accuracy: 0.9651
Epoch 37/65535
839/839 [==============================] - 185s 221ms/step - loss: 6.4576e-04 - categorical_accuracy: 0.9966 - val_loss: 0.0078 - val_categorical_accuracy: 0.9649
Epoch 38/65535
839/839 [==============================] - 186s 221ms/step - loss: 6.5195e-04 - categorical_accuracy: 0.9966 - val_loss: 0.0079 - val_categorical_accuracy: 0.9649
Epoch 39/65535
839/839 [==============================] - 191s 227ms/step - loss: 6.2447e-04 - categorical_accuracy: 0.9969 - val_loss: 0.0079 - val_categorical_accuracy: 0.9651

Epoch 00039: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 40/65535
839/839 [==============================] - 190s 226ms/step - loss: 6.3171e-04 - categorical_accuracy: 0.9965 - val_loss: 0.0079 - val_categorical_accuracy: 0.9652
Epoch 41/65535
839/839 [==============================] - 193s 230ms/step - loss: 6.1623e-04 - categorical_accuracy: 0.9970 - val_loss: 0.0079 - val_categorical_accuracy: 0.9652
Epoch 42/65535
839/839 [==============================] - 195s 233ms/step - loss: 5.9964e-04 - categorical_accuracy: 0.9972 - val_loss: 0.0079 - val_categorical_accuracy: 0.9654
Epoch 43/65535
839/839 [==============================] - 187s 223ms/step - loss: 5.8089e-04 - categorical_accuracy: 0.9971 - val_loss: 0.0079 - val_categorical_accuracy: 0.9654
Epoch 44/65535
839/839 [==============================] - 186s 222ms/step - loss: 6.0847e-04 - categorical_accuracy: 0.9970 - val_loss: 0.0079 - val_categorical_accuracy: 0.9654

Epoch 00044: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 45/65535
839/839 [==============================] - 185s 221ms/step - loss: 6.0037e-04 - categorical_accuracy: 0.9969 - val_loss: 0.0079 - val_categorical_accuracy: 0.9655
Epoch 00045: early stopping
========= generating oof predictions 14:39:02 =========
========= generating test set predictions 14:39:16 =========
========= fitting 3 th model 14:41:58 =========
Epoch 1/65535
840/840 [==============================] - 195s 232ms/step - loss: 0.0977 - categorical_accuracy: 0.3969 - val_loss: 0.2165 - val_categorical_accuracy: 0.0924
Epoch 2/65535
840/840 [==============================] - 187s 223ms/step - loss: 0.0292 - categorical_accuracy: 0.8319 - val_loss: 0.1325 - val_categorical_accuracy: 0.3252
Epoch 3/65535
840/840 [==============================] - 185s 221ms/step - loss: 0.0201 - categorical_accuracy: 0.8868 - val_loss: 0.0709 - val_categorical_accuracy: 0.5665
Epoch 4/65535
840/840 [==============================] - 186s 221ms/step - loss: 0.0157 - categorical_accuracy: 0.9120 - val_loss: 0.1252 - val_categorical_accuracy: 0.3571
Epoch 5/65535
840/840 [==============================] - 188s 223ms/step - loss: 0.0131 - categorical_accuracy: 0.9272 - val_loss: 0.0236 - val_categorical_accuracy: 0.8722
Epoch 6/65535
840/840 [==============================] - 190s 226ms/step - loss: 0.0114 - categorical_accuracy: 0.9366 - val_loss: 0.0233 - val_categorical_accuracy: 0.8743
Epoch 7/65535
840/840 [==============================] - 195s 232ms/step - loss: 0.0099 - categorical_accuracy: 0.9439 - val_loss: 0.0154 - val_categorical_accuracy: 0.9223
Epoch 8/65535
840/840 [==============================] - 186s 222ms/step - loss: 0.0090 - categorical_accuracy: 0.9496 - val_loss: 0.0126 - val_categorical_accuracy: 0.9316
Epoch 9/65535
840/840 [==============================] - 185s 220ms/step - loss: 0.0081 - categorical_accuracy: 0.9547 - val_loss: 0.0175 - val_categorical_accuracy: 0.9107
Epoch 10/65535
840/840 [==============================] - 186s 221ms/step - loss: 0.0074 - categorical_accuracy: 0.9578 - val_loss: 0.0147 - val_categorical_accuracy: 0.9266
Epoch 11/65535
840/840 [==============================] - 185s 220ms/step - loss: 0.0067 - categorical_accuracy: 0.9621 - val_loss: 0.0133 - val_categorical_accuracy: 0.9313
Epoch 12/65535
840/840 [==============================] - 186s 221ms/step - loss: 0.0063 - categorical_accuracy: 0.9641 - val_loss: 0.0548 - val_categorical_accuracy: 0.7167
Epoch 13/65535
840/840 [==============================] - 189s 225ms/step - loss: 0.0059 - categorical_accuracy: 0.9671 - val_loss: 0.0633 - val_categorical_accuracy: 0.6802
Epoch 14/65535
840/840 [==============================] - 187s 223ms/step - loss: 0.0062 - categorical_accuracy: 0.9646 - val_loss: 0.0517 - val_categorical_accuracy: 0.7259

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 15/65535
840/840 [==============================] - 191s 227ms/step - loss: 0.0045 - categorical_accuracy: 0.9752 - val_loss: 0.0089 - val_categorical_accuracy: 0.9573
Epoch 16/65535
840/840 [==============================] - 190s 226ms/step - loss: 0.0033 - categorical_accuracy: 0.9818 - val_loss: 0.0081 - val_categorical_accuracy: 0.9597
Epoch 17/65535
840/840 [==============================] - 185s 221ms/step - loss: 0.0029 - categorical_accuracy: 0.9846 - val_loss: 0.0096 - val_categorical_accuracy: 0.9586
Epoch 18/65535
840/840 [==============================] - 187s 223ms/step - loss: 0.0027 - categorical_accuracy: 0.9854 - val_loss: 0.0090 - val_categorical_accuracy: 0.9608
Epoch 19/65535
840/840 [==============================] - 188s 224ms/step - loss: 0.0025 - categorical_accuracy: 0.9866 - val_loss: 0.0091 - val_categorical_accuracy: 0.9608
Epoch 20/65535
840/840 [==============================] - 191s 227ms/step - loss: 0.0024 - categorical_accuracy: 0.9869 - val_loss: 0.0074 - val_categorical_accuracy: 0.9616
Epoch 21/65535
840/840 [==============================] - 193s 230ms/step - loss: 0.0023 - categorical_accuracy: 0.9876 - val_loss: 0.0091 - val_categorical_accuracy: 0.9609
Epoch 22/65535
840/840 [==============================] - 185s 221ms/step - loss: 0.0021 - categorical_accuracy: 0.9888 - val_loss: 0.0077 - val_categorical_accuracy: 0.9612
Epoch 23/65535
840/840 [==============================] - 186s 222ms/step - loss: 0.0021 - categorical_accuracy: 0.9890 - val_loss: 0.0082 - val_categorical_accuracy: 0.9605
Epoch 24/65535
840/840 [==============================] - 185s 220ms/step - loss: 0.0019 - categorical_accuracy: 0.9901 - val_loss: 0.0089 - val_categorical_accuracy: 0.9609
Epoch 25/65535
840/840 [==============================] - 185s 220ms/step - loss: 0.0018 - categorical_accuracy: 0.9905 - val_loss: 0.0092 - val_categorical_accuracy: 0.9607
Epoch 26/65535
840/840 [==============================] - 186s 222ms/step - loss: 0.0017 - categorical_accuracy: 0.9910 - val_loss: 0.0076 - val_categorical_accuracy: 0.9619

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 27/65535
840/840 [==============================] - 190s 226ms/step - loss: 0.0015 - categorical_accuracy: 0.9921 - val_loss: 0.0076 - val_categorical_accuracy: 0.9632
Epoch 28/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0014 - categorical_accuracy: 0.9931 - val_loss: 0.0079 - val_categorical_accuracy: 0.9635
Epoch 29/65535
840/840 [==============================] - 176s 209ms/step - loss: 0.0014 - categorical_accuracy: 0.9925 - val_loss: 0.0078 - val_categorical_accuracy: 0.9634
Epoch 30/65535
840/840 [==============================] - 176s 209ms/step - loss: 0.0014 - categorical_accuracy: 0.9928 - val_loss: 0.0078 - val_categorical_accuracy: 0.9637
Epoch 31/65535
840/840 [==============================] - 177s 210ms/step - loss: 0.0014 - categorical_accuracy: 0.9932 - val_loss: 0.0079 - val_categorical_accuracy: 0.9638

Epoch 00031: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 32/65535
840/840 [==============================] - 175s 209ms/step - loss: 0.0013 - categorical_accuracy: 0.9933 - val_loss: 0.0077 - val_categorical_accuracy: 0.9634
Epoch 33/65535
840/840 [==============================] - 179s 214ms/step - loss: 0.0014 - categorical_accuracy: 0.9931 - val_loss: 0.0077 - val_categorical_accuracy: 0.9633
Epoch 34/65535
840/840 [==============================] - 181s 215ms/step - loss: 0.0013 - categorical_accuracy: 0.9934 - val_loss: 0.0077 - val_categorical_accuracy: 0.9632
Epoch 35/65535
840/840 [==============================] - 180s 214ms/step - loss: 0.0013 - categorical_accuracy: 0.9936 - val_loss: 0.0077 - val_categorical_accuracy: 0.9635
Epoch 00035: early stopping
========= generating oof predictions 16:30:23 =========
========= generating test set predictions 16:30:36 =========
========= fitting 4 th model 16:33:10 =========
Epoch 1/65535
840/840 [==============================] - 181s 216ms/step - loss: 0.0911 - categorical_accuracy: 0.4339 - val_loss: 0.1066 - val_categorical_accuracy: 0.3966
Epoch 2/65535
840/840 [==============================] - 177s 211ms/step - loss: 0.0281 - categorical_accuracy: 0.8395 - val_loss: 0.0723 - val_categorical_accuracy: 0.5956
Epoch 3/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0188 - categorical_accuracy: 0.8951 - val_loss: 0.0333 - val_categorical_accuracy: 0.8149
Epoch 4/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0146 - categorical_accuracy: 0.9200 - val_loss: 0.0869 - val_categorical_accuracy: 0.6086
Epoch 5/65535
840/840 [==============================] - 176s 210ms/step - loss: 0.0125 - categorical_accuracy: 0.9312 - val_loss: 0.0306 - val_categorical_accuracy: 0.8456
Epoch 6/65535
840/840 [==============================] - 175s 208ms/step - loss: 0.0109 - categorical_accuracy: 0.9396 - val_loss: 0.0158 - val_categorical_accuracy: 0.9211
Epoch 7/65535
840/840 [==============================] - 175s 209ms/step - loss: 0.0094 - categorical_accuracy: 0.9487 - val_loss: 0.0154 - val_categorical_accuracy: 0.9235
Epoch 8/65535
840/840 [==============================] - 180s 214ms/step - loss: 0.0086 - categorical_accuracy: 0.9524 - val_loss: 0.0195 - val_categorical_accuracy: 0.8960
Epoch 9/65535
840/840 [==============================] - 179s 214ms/step - loss: 0.0077 - categorical_accuracy: 0.9577 - val_loss: 0.0146 - val_categorical_accuracy: 0.9292
Epoch 10/65535
840/840 [==============================] - 183s 217ms/step - loss: 0.0071 - categorical_accuracy: 0.9603 - val_loss: 0.0114 - val_categorical_accuracy: 0.9437
Epoch 11/65535
840/840 [==============================] - 180s 214ms/step - loss: 0.0064 - categorical_accuracy: 0.9647 - val_loss: 0.0145 - val_categorical_accuracy: 0.9265
Epoch 12/65535
840/840 [==============================] - 177s 210ms/step - loss: 0.0058 - categorical_accuracy: 0.9676 - val_loss: 0.0122 - val_categorical_accuracy: 0.9381
Epoch 13/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0054 - categorical_accuracy: 0.9700 - val_loss: 0.0141 - val_categorical_accuracy: 0.9316
Epoch 14/65535
840/840 [==============================] - 181s 216ms/step - loss: 0.0052 - categorical_accuracy: 0.9713 - val_loss: 0.0112 - val_categorical_accuracy: 0.9440
Epoch 15/65535
840/840 [==============================] - 181s 215ms/step - loss: 0.0045 - categorical_accuracy: 0.9748 - val_loss: 0.0101 - val_categorical_accuracy: 0.9465
Epoch 16/65535
840/840 [==============================] - 185s 220ms/step - loss: 0.0045 - categorical_accuracy: 0.9746 - val_loss: 0.0228 - val_categorical_accuracy: 0.8907
Epoch 17/65535
840/840 [==============================] - 181s 215ms/step - loss: 0.0041 - categorical_accuracy: 0.9770 - val_loss: 0.0170 - val_categorical_accuracy: 0.9235
Epoch 18/65535
840/840 [==============================] - 175s 208ms/step - loss: 0.0037 - categorical_accuracy: 0.9791 - val_loss: 0.0115 - val_categorical_accuracy: 0.9440
Epoch 19/65535
840/840 [==============================] - 175s 209ms/step - loss: 0.0035 - categorical_accuracy: 0.9805 - val_loss: 0.0166 - val_categorical_accuracy: 0.9255
Epoch 20/65535
840/840 [==============================] - 175s 209ms/step - loss: 0.0035 - categorical_accuracy: 0.9804 - val_loss: 0.0133 - val_categorical_accuracy: 0.9346
Epoch 21/65535
840/840 [==============================] - 176s 209ms/step - loss: 0.0033 - categorical_accuracy: 0.9816 - val_loss: 0.0120 - val_categorical_accuracy: 0.9441

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 22/65535
840/840 [==============================] - 176s 209ms/step - loss: 0.0021 - categorical_accuracy: 0.9883 - val_loss: 0.0075 - val_categorical_accuracy: 0.9640
Epoch 23/65535
840/840 [==============================] - 175s 209ms/step - loss: 0.0017 - categorical_accuracy: 0.9910 - val_loss: 0.0072 - val_categorical_accuracy: 0.9641
Epoch 24/65535
840/840 [==============================] - 177s 210ms/step - loss: 0.0016 - categorical_accuracy: 0.9918 - val_loss: 0.0075 - val_categorical_accuracy: 0.9651
Epoch 25/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0014 - categorical_accuracy: 0.9924 - val_loss: 0.0078 - val_categorical_accuracy: 0.9636
Epoch 26/65535
840/840 [==============================] - 182s 217ms/step - loss: 0.0014 - categorical_accuracy: 0.9929 - val_loss: 0.0076 - val_categorical_accuracy: 0.9649
Epoch 27/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0013 - categorical_accuracy: 0.9934 - val_loss: 0.0076 - val_categorical_accuracy: 0.9647
Epoch 28/65535
840/840 [==============================] - 176s 210ms/step - loss: 0.0012 - categorical_accuracy: 0.9935 - val_loss: 0.0077 - val_categorical_accuracy: 0.9641
Epoch 29/65535
840/840 [==============================] - 177s 211ms/step - loss: 0.0012 - categorical_accuracy: 0.9937 - val_loss: 0.0074 - val_categorical_accuracy: 0.9651

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 30/65535
840/840 [==============================] - 178s 211ms/step - loss: 0.0011 - categorical_accuracy: 0.9940 - val_loss: 0.0068 - val_categorical_accuracy: 0.9668
Epoch 31/65535
840/840 [==============================] - 183s 218ms/step - loss: 0.0011 - categorical_accuracy: 0.9943 - val_loss: 0.0070 - val_categorical_accuracy: 0.9662
Epoch 32/65535
840/840 [==============================] - 185s 221ms/step - loss: 0.0011 - categorical_accuracy: 0.9947 - val_loss: 0.0069 - val_categorical_accuracy: 0.9665
Epoch 33/65535
840/840 [==============================] - 185s 221ms/step - loss: 0.0010 - categorical_accuracy: 0.9945 - val_loss: 0.0068 - val_categorical_accuracy: 0.9665
Epoch 34/65535
840/840 [==============================] - 183s 218ms/step - loss: 0.0010 - categorical_accuracy: 0.9945 - val_loss: 0.0068 - val_categorical_accuracy: 0.9665
Epoch 35/65535
840/840 [==============================] - 180s 214ms/step - loss: 0.0010 - categorical_accuracy: 0.9945 - val_loss: 0.0068 - val_categorical_accuracy: 0.9661
Epoch 36/65535
840/840 [==============================] - 183s 217ms/step - loss: 0.0010 - categorical_accuracy: 0.9949 - val_loss: 0.0068 - val_categorical_accuracy: 0.9674

Epoch 00036: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 37/65535
840/840 [==============================] - 184s 219ms/step - loss: 9.7501e-04 - categorical_accuracy: 0.9948 - val_loss: 0.0068 - val_categorical_accuracy: 0.9672
Epoch 38/65535
840/840 [==============================] - 185s 220ms/step - loss: 9.7986e-04 - categorical_accuracy: 0.9949 - val_loss: 0.0068 - val_categorical_accuracy: 0.9670
Epoch 39/65535
840/840 [==============================] - 177s 211ms/step - loss: 9.6417e-04 - categorical_accuracy: 0.9950 - val_loss: 0.0068 - val_categorical_accuracy: 0.9673
Epoch 40/65535
840/840 [==============================] - 175s 209ms/step - loss: 9.7141e-04 - categorical_accuracy: 0.9950 - val_loss: 0.0068 - val_categorical_accuracy: 0.9670
Epoch 41/65535
840/840 [==============================] - 175s 209ms/step - loss: 9.3480e-04 - categorical_accuracy: 0.9951 - val_loss: 0.0068 - val_categorical_accuracy: 0.9671

Epoch 00041: ReduceLROnPlateau reducing learning rate to 0.0015999998897314072.
Epoch 42/65535
840/840 [==============================] - 175s 208ms/step - loss: 9.3209e-04 - categorical_accuracy: 0.9951 - val_loss: 0.0068 - val_categorical_accuracy: 0.9673
Epoch 43/65535
840/840 [==============================] - 176s 209ms/step - loss: 9.5325e-04 - categorical_accuracy: 0.9950 - val_loss: 0.0068 - val_categorical_accuracy: 0.9673
Epoch 44/65535
840/840 [==============================] - 177s 211ms/step - loss: 9.4895e-04 - categorical_accuracy: 0.9950 - val_loss: 0.0068 - val_categorical_accuracy: 0.9674
Epoch 45/65535
840/840 [==============================] - 185s 221ms/step - loss: 9.3881e-04 - categorical_accuracy: 0.9953 - val_loss: 0.0068 - val_categorical_accuracy: 0.9673
Epoch 46/65535
840/840 [==============================] - 183s 218ms/step - loss: 9.6503e-04 - categorical_accuracy: 0.9950 - val_loss: 0.0068 - val_categorical_accuracy: 0.9672

Epoch 00046: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 47/65535
840/840 [==============================] - 184s 219ms/step - loss: 9.4778e-04 - categorical_accuracy: 0.9950 - val_loss: 0.0068 - val_categorical_accuracy: 0.9674
Epoch 48/65535
840/840 [==============================] - 177s 210ms/step - loss: 9.1355e-04 - categorical_accuracy: 0.9952 - val_loss: 0.0068 - val_categorical_accuracy: 0.9672
Epoch 49/65535
840/840 [==============================] - 183s 217ms/step - loss: 9.3670e-04 - categorical_accuracy: 0.9953 - val_loss: 0.0068 - val_categorical_accuracy: 0.9674
Epoch 50/65535
840/840 [==============================] - 184s 219ms/step - loss: 9.4345e-04 - categorical_accuracy: 0.9953 - val_loss: 0.0068 - val_categorical_accuracy: 0.9674
Epoch 51/65535
840/840 [==============================] - 185s 220ms/step - loss: 9.4093e-04 - categorical_accuracy: 0.9951 - val_loss: 0.0068 - val_categorical_accuracy: 0.9674

Epoch 00051: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 52/65535
840/840 [==============================] - 187s 223ms/step - loss: 9.3316e-04 - categorical_accuracy: 0.9953 - val_loss: 0.0068 - val_categorical_accuracy: 0.9673
Epoch 53/65535
840/840 [==============================] - 188s 224ms/step - loss: 9.4637e-04 - categorical_accuracy: 0.9949 - val_loss: 0.0068 - val_categorical_accuracy: 0.9674
Epoch 54/65535
840/840 [==============================] - 186s 221ms/step - loss: 9.4927e-04 - categorical_accuracy: 0.9952 - val_loss: 0.0068 - val_categorical_accuracy: 0.9674
Epoch 00054: early stopping
========= generating oof predictions 19:15:12 =========
========= generating test set predictions 19:15:25 =========
========= fitting 5 th model 19:18:01 =========
Epoch 1/65535
840/840 [==============================] - 183s 218ms/step - loss: 0.0871 - categorical_accuracy: 0.4661 - val_loss: 0.0926 - val_categorical_accuracy: 0.4724
Epoch 2/65535
840/840 [==============================] - 179s 214ms/step - loss: 0.0270 - categorical_accuracy: 0.8461 - val_loss: 0.0597 - val_categorical_accuracy: 0.7025
Epoch 3/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0188 - categorical_accuracy: 0.8960 - val_loss: 0.0247 - val_categorical_accuracy: 0.8636
Epoch 4/65535
840/840 [==============================] - 180s 214ms/step - loss: 0.0151 - categorical_accuracy: 0.9166 - val_loss: 0.0172 - val_categorical_accuracy: 0.9067
Epoch 5/65535
840/840 [==============================] - 179s 214ms/step - loss: 0.0125 - categorical_accuracy: 0.9306 - val_loss: 0.0214 - val_categorical_accuracy: 0.8865
Epoch 6/65535
840/840 [==============================] - 180s 214ms/step - loss: 0.0109 - categorical_accuracy: 0.9393 - val_loss: 0.0161 - val_categorical_accuracy: 0.9129
Epoch 7/65535
840/840 [==============================] - 179s 214ms/step - loss: 0.0093 - categorical_accuracy: 0.9484 - val_loss: 0.0129 - val_categorical_accuracy: 0.9316
Epoch 8/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0085 - categorical_accuracy: 0.9533 - val_loss: 0.0237 - val_categorical_accuracy: 0.8802
Epoch 9/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0076 - categorical_accuracy: 0.9573 - val_loss: 0.0141 - val_categorical_accuracy: 0.9237
Epoch 10/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0068 - categorical_accuracy: 0.9625 - val_loss: 0.0125 - val_categorical_accuracy: 0.9342
Epoch 11/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0063 - categorical_accuracy: 0.9645 - val_loss: 0.0155 - val_categorical_accuracy: 0.9200
Epoch 12/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0057 - categorical_accuracy: 0.9685 - val_loss: 0.0191 - val_categorical_accuracy: 0.8967
Epoch 13/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0054 - categorical_accuracy: 0.9695 - val_loss: 0.0111 - val_categorical_accuracy: 0.9447
Epoch 14/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0051 - categorical_accuracy: 0.9718 - val_loss: 0.0157 - val_categorical_accuracy: 0.9234
Epoch 15/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0047 - categorical_accuracy: 0.9730 - val_loss: 0.0124 - val_categorical_accuracy: 0.9352
Epoch 16/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0042 - categorical_accuracy: 0.9767 - val_loss: 0.0133 - val_categorical_accuracy: 0.9312
Epoch 17/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0041 - categorical_accuracy: 0.9770 - val_loss: 0.0177 - val_categorical_accuracy: 0.9132
Epoch 18/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0037 - categorical_accuracy: 0.9791 - val_loss: 0.0123 - val_categorical_accuracy: 0.9381
Epoch 19/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0036 - categorical_accuracy: 0.9792 - val_loss: 0.0125 - val_categorical_accuracy: 0.9383

Epoch 00019: ReduceLROnPlateau reducing learning rate to 0.2.
Epoch 20/65535
840/840 [==============================] - 177s 211ms/step - loss: 0.0023 - categorical_accuracy: 0.9874 - val_loss: 0.0079 - val_categorical_accuracy: 0.9599
Epoch 21/65535
840/840 [==============================] - 177s 211ms/step - loss: 0.0019 - categorical_accuracy: 0.9901 - val_loss: 0.0087 - val_categorical_accuracy: 0.9591
Epoch 22/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0017 - categorical_accuracy: 0.9909 - val_loss: 0.0087 - val_categorical_accuracy: 0.9597
Epoch 23/65535
840/840 [==============================] - 179s 213ms/step - loss: 0.0015 - categorical_accuracy: 0.9919 - val_loss: 0.0080 - val_categorical_accuracy: 0.9617
Epoch 24/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0014 - categorical_accuracy: 0.9927 - val_loss: 0.0082 - val_categorical_accuracy: 0.9605
Epoch 25/65535
840/840 [==============================] - 178s 211ms/step - loss: 0.0014 - categorical_accuracy: 0.9930 - val_loss: 0.0082 - val_categorical_accuracy: 0.9612
Epoch 26/65535
840/840 [==============================] - 178s 212ms/step - loss: 0.0013 - categorical_accuracy: 0.9934 - val_loss: 0.0084 - val_categorical_accuracy: 0.9611

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
Epoch 27/65535
840/840 [==============================] - 179s 214ms/step - loss: 0.0012 - categorical_accuracy: 0.9941 - val_loss: 0.0081 - val_categorical_accuracy: 0.9623
Epoch 28/65535
840/840 [==============================] - 180s 214ms/step - loss: 0.0011 - categorical_accuracy: 0.9943 - val_loss: 0.0084 - val_categorical_accuracy: 0.9618
Epoch 29/65535
840/840 [==============================] - 176s 209ms/step - loss: 0.0011 - categorical_accuracy: 0.9943 - val_loss: 0.0083 - val_categorical_accuracy: 0.9620
Epoch 30/65535
840/840 [==============================] - 180s 215ms/step - loss: 0.0011 - categorical_accuracy: 0.9943 - val_loss: 0.0082 - val_categorical_accuracy: 0.9619
Epoch 31/65535
840/840 [==============================] - 177s 211ms/step - loss: 0.0011 - categorical_accuracy: 0.9946 - val_loss: 0.0081 - val_categorical_accuracy: 0.9621

Epoch 00031: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
Epoch 32/65535
840/840 [==============================] - 176s 210ms/step - loss: 0.0011 - categorical_accuracy: 0.9946 - val_loss: 0.0081 - val_categorical_accuracy: 0.9621
Epoch 33/65535
840/840 [==============================] - 177s 211ms/step - loss: 0.0010 - categorical_accuracy: 0.9949 - val_loss: 0.0081 - val_categorical_accuracy: 0.9623
Epoch 34/65535
840/840 [==============================] - 176s 210ms/step - loss: 0.0010 - categorical_accuracy: 0.9949 - val_loss: 0.0082 - val_categorical_accuracy: 0.9623
Epoch 35/65535
840/840 [==============================] - 176s 209ms/step - loss: 0.0011 - categorical_accuracy: 0.9946 - val_loss: 0.0081 - val_categorical_accuracy: 0.9623
Epoch 00035: early stopping
========= generating oof predictions 21:02:17 =========
========= generating test set predictions 21:02:28 =========
train loss avg 0.0009886529902248613 -- std 0.00022657286553277598, val loss avg 0.007638640337419141 -- std 0.00045438474212752724
train acc avg 0.9950835885232163 -- std 0.0010614871925447573, val acc avg 0.9649136439248724 -- std 0.0017802986526498721
mean nb epochs 40.8
dump oof predicted probs
dump test set predicted probs
