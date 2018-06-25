ren (master *+) python $ python train.py mfcc_10_40_20 arm_lstm xl 128
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
cu_dnnlstm_1 (CuDNNLSTM)     (None, 512)               1073152
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 31)                15903
=================================================================
Total params: 1,089,055
Trainable params: 1,089,055
Non-trainable params: 0
_________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 09:52:10 =========
Epoch 1/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.1038 - categorical_accuracy: 0.3664 - val_loss: 0.0661 - val_categorical_accuracy: 0.6562
Epoch 2/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0526 - categorical_accuracy: 0.7095 - val_loss: 0.0390 - val_categorical_accuracy: 0.7855
Epoch 3/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0358 - categorical_accuracy: 0.8034 - val_loss: 0.0296 - val_categorical_accuracy: 0.8370
Epoch 4/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0290 - categorical_accuracy: 0.8393 - val_loss: 0.0252 - val_categorical_accuracy: 0.8599
Epoch 5/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0248 - categorical_accuracy: 0.8621 - val_loss: 0.0237 - val_categorical_accuracy: 0.8663
Epoch 6/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0220 - categorical_accuracy: 0.8788 - val_loss: 0.0208 - val_categorical_accuracy: 0.8854
Epoch 7/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0201 - categorical_accuracy: 0.8894 - val_loss: 0.0196 - val_categorical_accuracy: 0.8902
Epoch 8/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0186 - categorical_accuracy: 0.8959 - val_loss: 0.0192 - val_categorical_accuracy: 0.8905
Epoch 9/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0176 - categorical_accuracy: 0.9024 - val_loss: 0.0185 - val_categorical_accuracy: 0.8957
Epoch 10/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0164 - categorical_accuracy: 0.9090 - val_loss: 0.0171 - val_categorical_accuracy: 0.9041
Epoch 11/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0157 - categorical_accuracy: 0.9135 - val_loss: 0.0166 - val_categorical_accuracy: 0.9071
Epoch 12/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0147 - categorical_accuracy: 0.9196 - val_loss: 0.0170 - val_categorical_accuracy: 0.9030
Epoch 13/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0141 - categorical_accuracy: 0.9222 - val_loss: 0.0163 - val_categorical_accuracy: 0.9097
Epoch 14/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0133 - categorical_accuracy: 0.9257 - val_loss: 0.0154 - val_categorical_accuracy: 0.9136
Epoch 15/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0128 - categorical_accuracy: 0.9294 - val_loss: 0.0153 - val_categorical_accuracy: 0.9141
Epoch 16/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0123 - categorical_accuracy: 0.9323 - val_loss: 0.0147 - val_categorical_accuracy: 0.9173
Epoch 17/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0120 - categorical_accuracy: 0.9329 - val_loss: 0.0149 - val_categorical_accuracy: 0.9169
Epoch 18/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0115 - categorical_accuracy: 0.9355 - val_loss: 0.0141 - val_categorical_accuracy: 0.9196
Epoch 19/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0112 - categorical_accuracy: 0.9378 - val_loss: 0.0150 - val_categorical_accuracy: 0.9163
Epoch 20/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0109 - categorical_accuracy: 0.9401 - val_loss: 0.0141 - val_categorical_accuracy: 0.9237
Epoch 21/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0110 - categorical_accuracy: 0.9390 - val_loss: 0.0139 - val_categorical_accuracy: 0.9223
Epoch 22/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0108 - categorical_accuracy: 0.9406 - val_loss: 0.0134 - val_categorical_accuracy: 0.9248
Epoch 23/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0103 - categorical_accuracy: 0.9433 - val_loss: 0.0137 - val_categorical_accuracy: 0.9249
Epoch 24/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0103 - categorical_accuracy: 0.9436 - val_loss: 0.0142 - val_categorical_accuracy: 0.9219
Epoch 25/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0097 - categorical_accuracy: 0.9458 - val_loss: 0.0137 - val_categorical_accuracy: 0.9250
Epoch 26/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0094 - categorical_accuracy: 0.9479 - val_loss: 0.0135 - val_categorical_accuracy: 0.9257
Epoch 27/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0091 - categorical_accuracy: 0.9499 - val_loss: 0.0136 - val_categorical_accuracy: 0.9273
Epoch 28/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0094 - categorical_accuracy: 0.9479 - val_loss: 0.0138 - val_categorical_accuracy: 0.9250

Epoch 00028: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 29/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0067 - categorical_accuracy: 0.9636 - val_loss: 0.0115 - val_categorical_accuracy: 0.9386
Epoch 30/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0057 - categorical_accuracy: 0.9698 - val_loss: 0.0112 - val_categorical_accuracy: 0.9392
Epoch 31/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0053 - categorical_accuracy: 0.9724 - val_loss: 0.0112 - val_categorical_accuracy: 0.9408
Epoch 32/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0051 - categorical_accuracy: 0.9736 - val_loss: 0.0111 - val_categorical_accuracy: 0.9408
Epoch 33/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0049 - categorical_accuracy: 0.9744 - val_loss: 0.0113 - val_categorical_accuracy: 0.9396
Epoch 34/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0047 - categorical_accuracy: 0.9757 - val_loss: 0.0112 - val_categorical_accuracy: 0.9398
Epoch 35/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0045 - categorical_accuracy: 0.9767 - val_loss: 0.0111 - val_categorical_accuracy: 0.9409
Epoch 36/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0044 - categorical_accuracy: 0.9771 - val_loss: 0.0113 - val_categorical_accuracy: 0.9410
Epoch 37/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0042 - categorical_accuracy: 0.9783 - val_loss: 0.0112 - val_categorical_accuracy: 0.9405
Epoch 38/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0041 - categorical_accuracy: 0.9789 - val_loss: 0.0113 - val_categorical_accuracy: 0.9414

Epoch 00038: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 39/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0037 - categorical_accuracy: 0.9814 - val_loss: 0.0111 - val_categorical_accuracy: 0.9414
Epoch 40/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0036 - categorical_accuracy: 0.9818 - val_loss: 0.0111 - val_categorical_accuracy: 0.9423
Epoch 41/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0035 - categorical_accuracy: 0.9826 - val_loss: 0.0111 - val_categorical_accuracy: 0.9417
Epoch 42/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0035 - categorical_accuracy: 0.9826 - val_loss: 0.0111 - val_categorical_accuracy: 0.9417
Epoch 43/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0034 - categorical_accuracy: 0.9829 - val_loss: 0.0111 - val_categorical_accuracy: 0.9420

Epoch 00043: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 44/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0034 - categorical_accuracy: 0.9840 - val_loss: 0.0111 - val_categorical_accuracy: 0.9422
Epoch 45/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0034 - categorical_accuracy: 0.9834 - val_loss: 0.0111 - val_categorical_accuracy: 0.9422
Epoch 46/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0033 - categorical_accuracy: 0.9837 - val_loss: 0.0111 - val_categorical_accuracy: 0.9420
Epoch 47/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0033 - categorical_accuracy: 0.9834 - val_loss: 0.0111 - val_categorical_accuracy: 0.9420
Epoch 48/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0033 - categorical_accuracy: 0.9834 - val_loss: 0.0111 - val_categorical_accuracy: 0.9420
Epoch 49/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0033 - categorical_accuracy: 0.9839 - val_loss: 0.0111 - val_categorical_accuracy: 0.9422
Epoch 50/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0033 - categorical_accuracy: 0.9836 - val_loss: 0.0112 - val_categorical_accuracy: 0.9418
Epoch 51/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0033 - categorical_accuracy: 0.9838 - val_loss: 0.0111 - val_categorical_accuracy: 0.9420
Epoch 52/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0032 - categorical_accuracy: 0.9841 - val_loss: 0.0111 - val_categorical_accuracy: 0.9416
Epoch 53/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0032 - categorical_accuracy: 0.9838 - val_loss: 0.0111 - val_categorical_accuracy: 0.9417
Epoch 54/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0032 - categorical_accuracy: 0.9843 - val_loss: 0.0112 - val_categorical_accuracy: 0.9413
Epoch 55/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0032 - categorical_accuracy: 0.9839 - val_loss: 0.0111 - val_categorical_accuracy: 0.9419
Epoch 56/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0032 - categorical_accuracy: 0.9845 - val_loss: 0.0112 - val_categorical_accuracy: 0.9417
Epoch 57/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9847 - val_loss: 0.0112 - val_categorical_accuracy: 0.9421
Epoch 58/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9851 - val_loss: 0.0112 - val_categorical_accuracy: 0.9412
Epoch 59/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9850 - val_loss: 0.0112 - val_categorical_accuracy: 0.9416
Epoch 60/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9847 - val_loss: 0.0112 - val_categorical_accuracy: 0.9419
Epoch 00060: early stopping
========= generating oof predictions 09:58:15 =========
========= generating test set predictions 09:58:16 =========
========= fitting 2 th model 09:58:27 =========
Epoch 1/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.1012 - categorical_accuracy: 0.3809 - val_loss: 0.0611 - val_categorical_accuracy: 0.6784
Epoch 2/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0476 - categorical_accuracy: 0.7400 - val_loss: 0.0346 - val_categorical_accuracy: 0.8157
Epoch 3/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0329 - categorical_accuracy: 0.8188 - val_loss: 0.0275 - val_categorical_accuracy: 0.8489
Epoch 4/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0272 - categorical_accuracy: 0.8488 - val_loss: 0.0243 - val_categorical_accuracy: 0.8623
Epoch 5/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0235 - categorical_accuracy: 0.8685 - val_loss: 0.0216 - val_categorical_accuracy: 0.8795
Epoch 6/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0213 - categorical_accuracy: 0.8811 - val_loss: 0.0205 - val_categorical_accuracy: 0.8833
Epoch 7/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0194 - categorical_accuracy: 0.8915 - val_loss: 0.0190 - val_categorical_accuracy: 0.8920
Epoch 8/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0176 - categorical_accuracy: 0.9020 - val_loss: 0.0183 - val_categorical_accuracy: 0.8979
Epoch 9/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0170 - categorical_accuracy: 0.9067 - val_loss: 0.0179 - val_categorical_accuracy: 0.8982
Epoch 10/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0158 - categorical_accuracy: 0.9112 - val_loss: 0.0164 - val_categorical_accuracy: 0.9089
Epoch 11/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0148 - categorical_accuracy: 0.9177 - val_loss: 0.0164 - val_categorical_accuracy: 0.9092
Epoch 12/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0140 - categorical_accuracy: 0.9227 - val_loss: 0.0158 - val_categorical_accuracy: 0.9110
Epoch 13/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0136 - categorical_accuracy: 0.9245 - val_loss: 0.0156 - val_categorical_accuracy: 0.9133
Epoch 14/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0131 - categorical_accuracy: 0.9270 - val_loss: 0.0149 - val_categorical_accuracy: 0.9153
Epoch 15/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0125 - categorical_accuracy: 0.9315 - val_loss: 0.0157 - val_categorical_accuracy: 0.9101
Epoch 16/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0121 - categorical_accuracy: 0.9329 - val_loss: 0.0149 - val_categorical_accuracy: 0.9164
Epoch 17/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0117 - categorical_accuracy: 0.9358 - val_loss: 0.0151 - val_categorical_accuracy: 0.9164
Epoch 18/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0115 - categorical_accuracy: 0.9362 - val_loss: 0.0149 - val_categorical_accuracy: 0.9165
Epoch 19/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0107 - categorical_accuracy: 0.9411 - val_loss: 0.0141 - val_categorical_accuracy: 0.9229
Epoch 20/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0107 - categorical_accuracy: 0.9411 - val_loss: 0.0139 - val_categorical_accuracy: 0.9212
Epoch 21/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0103 - categorical_accuracy: 0.9430 - val_loss: 0.0148 - val_categorical_accuracy: 0.9188
Epoch 22/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0102 - categorical_accuracy: 0.9437 - val_loss: 0.0148 - val_categorical_accuracy: 0.9162
Epoch 23/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0098 - categorical_accuracy: 0.9458 - val_loss: 0.0140 - val_categorical_accuracy: 0.9241
Epoch 24/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0097 - categorical_accuracy: 0.9454 - val_loss: 0.0142 - val_categorical_accuracy: 0.9211
Epoch 25/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0090 - categorical_accuracy: 0.9506 - val_loss: 0.0139 - val_categorical_accuracy: 0.9249
Epoch 26/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0090 - categorical_accuracy: 0.9494 - val_loss: 0.0133 - val_categorical_accuracy: 0.9271
Epoch 27/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0085 - categorical_accuracy: 0.9534 - val_loss: 0.0133 - val_categorical_accuracy: 0.9273
Epoch 28/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0086 - categorical_accuracy: 0.9523 - val_loss: 0.0134 - val_categorical_accuracy: 0.9282
Epoch 29/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0088 - categorical_accuracy: 0.9517 - val_loss: 0.0135 - val_categorical_accuracy: 0.9256
Epoch 30/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0084 - categorical_accuracy: 0.9533 - val_loss: 0.0134 - val_categorical_accuracy: 0.9264
Epoch 31/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0084 - categorical_accuracy: 0.9540 - val_loss: 0.0139 - val_categorical_accuracy: 0.9244
Epoch 32/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0083 - categorical_accuracy: 0.9539 - val_loss: 0.0133 - val_categorical_accuracy: 0.9301

Epoch 00032: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 33/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0058 - categorical_accuracy: 0.9681 - val_loss: 0.0123 - val_categorical_accuracy: 0.9347
Epoch 34/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0050 - categorical_accuracy: 0.9731 - val_loss: 0.0120 - val_categorical_accuracy: 0.9343
Epoch 35/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0047 - categorical_accuracy: 0.9747 - val_loss: 0.0119 - val_categorical_accuracy: 0.9353
Epoch 36/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0045 - categorical_accuracy: 0.9766 - val_loss: 0.0120 - val_categorical_accuracy: 0.9372
Epoch 37/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0043 - categorical_accuracy: 0.9770 - val_loss: 0.0119 - val_categorical_accuracy: 0.9368
Epoch 38/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0041 - categorical_accuracy: 0.9788 - val_loss: 0.0120 - val_categorical_accuracy: 0.9379
Epoch 39/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0040 - categorical_accuracy: 0.9795 - val_loss: 0.0120 - val_categorical_accuracy: 0.9360
Epoch 40/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0038 - categorical_accuracy: 0.9806 - val_loss: 0.0120 - val_categorical_accuracy: 0.9371

Epoch 00040: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 41/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0119 - val_categorical_accuracy: 0.9384
Epoch 42/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0033 - categorical_accuracy: 0.9834 - val_loss: 0.0119 - val_categorical_accuracy: 0.9379
Epoch 43/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0033 - categorical_accuracy: 0.9836 - val_loss: 0.0118 - val_categorical_accuracy: 0.9384
Epoch 44/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0032 - categorical_accuracy: 0.9839 - val_loss: 0.0119 - val_categorical_accuracy: 0.9387
Epoch 45/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0032 - categorical_accuracy: 0.9839 - val_loss: 0.0119 - val_categorical_accuracy: 0.9379
Epoch 46/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0032 - categorical_accuracy: 0.9840 - val_loss: 0.0119 - val_categorical_accuracy: 0.9379
Epoch 47/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0032 - categorical_accuracy: 0.9842 - val_loss: 0.0119 - val_categorical_accuracy: 0.9383

Epoch 00047: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 48/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0031 - categorical_accuracy: 0.9850 - val_loss: 0.0119 - val_categorical_accuracy: 0.9387
Epoch 49/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0031 - categorical_accuracy: 0.9848 - val_loss: 0.0119 - val_categorical_accuracy: 0.9383
Epoch 50/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9847 - val_loss: 0.0119 - val_categorical_accuracy: 0.9381
Epoch 51/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9850 - val_loss: 0.0119 - val_categorical_accuracy: 0.9378
Epoch 52/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9852 - val_loss: 0.0120 - val_categorical_accuracy: 0.9381
Epoch 53/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9850 - val_loss: 0.0119 - val_categorical_accuracy: 0.9382
Epoch 54/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0030 - categorical_accuracy: 0.9853 - val_loss: 0.0119 - val_categorical_accuracy: 0.9379
Epoch 55/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9852 - val_loss: 0.0119 - val_categorical_accuracy: 0.9386
Epoch 56/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9853 - val_loss: 0.0119 - val_categorical_accuracy: 0.9384
Epoch 57/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0029 - categorical_accuracy: 0.9857 - val_loss: 0.0120 - val_categorical_accuracy: 0.9381
Epoch 58/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0029 - categorical_accuracy: 0.9857 - val_loss: 0.0119 - val_categorical_accuracy: 0.9381
Epoch 00058: early stopping
========= generating oof predictions 10:04:23 =========
========= generating test set predictions 10:04:23 =========
========= fitting 3 th model 10:04:36 =========
Epoch 1/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0987 - categorical_accuracy: 0.4012 - val_loss: 0.0586 - val_categorical_accuracy: 0.6720
Epoch 2/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0460 - categorical_accuracy: 0.7462 - val_loss: 0.0356 - val_categorical_accuracy: 0.8050
Epoch 3/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0322 - categorical_accuracy: 0.8225 - val_loss: 0.0273 - val_categorical_accuracy: 0.8493
Epoch 4/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0262 - categorical_accuracy: 0.8553 - val_loss: 0.0237 - val_categorical_accuracy: 0.8684
Epoch 5/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0228 - categorical_accuracy: 0.8751 - val_loss: 0.0225 - val_categorical_accuracy: 0.8738
Epoch 6/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0206 - categorical_accuracy: 0.8854 - val_loss: 0.0209 - val_categorical_accuracy: 0.8818
Epoch 7/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0190 - categorical_accuracy: 0.8947 - val_loss: 0.0189 - val_categorical_accuracy: 0.8949
Epoch 8/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0175 - categorical_accuracy: 0.9031 - val_loss: 0.0181 - val_categorical_accuracy: 0.8990
Epoch 9/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0162 - categorical_accuracy: 0.9096 - val_loss: 0.0172 - val_categorical_accuracy: 0.9034
Epoch 10/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0155 - categorical_accuracy: 0.9152 - val_loss: 0.0163 - val_categorical_accuracy: 0.9094
Epoch 11/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0144 - categorical_accuracy: 0.9202 - val_loss: 0.0163 - val_categorical_accuracy: 0.9087
Epoch 12/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0140 - categorical_accuracy: 0.9227 - val_loss: 0.0159 - val_categorical_accuracy: 0.9103
Epoch 13/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0137 - categorical_accuracy: 0.9237 - val_loss: 0.0149 - val_categorical_accuracy: 0.9178
Epoch 14/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0129 - categorical_accuracy: 0.9279 - val_loss: 0.0149 - val_categorical_accuracy: 0.9139
Epoch 15/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0126 - categorical_accuracy: 0.9299 - val_loss: 0.0150 - val_categorical_accuracy: 0.9136
Epoch 16/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0120 - categorical_accuracy: 0.9338 - val_loss: 0.0144 - val_categorical_accuracy: 0.9199
Epoch 17/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0113 - categorical_accuracy: 0.9366 - val_loss: 0.0139 - val_categorical_accuracy: 0.9241
Epoch 18/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0112 - categorical_accuracy: 0.9379 - val_loss: 0.0148 - val_categorical_accuracy: 0.9177
Epoch 19/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0106 - categorical_accuracy: 0.9409 - val_loss: 0.0140 - val_categorical_accuracy: 0.9209
Epoch 20/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0102 - categorical_accuracy: 0.9434 - val_loss: 0.0134 - val_categorical_accuracy: 0.9271
Epoch 21/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0100 - categorical_accuracy: 0.9439 - val_loss: 0.0136 - val_categorical_accuracy: 0.9234
Epoch 22/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0099 - categorical_accuracy: 0.9451 - val_loss: 0.0144 - val_categorical_accuracy: 0.9200
Epoch 23/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0095 - categorical_accuracy: 0.9473 - val_loss: 0.0140 - val_categorical_accuracy: 0.9227
Epoch 24/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0092 - categorical_accuracy: 0.9489 - val_loss: 0.0136 - val_categorical_accuracy: 0.9245
Epoch 25/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0090 - categorical_accuracy: 0.9500 - val_loss: 0.0140 - val_categorical_accuracy: 0.9256
Epoch 26/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0089 - categorical_accuracy: 0.9503 - val_loss: 0.0131 - val_categorical_accuracy: 0.9286
Epoch 27/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0087 - categorical_accuracy: 0.9513 - val_loss: 0.0133 - val_categorical_accuracy: 0.9255
Epoch 28/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0085 - categorical_accuracy: 0.9523 - val_loss: 0.0143 - val_categorical_accuracy: 0.9241
Epoch 29/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0082 - categorical_accuracy: 0.9546 - val_loss: 0.0136 - val_categorical_accuracy: 0.9234
Epoch 30/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0082 - categorical_accuracy: 0.9546 - val_loss: 0.0133 - val_categorical_accuracy: 0.9268
Epoch 31/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0078 - categorical_accuracy: 0.9568 - val_loss: 0.0133 - val_categorical_accuracy: 0.9275
Epoch 32/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0079 - categorical_accuracy: 0.9563 - val_loss: 0.0133 - val_categorical_accuracy: 0.9297

Epoch 00032: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 33/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0059 - categorical_accuracy: 0.9677 - val_loss: 0.0117 - val_categorical_accuracy: 0.9385
Epoch 34/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0049 - categorical_accuracy: 0.9739 - val_loss: 0.0115 - val_categorical_accuracy: 0.9405
Epoch 35/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0045 - categorical_accuracy: 0.9757 - val_loss: 0.0114 - val_categorical_accuracy: 0.9414
Epoch 36/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0042 - categorical_accuracy: 0.9779 - val_loss: 0.0113 - val_categorical_accuracy: 0.9407
Epoch 37/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0040 - categorical_accuracy: 0.9791 - val_loss: 0.0112 - val_categorical_accuracy: 0.9406
Epoch 38/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0040 - categorical_accuracy: 0.9791 - val_loss: 0.0113 - val_categorical_accuracy: 0.9416
Epoch 39/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0038 - categorical_accuracy: 0.9805 - val_loss: 0.0113 - val_categorical_accuracy: 0.9407
Epoch 40/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0036 - categorical_accuracy: 0.9814 - val_loss: 0.0114 - val_categorical_accuracy: 0.9401
Epoch 41/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0035 - categorical_accuracy: 0.9819 - val_loss: 0.0113 - val_categorical_accuracy: 0.9414
Epoch 42/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0035 - categorical_accuracy: 0.9823 - val_loss: 0.0114 - val_categorical_accuracy: 0.9401
Epoch 43/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0033 - categorical_accuracy: 0.9831 - val_loss: 0.0112 - val_categorical_accuracy: 0.9409

Epoch 00043: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 44/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0030 - categorical_accuracy: 0.9850 - val_loss: 0.0113 - val_categorical_accuracy: 0.9417
Epoch 45/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0030 - categorical_accuracy: 0.9848 - val_loss: 0.0112 - val_categorical_accuracy: 0.9422
Epoch 46/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0029 - categorical_accuracy: 0.9854 - val_loss: 0.0113 - val_categorical_accuracy: 0.9420
Epoch 47/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0029 - categorical_accuracy: 0.9855 - val_loss: 0.0113 - val_categorical_accuracy: 0.9420
Epoch 48/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0029 - categorical_accuracy: 0.9856 - val_loss: 0.0113 - val_categorical_accuracy: 0.9423

Epoch 00048: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 49/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9864 - val_loss: 0.0114 - val_categorical_accuracy: 0.9417
Epoch 50/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0027 - categorical_accuracy: 0.9867 - val_loss: 0.0113 - val_categorical_accuracy: 0.9417
Epoch 51/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0027 - categorical_accuracy: 0.9860 - val_loss: 0.0114 - val_categorical_accuracy: 0.9421
Epoch 52/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0027 - categorical_accuracy: 0.9863 - val_loss: 0.0114 - val_categorical_accuracy: 0.9420
Epoch 53/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9866 - val_loss: 0.0114 - val_categorical_accuracy: 0.9421
Epoch 54/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9866 - val_loss: 0.0114 - val_categorical_accuracy: 0.9424
Epoch 55/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0027 - categorical_accuracy: 0.9865 - val_loss: 0.0114 - val_categorical_accuracy: 0.9423
Epoch 56/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0027 - categorical_accuracy: 0.9865 - val_loss: 0.0114 - val_categorical_accuracy: 0.9424
Epoch 57/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0027 - categorical_accuracy: 0.9865 - val_loss: 0.0114 - val_categorical_accuracy: 0.9424
Epoch 58/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0026 - categorical_accuracy: 0.9871 - val_loss: 0.0114 - val_categorical_accuracy: 0.9419
Epoch 00058: early stopping
========= generating oof predictions 10:10:31 =========
========= generating test set predictions 10:10:32 =========
========= fitting 4 th model 10:10:42 =========
Epoch 1/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0987 - categorical_accuracy: 0.4048 - val_loss: 0.0592 - val_categorical_accuracy: 0.6828
Epoch 2/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0467 - categorical_accuracy: 0.7427 - val_loss: 0.0347 - val_categorical_accuracy: 0.8144
Epoch 3/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0323 - categorical_accuracy: 0.8223 - val_loss: 0.0276 - val_categorical_accuracy: 0.8478
Epoch 4/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0264 - categorical_accuracy: 0.8541 - val_loss: 0.0236 - val_categorical_accuracy: 0.8714
Epoch 5/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0230 - categorical_accuracy: 0.8728 - val_loss: 0.0211 - val_categorical_accuracy: 0.8824
Epoch 6/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0205 - categorical_accuracy: 0.8875 - val_loss: 0.0198 - val_categorical_accuracy: 0.8877
Epoch 7/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0186 - categorical_accuracy: 0.8962 - val_loss: 0.0190 - val_categorical_accuracy: 0.8928
Epoch 8/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0174 - categorical_accuracy: 0.9032 - val_loss: 0.0177 - val_categorical_accuracy: 0.9008
Epoch 9/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0163 - categorical_accuracy: 0.9097 - val_loss: 0.0170 - val_categorical_accuracy: 0.9033
Epoch 10/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0151 - categorical_accuracy: 0.9174 - val_loss: 0.0168 - val_categorical_accuracy: 0.9059
Epoch 11/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0144 - categorical_accuracy: 0.9206 - val_loss: 0.0160 - val_categorical_accuracy: 0.9085
Epoch 12/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0136 - categorical_accuracy: 0.9247 - val_loss: 0.0154 - val_categorical_accuracy: 0.9127
Epoch 13/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0129 - categorical_accuracy: 0.9294 - val_loss: 0.0146 - val_categorical_accuracy: 0.9147
Epoch 14/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0123 - categorical_accuracy: 0.9325 - val_loss: 0.0149 - val_categorical_accuracy: 0.9158
Epoch 15/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0119 - categorical_accuracy: 0.9338 - val_loss: 0.0147 - val_categorical_accuracy: 0.9170
Epoch 16/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0113 - categorical_accuracy: 0.9382 - val_loss: 0.0143 - val_categorical_accuracy: 0.9185
Epoch 17/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0112 - categorical_accuracy: 0.9382 - val_loss: 0.0141 - val_categorical_accuracy: 0.9214
Epoch 18/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0108 - categorical_accuracy: 0.9404 - val_loss: 0.0142 - val_categorical_accuracy: 0.9185
Epoch 19/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0104 - categorical_accuracy: 0.9426 - val_loss: 0.0136 - val_categorical_accuracy: 0.9226
Epoch 20/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0099 - categorical_accuracy: 0.9458 - val_loss: 0.0132 - val_categorical_accuracy: 0.9280
Epoch 21/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0096 - categorical_accuracy: 0.9463 - val_loss: 0.0139 - val_categorical_accuracy: 0.9221
Epoch 22/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0093 - categorical_accuracy: 0.9482 - val_loss: 0.0138 - val_categorical_accuracy: 0.9223
Epoch 23/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0091 - categorical_accuracy: 0.9493 - val_loss: 0.0131 - val_categorical_accuracy: 0.9274
Epoch 24/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0089 - categorical_accuracy: 0.9504 - val_loss: 0.0136 - val_categorical_accuracy: 0.9236
Epoch 25/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0087 - categorical_accuracy: 0.9519 - val_loss: 0.0140 - val_categorical_accuracy: 0.9230
Epoch 26/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0088 - categorical_accuracy: 0.9516 - val_loss: 0.0135 - val_categorical_accuracy: 0.9249
Epoch 27/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0085 - categorical_accuracy: 0.9537 - val_loss: 0.0134 - val_categorical_accuracy: 0.9254
Epoch 28/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0087 - categorical_accuracy: 0.9518 - val_loss: 0.0142 - val_categorical_accuracy: 0.9204
Epoch 29/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0082 - categorical_accuracy: 0.9547 - val_loss: 0.0137 - val_categorical_accuracy: 0.9241

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 30/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0061 - categorical_accuracy: 0.9659 - val_loss: 0.0119 - val_categorical_accuracy: 0.9356
Epoch 31/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0052 - categorical_accuracy: 0.9723 - val_loss: 0.0116 - val_categorical_accuracy: 0.9378
Epoch 32/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0048 - categorical_accuracy: 0.9742 - val_loss: 0.0115 - val_categorical_accuracy: 0.9376
Epoch 33/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0046 - categorical_accuracy: 0.9761 - val_loss: 0.0114 - val_categorical_accuracy: 0.9372
Epoch 34/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0044 - categorical_accuracy: 0.9772 - val_loss: 0.0116 - val_categorical_accuracy: 0.9375
Epoch 35/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0042 - categorical_accuracy: 0.9783 - val_loss: 0.0116 - val_categorical_accuracy: 0.9393
Epoch 36/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0040 - categorical_accuracy: 0.9788 - val_loss: 0.0114 - val_categorical_accuracy: 0.9389
Epoch 37/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0039 - categorical_accuracy: 0.9792 - val_loss: 0.0115 - val_categorical_accuracy: 0.9391
Epoch 38/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0038 - categorical_accuracy: 0.9810 - val_loss: 0.0114 - val_categorical_accuracy: 0.9393
Epoch 39/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0037 - categorical_accuracy: 0.9808 - val_loss: 0.0115 - val_categorical_accuracy: 0.9399

Epoch 00039: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 40/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0035 - categorical_accuracy: 0.9823 - val_loss: 0.0111 - val_categorical_accuracy: 0.9404
Epoch 41/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0033 - categorical_accuracy: 0.9834 - val_loss: 0.0111 - val_categorical_accuracy: 0.9407
Epoch 42/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0032 - categorical_accuracy: 0.9839 - val_loss: 0.0111 - val_categorical_accuracy: 0.9405
Epoch 43/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9845 - val_loss: 0.0111 - val_categorical_accuracy: 0.9407
Epoch 44/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9846 - val_loss: 0.0112 - val_categorical_accuracy: 0.9407
Epoch 45/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0031 - categorical_accuracy: 0.9848 - val_loss: 0.0111 - val_categorical_accuracy: 0.9414
Epoch 46/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9846 - val_loss: 0.0112 - val_categorical_accuracy: 0.9415

Epoch 00046: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 47/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0030 - categorical_accuracy: 0.9848 - val_loss: 0.0111 - val_categorical_accuracy: 0.9412
Epoch 48/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0030 - categorical_accuracy: 0.9852 - val_loss: 0.0112 - val_categorical_accuracy: 0.9416
Epoch 49/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9850 - val_loss: 0.0111 - val_categorical_accuracy: 0.9410
Epoch 50/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9852 - val_loss: 0.0112 - val_categorical_accuracy: 0.9413
Epoch 51/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0029 - categorical_accuracy: 0.9852 - val_loss: 0.0112 - val_categorical_accuracy: 0.9413
Epoch 52/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0029 - categorical_accuracy: 0.9854 - val_loss: 0.0112 - val_categorical_accuracy: 0.9409
Epoch 53/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0029 - categorical_accuracy: 0.9853 - val_loss: 0.0112 - val_categorical_accuracy: 0.9409
Epoch 54/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0029 - categorical_accuracy: 0.9853 - val_loss: 0.0112 - val_categorical_accuracy: 0.9411
Epoch 55/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0029 - categorical_accuracy: 0.9856 - val_loss: 0.0112 - val_categorical_accuracy: 0.9410
Epoch 56/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0029 - categorical_accuracy: 0.9855 - val_loss: 0.0112 - val_categorical_accuracy: 0.9413
Epoch 00056: early stopping
========= generating oof predictions 10:16:25 =========
========= generating test set predictions 10:16:26 =========
========= fitting 5 th model 10:16:36 =========
Epoch 1/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.1011 - categorical_accuracy: 0.3837 - val_loss: 0.0619 - val_categorical_accuracy: 0.6820
Epoch 2/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0497 - categorical_accuracy: 0.7270 - val_loss: 0.0367 - val_categorical_accuracy: 0.8023
Epoch 3/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0339 - categorical_accuracy: 0.8139 - val_loss: 0.0284 - val_categorical_accuracy: 0.8420
Epoch 4/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0275 - categorical_accuracy: 0.8467 - val_loss: 0.0245 - val_categorical_accuracy: 0.8639
Epoch 5/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0236 - categorical_accuracy: 0.8689 - val_loss: 0.0219 - val_categorical_accuracy: 0.8755
Epoch 6/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0209 - categorical_accuracy: 0.8841 - val_loss: 0.0205 - val_categorical_accuracy: 0.8855
Epoch 7/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0191 - categorical_accuracy: 0.8938 - val_loss: 0.0185 - val_categorical_accuracy: 0.8942
Epoch 8/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0178 - categorical_accuracy: 0.9000 - val_loss: 0.0183 - val_categorical_accuracy: 0.8973
Epoch 9/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0165 - categorical_accuracy: 0.9077 - val_loss: 0.0173 - val_categorical_accuracy: 0.9017
Epoch 10/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0157 - categorical_accuracy: 0.9113 - val_loss: 0.0172 - val_categorical_accuracy: 0.9023
Epoch 11/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0146 - categorical_accuracy: 0.9197 - val_loss: 0.0165 - val_categorical_accuracy: 0.9069
Epoch 12/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0139 - categorical_accuracy: 0.9223 - val_loss: 0.0157 - val_categorical_accuracy: 0.9105
Epoch 13/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0132 - categorical_accuracy: 0.9269 - val_loss: 0.0160 - val_categorical_accuracy: 0.9105
Epoch 14/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0128 - categorical_accuracy: 0.9279 - val_loss: 0.0154 - val_categorical_accuracy: 0.9134
Epoch 15/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0124 - categorical_accuracy: 0.9313 - val_loss: 0.0165 - val_categorical_accuracy: 0.9069
Epoch 16/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0120 - categorical_accuracy: 0.9336 - val_loss: 0.0146 - val_categorical_accuracy: 0.9175
Epoch 17/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0115 - categorical_accuracy: 0.9351 - val_loss: 0.0161 - val_categorical_accuracy: 0.9091
Epoch 18/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0113 - categorical_accuracy: 0.9367 - val_loss: 0.0149 - val_categorical_accuracy: 0.9164
Epoch 19/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0109 - categorical_accuracy: 0.9391 - val_loss: 0.0143 - val_categorical_accuracy: 0.9196
Epoch 20/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0107 - categorical_accuracy: 0.9403 - val_loss: 0.0145 - val_categorical_accuracy: 0.9190
Epoch 21/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0103 - categorical_accuracy: 0.9419 - val_loss: 0.0140 - val_categorical_accuracy: 0.9225
Epoch 22/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0101 - categorical_accuracy: 0.9435 - val_loss: 0.0142 - val_categorical_accuracy: 0.9232
Epoch 23/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0096 - categorical_accuracy: 0.9475 - val_loss: 0.0138 - val_categorical_accuracy: 0.9238
Epoch 24/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0093 - categorical_accuracy: 0.9474 - val_loss: 0.0146 - val_categorical_accuracy: 0.9205
Epoch 25/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0089 - categorical_accuracy: 0.9504 - val_loss: 0.0138 - val_categorical_accuracy: 0.9233
Epoch 26/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0090 - categorical_accuracy: 0.9503 - val_loss: 0.0141 - val_categorical_accuracy: 0.9214
Epoch 27/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0089 - categorical_accuracy: 0.9506 - val_loss: 0.0140 - val_categorical_accuracy: 0.9234
Epoch 28/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0087 - categorical_accuracy: 0.9509 - val_loss: 0.0133 - val_categorical_accuracy: 0.9297
Epoch 29/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0082 - categorical_accuracy: 0.9551 - val_loss: 0.0137 - val_categorical_accuracy: 0.9254
Epoch 30/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0081 - categorical_accuracy: 0.9543 - val_loss: 0.0145 - val_categorical_accuracy: 0.9222
Epoch 31/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0083 - categorical_accuracy: 0.9537 - val_loss: 0.0136 - val_categorical_accuracy: 0.9249
Epoch 32/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0085 - categorical_accuracy: 0.9525 - val_loss: 0.0141 - val_categorical_accuracy: 0.9238
Epoch 33/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0077 - categorical_accuracy: 0.9563 - val_loss: 0.0138 - val_categorical_accuracy: 0.9251
Epoch 34/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0076 - categorical_accuracy: 0.9580 - val_loss: 0.0141 - val_categorical_accuracy: 0.9231

Epoch 00034: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 35/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0057 - categorical_accuracy: 0.9682 - val_loss: 0.0120 - val_categorical_accuracy: 0.9375
Epoch 36/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0048 - categorical_accuracy: 0.9734 - val_loss: 0.0119 - val_categorical_accuracy: 0.9371
Epoch 37/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0045 - categorical_accuracy: 0.9761 - val_loss: 0.0119 - val_categorical_accuracy: 0.9368
Epoch 38/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0043 - categorical_accuracy: 0.9774 - val_loss: 0.0120 - val_categorical_accuracy: 0.9385
Epoch 39/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0041 - categorical_accuracy: 0.9782 - val_loss: 0.0119 - val_categorical_accuracy: 0.9383
Epoch 40/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0040 - categorical_accuracy: 0.9788 - val_loss: 0.0119 - val_categorical_accuracy: 0.9383
Epoch 41/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0037 - categorical_accuracy: 0.9803 - val_loss: 0.0119 - val_categorical_accuracy: 0.9377
Epoch 42/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0037 - categorical_accuracy: 0.9809 - val_loss: 0.0120 - val_categorical_accuracy: 0.9369
Epoch 43/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0036 - categorical_accuracy: 0.9814 - val_loss: 0.0120 - val_categorical_accuracy: 0.9371

Epoch 00043: ReduceLROnPlateau reducing learning rate to 2.0000000949949027e-05.
Epoch 44/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0033 - categorical_accuracy: 0.9833 - val_loss: 0.0116 - val_categorical_accuracy: 0.9394
Epoch 45/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0031 - categorical_accuracy: 0.9839 - val_loss: 0.0116 - val_categorical_accuracy: 0.9386
Epoch 46/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9841 - val_loss: 0.0117 - val_categorical_accuracy: 0.9385
Epoch 47/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0031 - categorical_accuracy: 0.9846 - val_loss: 0.0116 - val_categorical_accuracy: 0.9390
Epoch 48/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9846 - val_loss: 0.0116 - val_categorical_accuracy: 0.9397
Epoch 49/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9854 - val_loss: 0.0116 - val_categorical_accuracy: 0.9395
Epoch 50/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0030 - categorical_accuracy: 0.9850 - val_loss: 0.0117 - val_categorical_accuracy: 0.9386

Epoch 00050: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 51/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0029 - categorical_accuracy: 0.9853 - val_loss: 0.0116 - val_categorical_accuracy: 0.9395
Epoch 52/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0029 - categorical_accuracy: 0.9853 - val_loss: 0.0116 - val_categorical_accuracy: 0.9401
Epoch 53/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0029 - categorical_accuracy: 0.9857 - val_loss: 0.0116 - val_categorical_accuracy: 0.9397
Epoch 54/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0028 - categorical_accuracy: 0.9857 - val_loss: 0.0116 - val_categorical_accuracy: 0.9389
Epoch 55/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0028 - categorical_accuracy: 0.9854 - val_loss: 0.0116 - val_categorical_accuracy: 0.9398
Epoch 56/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0028 - categorical_accuracy: 0.9857 - val_loss: 0.0116 - val_categorical_accuracy: 0.9392
Epoch 57/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0028 - categorical_accuracy: 0.9860 - val_loss: 0.0117 - val_categorical_accuracy: 0.9390
Epoch 58/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0028 - categorical_accuracy: 0.9859 - val_loss: 0.0116 - val_categorical_accuracy: 0.9394
Epoch 59/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0028 - categorical_accuracy: 0.9861 - val_loss: 0.0117 - val_categorical_accuracy: 0.9395
Epoch 60/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0028 - categorical_accuracy: 0.9863 - val_loss: 0.0117 - val_categorical_accuracy: 0.9390
Epoch 61/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0028 - categorical_accuracy: 0.9864 - val_loss: 0.0117 - val_categorical_accuracy: 0.9390
Epoch 62/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9863 - val_loss: 0.0116 - val_categorical_accuracy: 0.9398
Epoch 63/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0028 - categorical_accuracy: 0.9858 - val_loss: 0.0117 - val_categorical_accuracy: 0.9391
Epoch 64/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0027 - categorical_accuracy: 0.9863 - val_loss: 0.0116 - val_categorical_accuracy: 0.9394
Epoch 65/65535
420/420 [==============================] - 6s 15ms/step - loss: 0.0027 - categorical_accuracy: 0.9858 - val_loss: 0.0117 - val_categorical_accuracy: 0.9389
Epoch 66/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9860 - val_loss: 0.0117 - val_categorical_accuracy: 0.9398
Epoch 67/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9866 - val_loss: 0.0117 - val_categorical_accuracy: 0.9394
Epoch 68/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0026 - categorical_accuracy: 0.9868 - val_loss: 0.0117 - val_categorical_accuracy: 0.9396
Epoch 69/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0027 - categorical_accuracy: 0.9865 - val_loss: 0.0117 - val_categorical_accuracy: 0.9394
Epoch 70/65535
420/420 [==============================] - 6s 14ms/step - loss: 0.0026 - categorical_accuracy: 0.9869 - val_loss: 0.0117 - val_categorical_accuracy: 0.9392
Epoch 00070: early stopping
========= generating oof predictions 10:23:47 =========
========= generating test set predictions 10:23:47 =========
train loss avg 0.002829490300639265 -- std 0.00018711926535396183, val loss avg 0.011486335261519764 -- std 0.0003055965428227593
train acc avg 0.9859506452631187 -- std 0.0008822056921025328, val acc avg 0.9404805543484726 -- std 0.0015233489396083608
mean nb epochs 60.4
dump oof predicted probs
dump test set predicted probs
