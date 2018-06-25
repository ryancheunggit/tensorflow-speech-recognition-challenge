ren (master *+) python $ python train.py logspectrogram_25_18.75 join t 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (55, 201) ===========
------------- SUMMARY OF MODEL -------------
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 55, 201)      0
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 55, 201, 1)   0           input_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 55, 201, 32)  544         reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 55, 201, 32)  128         conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 55, 201, 32)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 55, 201, 32)  0           activation_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 55, 201, 32)  16416       dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 55, 201, 32)  128         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 55, 201, 32)  0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 55, 201, 32)  0           activation_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 27, 100, 32)  0           dropout_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 27, 100, 64)  32832       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 27, 100, 64)  256         conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 27, 100, 64)  0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 27, 100, 64)  0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 27, 100, 64)  65600       dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 27, 100, 64)  256         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 27, 100, 64)  0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 27, 100, 64)  0           activation_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 13, 50, 64)   0           dropout_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 13, 50, 128)  131200      max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 13, 50, 128)  512         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 13, 50, 128)  0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 13, 50, 128)  0           activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 13, 50, 128)  262272      dropout_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 13, 50, 128)  512         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 13, 50, 128)  0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 13, 50, 128)  0           activation_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 6, 25, 128)   0           dropout_6[0][0]
__________________________________________________________________________________________________
cu_dnngru_1 (CuDNNGRU)          (None, 256)          352512      input_1[0][0]
__________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalM (None, 128)          0           max_pooling2d_3[0][0]
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 128)          0           max_pooling2d_3[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 256)          0           cu_dnngru_1[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 512)          0           global_max_pooling2d_1[0][0]
                                                                 global_average_pooling2d_1[0][0]
                                                                 dropout_7[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 512)          262656      concatenate_1[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 512)          2048        dense_1[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 512)          0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 512)          262656      dropout_8[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 512)          2048        dense_2[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 512)          0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 31)           15903       dropout_9[0][0]
==================================================================================================
Total params: 1,408,479
Trainable params: 1,405,535
Non-trainable params: 2,944
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 06:04:21 =========
Epoch 1/65535
420/420 [==============================] - 99s 235ms/step - loss: 0.1850 - categorical_accuracy: 0.0341 - val_loss: 0.2925 - val_categorical_accuracy: 0.0320
Epoch 2/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.1831 - categorical_accuracy: 0.0328 - val_loss: 0.2902 - val_categorical_accuracy: 0.0313
Epoch 3/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.1818 - categorical_accuracy: 0.0328 - val_loss: 0.2884 - val_categorical_accuracy: 0.0313
Epoch 4/65535
356/420 [========================>.....] - ETA: 14s - loss: 0.1803 - categorical_accuracy: 0.0341^CTraceback (most recent call last):
  File "train.py", line 131, in <module>
    main(dataset, model_arch, model_size, batch_size)
  File "train.py", line 66, in main
    verbose = 1
  File "/home/ren/anaconda3/lib/python3.6/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/keras/engine/training.py", line 2244, in fit_generator
    class_weight=class_weight)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/keras/engine/training.py", line 1890, in train_on_batch
    outputs = self.train_function(ins)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py", line 2475, in __call__
    **self.session_kwargs)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 889, in run
    run_metadata_ptr)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1120, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1317, in _do_run
    options, run_metadata)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1323, in _do_call
    return fn(*args)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1302, in _run_fn
    status, run_metadata)
KeyboardInterrupt
ren (master *+) python $ python train.py logspectrogram_25_18.75 join t 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (55, 201) ===========
------------- SUMMARY OF MODEL -------------
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 55, 201)      0
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 55, 201, 1)   0           input_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 55, 201, 32)  544         reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 55, 201, 32)  128         conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 55, 201, 32)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 55, 201, 32)  0           activation_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 55, 201, 32)  16416       dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 55, 201, 32)  128         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 55, 201, 32)  0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 55, 201, 32)  0           activation_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 27, 100, 32)  0           dropout_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 27, 100, 64)  32832       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 27, 100, 64)  256         conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 27, 100, 64)  0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 27, 100, 64)  0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 27, 100, 64)  65600       dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 27, 100, 64)  256         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 27, 100, 64)  0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 27, 100, 64)  0           activation_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 13, 50, 64)   0           dropout_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 13, 50, 128)  131200      max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 13, 50, 128)  512         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 13, 50, 128)  0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 13, 50, 128)  0           activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 13, 50, 128)  262272      dropout_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 13, 50, 128)  512         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 13, 50, 128)  0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 13, 50, 128)  0           activation_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 6, 25, 128)   0           dropout_6[0][0]
__________________________________________________________________________________________________
cu_dnngru_1 (CuDNNGRU)          (None, 256)          352512      input_1[0][0]
__________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalM (None, 128)          0           max_pooling2d_3[0][0]
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 128)          0           max_pooling2d_3[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 256)          0           cu_dnngru_1[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 512)          0           global_max_pooling2d_1[0][0]
                                                                 global_average_pooling2d_1[0][0]
                                                                 dropout_7[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 512)          262656      concatenate_1[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 512)          2048        dense_1[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 512)          0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 512)          262656      dropout_8[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 512)          2048        dense_2[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 512)          0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 31)           15903       dropout_9[0][0]
==================================================================================================
Total params: 1,408,479
Trainable params: 1,405,535
Non-trainable params: 2,944
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 06:11:07 =========
Epoch 1/65535
420/420 [==============================] - 100s 239ms/step - loss: 0.1819 - categorical_accuracy: 0.0356 - val_loss: 0.2936 - val_categorical_accuracy: 0.0309
Epoch 2/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.1764 - categorical_accuracy: 0.0427 - val_loss: 0.2996 - val_categorical_accuracy: 0.0308
Epoch 3/65535
420/420 [==============================] - 98s 233ms/step - loss: 0.1734 - categorical_accuracy: 0.0503 - val_loss: 0.2799 - val_categorical_accuracy: 0.0315
Epoch 4/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.1713 - categorical_accuracy: 0.0527 - val_loss: 0.2859 - val_categorical_accuracy: 0.0317
Epoch 5/65535
420/420 [==============================] - 98s 233ms/step - loss: 0.1697 - categorical_accuracy: 0.0547 - val_loss: 0.2887 - val_categorical_accuracy: 0.0316
Epoch 6/65535
 89/420 [=====>........................] - ETA: 1:13 - loss: 0.1682 - categorical_accuracy: 0.0585^CTraceback (most recent call last):
  File "train.py", line 131, in <module>
    main(dataset, model_arch, model_size, batch_size)
  File "train.py", line 66, in main
    verbose = 1
  File "/home/ren/anaconda3/lib/python3.6/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/keras/engine/training.py", line 2244, in fit_generator
    class_weight=class_weight)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/keras/engine/training.py", line 1890, in train_on_batch
    outputs = self.train_function(ins)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py", line 2475, in __call__
    **self.session_kwargs)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 889, in run
    run_metadata_ptr)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1120, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1317, in _do_run
    options, run_metadata)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1323, in _do_call
    return fn(*args)
  File "/home/ren/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1302, in _run_fn
    status, run_metadata)
KeyboardInterrupt
ren (master *+) python $ python train.py logspectrogram_25_18.75 join t 128
/home/ren/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
======= loading data =======
========== input shape is : (55, 201) ===========
------------- SUMMARY OF MODEL -------------
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 55, 201)      0
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 55, 201, 1)   0           input_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 55, 201, 32)  544         reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 55, 201, 32)  128         conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 55, 201, 32)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 55, 201, 32)  0           activation_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 55, 201, 32)  16416       dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 55, 201, 32)  128         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 55, 201, 32)  0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 55, 201, 32)  0           activation_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 27, 100, 32)  0           dropout_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 27, 100, 64)  32832       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 27, 100, 64)  256         conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 27, 100, 64)  0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 27, 100, 64)  0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 27, 100, 64)  65600       dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 27, 100, 64)  256         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 27, 100, 64)  0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 27, 100, 64)  0           activation_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 13, 50, 64)   0           dropout_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 13, 50, 128)  131200      max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 13, 50, 128)  512         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 13, 50, 128)  0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 13, 50, 128)  0           activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 13, 50, 128)  262272      dropout_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 13, 50, 128)  512         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 13, 50, 128)  0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 13, 50, 128)  0           activation_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 6, 25, 128)   0           dropout_6[0][0]
__________________________________________________________________________________________________
cu_dnngru_1 (CuDNNGRU)          (None, 256)          352512      input_1[0][0]
__________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalM (None, 128)          0           max_pooling2d_3[0][0]
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 128)          0           max_pooling2d_3[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 256)          0           cu_dnngru_1[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 512)          0           global_max_pooling2d_1[0][0]
                                                                 global_average_pooling2d_1[0][0]
                                                                 dropout_7[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 512)          262656      concatenate_1[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 512)          2048        dense_1[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 512)          0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 512)          262656      dropout_8[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 512)          2048        dense_2[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 512)          0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 31)           15903       dropout_9[0][0]
==================================================================================================
Total params: 1,408,479
Trainable params: 1,405,535
Non-trainable params: 2,944
__________________________________________________________________________________________________
None
--------------------------------------------
========= fitting 1 th model 06:20:26 =========
Epoch 1/65535
420/420 [==============================] - 99s 235ms/step - loss: 0.1016 - categorical_accuracy: 0.3578 - val_loss: 0.1357 - val_categorical_accuracy: 0.2850
Epoch 2/65535
420/420 [==============================] - 97s 232ms/step - loss: 0.0319 - categorical_accuracy: 0.8107 - val_loss: 0.0645 - val_categorical_accuracy: 0.5847
Epoch 3/65535
420/420 [==============================] - 98s 232ms/step - loss: 0.0178 - categorical_accuracy: 0.8980 - val_loss: 0.0342 - val_categorical_accuracy: 0.8238
Epoch 4/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0138 - categorical_accuracy: 0.9224 - val_loss: 0.0350 - val_categorical_accuracy: 0.7991
Epoch 5/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0116 - categorical_accuracy: 0.9340 - val_loss: 0.0201 - val_categorical_accuracy: 0.8937
Epoch 6/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0105 - categorical_accuracy: 0.9410 - val_loss: 0.0203 - val_categorical_accuracy: 0.8972
Epoch 7/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0098 - categorical_accuracy: 0.9437 - val_loss: 0.0264 - val_categorical_accuracy: 0.8567
Epoch 8/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0090 - categorical_accuracy: 0.9494 - val_loss: 0.0297 - val_categorical_accuracy: 0.8434
Epoch 9/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0087 - categorical_accuracy: 0.9502 - val_loss: 0.0350 - val_categorical_accuracy: 0.8063
Epoch 10/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0080 - categorical_accuracy: 0.9543 - val_loss: 0.0561 - val_categorical_accuracy: 0.7092
Epoch 11/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0077 - categorical_accuracy: 0.9571 - val_loss: 0.1086 - val_categorical_accuracy: 0.5141

Epoch 00011: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
Epoch 12/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0057 - categorical_accuracy: 0.9678 - val_loss: 0.0270 - val_categorical_accuracy: 0.8609
Epoch 13/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0048 - categorical_accuracy: 0.9727 - val_loss: 0.0250 - val_categorical_accuracy: 0.8693
Epoch 14/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0046 - categorical_accuracy: 0.9737 - val_loss: 0.0264 - val_categorical_accuracy: 0.8698
Epoch 15/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0043 - categorical_accuracy: 0.9753 - val_loss: 0.0267 - val_categorical_accuracy: 0.8706
Epoch 16/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0042 - categorical_accuracy: 0.9757 - val_loss: 0.0269 - val_categorical_accuracy: 0.8676

Epoch 00016: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
Epoch 17/65535
420/420 [==============================] - 97s 232ms/step - loss: 0.0038 - categorical_accuracy: 0.9785 - val_loss: 0.0195 - val_categorical_accuracy: 0.9041
Epoch 18/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0035 - categorical_accuracy: 0.9793 - val_loss: 0.0176 - val_categorical_accuracy: 0.9131
Epoch 19/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0035 - categorical_accuracy: 0.9797 - val_loss: 0.0172 - val_categorical_accuracy: 0.9166
Epoch 20/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0035 - categorical_accuracy: 0.9796 - val_loss: 0.0177 - val_categorical_accuracy: 0.9151
Epoch 21/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0034 - categorical_accuracy: 0.9804 - val_loss: 0.0176 - val_categorical_accuracy: 0.9166
Epoch 22/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0033 - categorical_accuracy: 0.9815 - val_loss: 0.0178 - val_categorical_accuracy: 0.9158
Epoch 23/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0033 - categorical_accuracy: 0.9810 - val_loss: 0.0176 - val_categorical_accuracy: 0.9164
Epoch 24/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0033 - categorical_accuracy: 0.9812 - val_loss: 0.0187 - val_categorical_accuracy: 0.9114
Epoch 25/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0031 - categorical_accuracy: 0.9825 - val_loss: 0.0180 - val_categorical_accuracy: 0.9149

Epoch 00025: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 26/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0030 - categorical_accuracy: 0.9827 - val_loss: 0.0180 - val_categorical_accuracy: 0.9149
Epoch 27/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0031 - categorical_accuracy: 0.9822 - val_loss: 0.0175 - val_categorical_accuracy: 0.9175
Epoch 28/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0030 - categorical_accuracy: 0.9825 - val_loss: 0.0173 - val_categorical_accuracy: 0.9187
Epoch 29/65535
420/420 [==============================] - 97s 232ms/step - loss: 0.0030 - categorical_accuracy: 0.9827 - val_loss: 0.0172 - val_categorical_accuracy: 0.9187
Epoch 30/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0030 - categorical_accuracy: 0.9826 - val_loss: 0.0168 - val_categorical_accuracy: 0.9210
Epoch 31/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0030 - categorical_accuracy: 0.9824 - val_loss: 0.0166 - val_categorical_accuracy: 0.9213
Epoch 32/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0030 - categorical_accuracy: 0.9829 - val_loss: 0.0163 - val_categorical_accuracy: 0.9234
Epoch 33/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0029 - categorical_accuracy: 0.9832 - val_loss: 0.0168 - val_categorical_accuracy: 0.9216
Epoch 34/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0029 - categorical_accuracy: 0.9831 - val_loss: 0.0163 - val_categorical_accuracy: 0.9238
Epoch 35/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0029 - categorical_accuracy: 0.9835 - val_loss: 0.0161 - val_categorical_accuracy: 0.9246
Epoch 36/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0029 - categorical_accuracy: 0.9836 - val_loss: 0.0167 - val_categorical_accuracy: 0.9215
Epoch 37/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0029 - categorical_accuracy: 0.9837 - val_loss: 0.0168 - val_categorical_accuracy: 0.9224
Epoch 38/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0029 - categorical_accuracy: 0.9833 - val_loss: 0.0165 - val_categorical_accuracy: 0.9235
Epoch 39/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0029 - categorical_accuracy: 0.9834 - val_loss: 0.0160 - val_categorical_accuracy: 0.9254
Epoch 40/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0029 - categorical_accuracy: 0.9833 - val_loss: 0.0163 - val_categorical_accuracy: 0.9244
Epoch 41/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0028 - categorical_accuracy: 0.9836 - val_loss: 0.0166 - val_categorical_accuracy: 0.9224
Epoch 42/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0028 - categorical_accuracy: 0.9837 - val_loss: 0.0166 - val_categorical_accuracy: 0.9230
Epoch 43/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0028 - categorical_accuracy: 0.9842 - val_loss: 0.0169 - val_categorical_accuracy: 0.9216
Epoch 44/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0028 - categorical_accuracy: 0.9837 - val_loss: 0.0167 - val_categorical_accuracy: 0.9213
Epoch 45/65535
420/420 [==============================] - 96s 228ms/step - loss: 0.0028 - categorical_accuracy: 0.9832 - val_loss: 0.0164 - val_categorical_accuracy: 0.9236
Epoch 46/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0028 - categorical_accuracy: 0.9840 - val_loss: 0.0160 - val_categorical_accuracy: 0.9243
Epoch 47/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0028 - categorical_accuracy: 0.9837 - val_loss: 0.0160 - val_categorical_accuracy: 0.9253
Epoch 48/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0028 - categorical_accuracy: 0.9838 - val_loss: 0.0161 - val_categorical_accuracy: 0.9248
Epoch 49/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0028 - categorical_accuracy: 0.9841 - val_loss: 0.0160 - val_categorical_accuracy: 0.9248
Epoch 50/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0028 - categorical_accuracy: 0.9838 - val_loss: 0.0159 - val_categorical_accuracy: 0.9248
Epoch 51/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0027 - categorical_accuracy: 0.9841 - val_loss: 0.0157 - val_categorical_accuracy: 0.9254
Epoch 52/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0028 - categorical_accuracy: 0.9839 - val_loss: 0.0164 - val_categorical_accuracy: 0.9235
Epoch 53/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0027 - categorical_accuracy: 0.9848 - val_loss: 0.0162 - val_categorical_accuracy: 0.9240
Epoch 54/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0027 - categorical_accuracy: 0.9846 - val_loss: 0.0166 - val_categorical_accuracy: 0.9233
Epoch 55/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0027 - categorical_accuracy: 0.9847 - val_loss: 0.0168 - val_categorical_accuracy: 0.9217
Epoch 56/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0027 - categorical_accuracy: 0.9849 - val_loss: 0.0165 - val_categorical_accuracy: 0.9229
Epoch 57/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0026 - categorical_accuracy: 0.9851 - val_loss: 0.0159 - val_categorical_accuracy: 0.9259
Epoch 58/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0026 - categorical_accuracy: 0.9851 - val_loss: 0.0159 - val_categorical_accuracy: 0.9257
Epoch 59/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0026 - categorical_accuracy: 0.9857 - val_loss: 0.0159 - val_categorical_accuracy: 0.9251
Epoch 60/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0026 - categorical_accuracy: 0.9853 - val_loss: 0.0159 - val_categorical_accuracy: 0.9261
Epoch 61/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0026 - categorical_accuracy: 0.9850 - val_loss: 0.0158 - val_categorical_accuracy: 0.9274
Epoch 62/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0026 - categorical_accuracy: 0.9851 - val_loss: 0.0156 - val_categorical_accuracy: 0.9277
Epoch 63/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0025 - categorical_accuracy: 0.9853 - val_loss: 0.0158 - val_categorical_accuracy: 0.9273
Epoch 64/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0027 - categorical_accuracy: 0.9847 - val_loss: 0.0159 - val_categorical_accuracy: 0.9268
Epoch 65/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0026 - categorical_accuracy: 0.9856 - val_loss: 0.0166 - val_categorical_accuracy: 0.9232
Epoch 66/65535
420/420 [==============================] - 98s 233ms/step - loss: 0.0026 - categorical_accuracy: 0.9849 - val_loss: 0.0162 - val_categorical_accuracy: 0.9254
Epoch 67/65535
420/420 [==============================] - 99s 235ms/step - loss: 0.0026 - categorical_accuracy: 0.9855 - val_loss: 0.0165 - val_categorical_accuracy: 0.9236
Epoch 68/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0026 - categorical_accuracy: 0.9850 - val_loss: 0.0159 - val_categorical_accuracy: 0.9268
Epoch 69/65535
420/420 [==============================] - 99s 236ms/step - loss: 0.0026 - categorical_accuracy: 0.9853 - val_loss: 0.0163 - val_categorical_accuracy: 0.9260
Epoch 70/65535
420/420 [==============================] - 99s 235ms/step - loss: 0.0025 - categorical_accuracy: 0.9859 - val_loss: 0.0163 - val_categorical_accuracy: 0.9259
Epoch 71/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0026 - categorical_accuracy: 0.9850 - val_loss: 0.0164 - val_categorical_accuracy: 0.9258
Epoch 72/65535
420/420 [==============================] - 99s 235ms/step - loss: 0.0025 - categorical_accuracy: 0.9860 - val_loss: 0.0162 - val_categorical_accuracy: 0.9262
Epoch 73/65535
420/420 [==============================] - 99s 236ms/step - loss: 0.0025 - categorical_accuracy: 0.9855 - val_loss: 0.0158 - val_categorical_accuracy: 0.9265
Epoch 74/65535
420/420 [==============================] - 99s 235ms/step - loss: 0.0025 - categorical_accuracy: 0.9858 - val_loss: 0.0162 - val_categorical_accuracy: 0.9260
Epoch 75/65535
420/420 [==============================] - 99s 236ms/step - loss: 0.0025 - categorical_accuracy: 0.9856 - val_loss: 0.0161 - val_categorical_accuracy: 0.9260
Epoch 76/65535
420/420 [==============================] - 99s 236ms/step - loss: 0.0025 - categorical_accuracy: 0.9856 - val_loss: 0.0166 - val_categorical_accuracy: 0.9239
Epoch 77/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0025 - categorical_accuracy: 0.9857 - val_loss: 0.0160 - val_categorical_accuracy: 0.9260
Epoch 00077: early stopping
========= generating oof predictions 08:25:00 =========
========= generating test set predictions 08:25:07 =========
========= fitting 2 th model 08:26:28 =========
Epoch 1/65535
420/420 [==============================] - 99s 236ms/step - loss: 0.0836 - categorical_accuracy: 0.4837 - val_loss: 0.0568 - val_categorical_accuracy: 0.6665
Epoch 2/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0216 - categorical_accuracy: 0.8759 - val_loss: 0.0375 - val_categorical_accuracy: 0.7796
Epoch 3/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0151 - categorical_accuracy: 0.9145 - val_loss: 0.0653 - val_categorical_accuracy: 0.6133
Epoch 4/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0127 - categorical_accuracy: 0.9291 - val_loss: 0.0236 - val_categorical_accuracy: 0.8773
Epoch 5/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0115 - categorical_accuracy: 0.9354 - val_loss: 0.0223 - val_categorical_accuracy: 0.8871
Epoch 6/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0101 - categorical_accuracy: 0.9438 - val_loss: 0.0163 - val_categorical_accuracy: 0.9226
Epoch 7/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0094 - categorical_accuracy: 0.9471 - val_loss: 0.0211 - val_categorical_accuracy: 0.8990
Epoch 8/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0090 - categorical_accuracy: 0.9495 - val_loss: 0.0239 - val_categorical_accuracy: 0.8767
Epoch 9/65535
420/420 [==============================] - 98s 233ms/step - loss: 0.0085 - categorical_accuracy: 0.9523 - val_loss: 0.0195 - val_categorical_accuracy: 0.9005
Epoch 10/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0081 - categorical_accuracy: 0.9542 - val_loss: 0.0204 - val_categorical_accuracy: 0.8898
Epoch 11/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0077 - categorical_accuracy: 0.9564 - val_loss: 0.0187 - val_categorical_accuracy: 0.9003
Epoch 12/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0072 - categorical_accuracy: 0.9585 - val_loss: 0.0146 - val_categorical_accuracy: 0.9263
Epoch 13/65535
420/420 [==============================] - 98s 233ms/step - loss: 0.0070 - categorical_accuracy: 0.9594 - val_loss: 0.0233 - val_categorical_accuracy: 0.8774
Epoch 14/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0068 - categorical_accuracy: 0.9618 - val_loss: 0.0283 - val_categorical_accuracy: 0.8584
Epoch 15/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0064 - categorical_accuracy: 0.9645 - val_loss: 0.0260 - val_categorical_accuracy: 0.8685
Epoch 16/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0063 - categorical_accuracy: 0.9649 - val_loss: 0.0988 - val_categorical_accuracy: 0.5740
Epoch 17/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0061 - categorical_accuracy: 0.9656 - val_loss: 0.0233 - val_categorical_accuracy: 0.8757
Epoch 18/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0061 - categorical_accuracy: 0.9652 - val_loss: 0.0284 - val_categorical_accuracy: 0.8611

Epoch 00018: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
Epoch 19/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0044 - categorical_accuracy: 0.9745 - val_loss: 0.0185 - val_categorical_accuracy: 0.9097
Epoch 20/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0038 - categorical_accuracy: 0.9773 - val_loss: 0.0171 - val_categorical_accuracy: 0.9223
Epoch 21/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0037 - categorical_accuracy: 0.9793 - val_loss: 0.0196 - val_categorical_accuracy: 0.9112
Epoch 22/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0035 - categorical_accuracy: 0.9797 - val_loss: 0.0179 - val_categorical_accuracy: 0.9194
Epoch 23/65535
420/420 [==============================] - 98s 234ms/step - loss: 0.0034 - categorical_accuracy: 0.9801 - val_loss: 0.0196 - val_categorical_accuracy: 0.9113

Epoch 00023: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
Epoch 24/65535
420/420 [==============================] - 99s 236ms/step - loss: 0.0031 - categorical_accuracy: 0.9820 - val_loss: 0.0162 - val_categorical_accuracy: 0.9271
Epoch 25/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0029 - categorical_accuracy: 0.9827 - val_loss: 0.0162 - val_categorical_accuracy: 0.9280
Epoch 26/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0029 - categorical_accuracy: 0.9830 - val_loss: 0.0158 - val_categorical_accuracy: 0.9289
Epoch 27/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0029 - categorical_accuracy: 0.9828 - val_loss: 0.0157 - val_categorical_accuracy: 0.9296
Epoch 00027: early stopping
========= generating oof predictions 09:10:37 =========
========= generating test set predictions 09:10:43 =========
========= fitting 3 th model 09:12:02 =========
Epoch 1/65535
420/420 [==============================] - 97s 232ms/step - loss: 0.0898 - categorical_accuracy: 0.4410 - val_loss: 0.0653 - val_categorical_accuracy: 0.5814
Epoch 2/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0230 - categorical_accuracy: 0.8676 - val_loss: 0.0355 - val_categorical_accuracy: 0.8044
Epoch 3/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0151 - categorical_accuracy: 0.9141 - val_loss: 0.0583 - val_categorical_accuracy: 0.6661
Epoch 4/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0124 - categorical_accuracy: 0.9291 - val_loss: 0.0241 - val_categorical_accuracy: 0.8776
Epoch 5/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0108 - categorical_accuracy: 0.9396 - val_loss: 0.0439 - val_categorical_accuracy: 0.7488
Epoch 6/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0101 - categorical_accuracy: 0.9424 - val_loss: 0.0340 - val_categorical_accuracy: 0.8144
Epoch 7/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0092 - categorical_accuracy: 0.9470 - val_loss: 0.0260 - val_categorical_accuracy: 0.8712
Epoch 8/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0085 - categorical_accuracy: 0.9522 - val_loss: 0.0559 - val_categorical_accuracy: 0.6913
Epoch 9/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0086 - categorical_accuracy: 0.9508 - val_loss: 0.0271 - val_categorical_accuracy: 0.8544
Epoch 10/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0079 - categorical_accuracy: 0.9550 - val_loss: 0.0552 - val_categorical_accuracy: 0.7083

Epoch 00010: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
Epoch 11/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0056 - categorical_accuracy: 0.9677 - val_loss: 0.0361 - val_categorical_accuracy: 0.8070
Epoch 12/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0050 - categorical_accuracy: 0.9715 - val_loss: 0.0323 - val_categorical_accuracy: 0.8355
Epoch 13/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0047 - categorical_accuracy: 0.9727 - val_loss: 0.0382 - val_categorical_accuracy: 0.8038
Epoch 14/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0045 - categorical_accuracy: 0.9740 - val_loss: 0.0316 - val_categorical_accuracy: 0.8426
Epoch 15/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0044 - categorical_accuracy: 0.9752 - val_loss: 0.0265 - val_categorical_accuracy: 0.8715

Epoch 00015: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
Epoch 16/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0038 - categorical_accuracy: 0.9782 - val_loss: 0.0223 - val_categorical_accuracy: 0.8895
Epoch 17/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0037 - categorical_accuracy: 0.9788 - val_loss: 0.0236 - val_categorical_accuracy: 0.8847
Epoch 18/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0036 - categorical_accuracy: 0.9793 - val_loss: 0.0227 - val_categorical_accuracy: 0.8879
Epoch 19/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0035 - categorical_accuracy: 0.9797 - val_loss: 0.0222 - val_categorical_accuracy: 0.8932
Epoch 20/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0035 - categorical_accuracy: 0.9801 - val_loss: 0.0216 - val_categorical_accuracy: 0.8949
Epoch 21/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0035 - categorical_accuracy: 0.9796 - val_loss: 0.0221 - val_categorical_accuracy: 0.8917
Epoch 22/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0034 - categorical_accuracy: 0.9809 - val_loss: 0.0211 - val_categorical_accuracy: 0.8967
Epoch 23/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0033 - categorical_accuracy: 0.9805 - val_loss: 0.0223 - val_categorical_accuracy: 0.8920
Epoch 24/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0033 - categorical_accuracy: 0.9810 - val_loss: 0.0226 - val_categorical_accuracy: 0.8904
Epoch 25/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0032 - categorical_accuracy: 0.9818 - val_loss: 0.0196 - val_categorical_accuracy: 0.9042
Epoch 26/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0032 - categorical_accuracy: 0.9816 - val_loss: 0.0183 - val_categorical_accuracy: 0.9133
Epoch 27/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0031 - categorical_accuracy: 0.9820 - val_loss: 0.0190 - val_categorical_accuracy: 0.9093
Epoch 28/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0030 - categorical_accuracy: 0.9825 - val_loss: 0.0200 - val_categorical_accuracy: 0.9053
Epoch 29/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0030 - categorical_accuracy: 0.9823 - val_loss: 0.0220 - val_categorical_accuracy: 0.8970
Epoch 30/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0030 - categorical_accuracy: 0.9830 - val_loss: 0.0212 - val_categorical_accuracy: 0.9019
Epoch 31/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0030 - categorical_accuracy: 0.9829 - val_loss: 0.0206 - val_categorical_accuracy: 0.9017
Epoch 32/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0029 - categorical_accuracy: 0.9826 - val_loss: 0.0224 - val_categorical_accuracy: 0.8962

Epoch 00032: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 33/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0028 - categorical_accuracy: 0.9842 - val_loss: 0.0209 - val_categorical_accuracy: 0.9039
Epoch 34/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0028 - categorical_accuracy: 0.9837 - val_loss: 0.0210 - val_categorical_accuracy: 0.9049
Epoch 35/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0027 - categorical_accuracy: 0.9848 - val_loss: 0.0211 - val_categorical_accuracy: 0.9038
Epoch 36/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0028 - categorical_accuracy: 0.9836 - val_loss: 0.0208 - val_categorical_accuracy: 0.9059
Epoch 37/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0028 - categorical_accuracy: 0.9841 - val_loss: 0.0205 - val_categorical_accuracy: 0.9072
Epoch 38/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0028 - categorical_accuracy: 0.9839 - val_loss: 0.0201 - val_categorical_accuracy: 0.9092
Epoch 39/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0027 - categorical_accuracy: 0.9845 - val_loss: 0.0205 - val_categorical_accuracy: 0.9069
Epoch 40/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0027 - categorical_accuracy: 0.9846 - val_loss: 0.0208 - val_categorical_accuracy: 0.9062
Epoch 41/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0027 - categorical_accuracy: 0.9842 - val_loss: 0.0206 - val_categorical_accuracy: 0.9074
Epoch 00041: early stopping
========= generating oof predictions 10:17:55 =========
========= generating test set predictions 10:18:01 =========
========= fitting 4 th model 10:19:21 =========
Epoch 1/65535
420/420 [==============================] - 97s 232ms/step - loss: 0.0760 - categorical_accuracy: 0.5328 - val_loss: 0.0753 - val_categorical_accuracy: 0.5341
Epoch 2/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0189 - categorical_accuracy: 0.8942 - val_loss: 0.0371 - val_categorical_accuracy: 0.7997
Epoch 3/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0141 - categorical_accuracy: 0.9192 - val_loss: 0.0377 - val_categorical_accuracy: 0.8009
Epoch 4/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0118 - categorical_accuracy: 0.9336 - val_loss: 0.0581 - val_categorical_accuracy: 0.6787
Epoch 5/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0107 - categorical_accuracy: 0.9402 - val_loss: 0.0599 - val_categorical_accuracy: 0.6754
Epoch 6/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0095 - categorical_accuracy: 0.9454 - val_loss: 0.0449 - val_categorical_accuracy: 0.7411
Epoch 7/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0092 - categorical_accuracy: 0.9475 - val_loss: 0.0446 - val_categorical_accuracy: 0.7403
Epoch 8/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0086 - categorical_accuracy: 0.9513 - val_loss: 0.0379 - val_categorical_accuracy: 0.7945

Epoch 00008: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
Epoch 9/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0062 - categorical_accuracy: 0.9644 - val_loss: 0.0595 - val_categorical_accuracy: 0.6754
Epoch 10/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0055 - categorical_accuracy: 0.9693 - val_loss: 0.0345 - val_categorical_accuracy: 0.8187
Epoch 11/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0052 - categorical_accuracy: 0.9708 - val_loss: 0.0352 - val_categorical_accuracy: 0.8202
Epoch 12/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0048 - categorical_accuracy: 0.9727 - val_loss: 0.0308 - val_categorical_accuracy: 0.8431
Epoch 13/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0046 - categorical_accuracy: 0.9738 - val_loss: 0.0352 - val_categorical_accuracy: 0.8196
Epoch 14/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0044 - categorical_accuracy: 0.9745 - val_loss: 0.0387 - val_categorical_accuracy: 0.8072
Epoch 15/65535
420/420 [==============================] - 97s 230ms/step - loss: 0.0043 - categorical_accuracy: 0.9756 - val_loss: 0.0370 - val_categorical_accuracy: 0.8285
Epoch 16/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0040 - categorical_accuracy: 0.9772 - val_loss: 0.0364 - val_categorical_accuracy: 0.8239
Epoch 17/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0039 - categorical_accuracy: 0.9782 - val_loss: 0.0260 - val_categorical_accuracy: 0.8697
Epoch 18/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0037 - categorical_accuracy: 0.9782 - val_loss: 0.0379 - val_categorical_accuracy: 0.8172
Epoch 19/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0037 - categorical_accuracy: 0.9786 - val_loss: 0.0178 - val_categorical_accuracy: 0.9141
Epoch 20/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0035 - categorical_accuracy: 0.9792 - val_loss: 0.0331 - val_categorical_accuracy: 0.8439
Epoch 21/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0034 - categorical_accuracy: 0.9802 - val_loss: 0.0298 - val_categorical_accuracy: 0.8545
Epoch 22/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0033 - categorical_accuracy: 0.9801 - val_loss: 0.0298 - val_categorical_accuracy: 0.8524
Epoch 23/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0032 - categorical_accuracy: 0.9813 - val_loss: 0.0314 - val_categorical_accuracy: 0.8471
Epoch 24/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0030 - categorical_accuracy: 0.9820 - val_loss: 0.0243 - val_categorical_accuracy: 0.8859
Epoch 25/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0029 - categorical_accuracy: 0.9826 - val_loss: 0.0408 - val_categorical_accuracy: 0.8210

Epoch 00025: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
Epoch 26/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0026 - categorical_accuracy: 0.9846 - val_loss: 0.0291 - val_categorical_accuracy: 0.8662
Epoch 27/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0024 - categorical_accuracy: 0.9854 - val_loss: 0.0284 - val_categorical_accuracy: 0.8732
Epoch 28/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0024 - categorical_accuracy: 0.9858 - val_loss: 0.0264 - val_categorical_accuracy: 0.8811
Epoch 29/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0023 - categorical_accuracy: 0.9860 - val_loss: 0.0250 - val_categorical_accuracy: 0.8887
Epoch 30/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0023 - categorical_accuracy: 0.9866 - val_loss: 0.0248 - val_categorical_accuracy: 0.8909

Epoch 00030: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 31/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0022 - categorical_accuracy: 0.9870 - val_loss: 0.0252 - val_categorical_accuracy: 0.8920
Epoch 32/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0021 - categorical_accuracy: 0.9874 - val_loss: 0.0239 - val_categorical_accuracy: 0.8957
Epoch 33/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0021 - categorical_accuracy: 0.9878 - val_loss: 0.0234 - val_categorical_accuracy: 0.8983
Epoch 34/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0021 - categorical_accuracy: 0.9875 - val_loss: 0.0241 - val_categorical_accuracy: 0.8971
Epoch 00034: early stopping
========= generating oof predictions 11:13:55 =========
========= generating test set predictions 11:14:02 =========
========= fitting 5 th model 11:15:21 =========
Epoch 1/65535
420/420 [==============================] - 97s 231ms/step - loss: 0.0883 - categorical_accuracy: 0.4474 - val_loss: 0.0839 - val_categorical_accuracy: 0.5234
Epoch 2/65535
420/420 [==============================] - 96s 230ms/step - loss: 0.0237 - categorical_accuracy: 0.8625 - val_loss: 0.0299 - val_categorical_accuracy: 0.8381
Epoch 3/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0157 - categorical_accuracy: 0.9113 - val_loss: 0.0236 - val_categorical_accuracy: 0.8784
Epoch 4/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0129 - categorical_accuracy: 0.9269 - val_loss: 0.0340 - val_categorical_accuracy: 0.8139
Epoch 5/65535
420/420 [==============================] - 96s 228ms/step - loss: 0.0111 - categorical_accuracy: 0.9375 - val_loss: 0.0283 - val_categorical_accuracy: 0.8503
Epoch 6/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0103 - categorical_accuracy: 0.9418 - val_loss: 0.0397 - val_categorical_accuracy: 0.7824
Epoch 7/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0093 - categorical_accuracy: 0.9475 - val_loss: 0.0406 - val_categorical_accuracy: 0.7837
Epoch 8/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0088 - categorical_accuracy: 0.9500 - val_loss: 0.0189 - val_categorical_accuracy: 0.9037
Epoch 9/65535
420/420 [==============================] - 96s 228ms/step - loss: 0.0083 - categorical_accuracy: 0.9529 - val_loss: 0.0347 - val_categorical_accuracy: 0.8207
Epoch 10/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0083 - categorical_accuracy: 0.9530 - val_loss: 0.0383 - val_categorical_accuracy: 0.8101
Epoch 11/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0076 - categorical_accuracy: 0.9569 - val_loss: 0.0236 - val_categorical_accuracy: 0.8831
Epoch 12/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0070 - categorical_accuracy: 0.9600 - val_loss: 0.0328 - val_categorical_accuracy: 0.8375
Epoch 13/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0068 - categorical_accuracy: 0.9606 - val_loss: 0.0405 - val_categorical_accuracy: 0.7935
Epoch 14/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0067 - categorical_accuracy: 0.9615 - val_loss: 0.0401 - val_categorical_accuracy: 0.8053

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
Epoch 15/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0049 - categorical_accuracy: 0.9713 - val_loss: 0.0391 - val_categorical_accuracy: 0.8076
Epoch 16/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0043 - categorical_accuracy: 0.9751 - val_loss: 0.0325 - val_categorical_accuracy: 0.8442
Epoch 17/65535
420/420 [==============================] - 96s 228ms/step - loss: 0.0041 - categorical_accuracy: 0.9764 - val_loss: 0.0303 - val_categorical_accuracy: 0.8615
Epoch 18/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0039 - categorical_accuracy: 0.9775 - val_loss: 0.0265 - val_categorical_accuracy: 0.8811
Epoch 19/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0037 - categorical_accuracy: 0.9784 - val_loss: 0.0289 - val_categorical_accuracy: 0.8721

Epoch 00019: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
Epoch 20/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0034 - categorical_accuracy: 0.9803 - val_loss: 0.0264 - val_categorical_accuracy: 0.8824
Epoch 21/65535
420/420 [==============================] - 96s 228ms/step - loss: 0.0033 - categorical_accuracy: 0.9808 - val_loss: 0.0239 - val_categorical_accuracy: 0.8917
Epoch 22/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0032 - categorical_accuracy: 0.9816 - val_loss: 0.0244 - val_categorical_accuracy: 0.8912
Epoch 23/65535
420/420 [==============================] - 96s 229ms/step - loss: 0.0032 - categorical_accuracy: 0.9812 - val_loss: 0.0222 - val_categorical_accuracy: 0.9005
Epoch 00023: early stopping
========= generating oof predictions 11:52:14 =========
========= generating test set predictions 11:52:20 =========
train loss avg 0.002685847911518145 -- std 0.00036072676219263064, val loss avg 0.019704916495328982 -- std 0.003355876339673596
train acc avg 0.9842784556461069 -- std 0.0021849803378649166, val acc avg 0.9121204569818493 -- std 0.013268229269318874
mean nb epochs 40.4
dump oof predicted probs
dump test set predicted probs
