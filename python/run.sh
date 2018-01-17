#!/bin/bash
python generate_silence_wav.py
python generate_training_df.py
python generate_folds.py
mkdir ../models
mkdir ../submissions

python train.py rawwav 1dcnn 1 128
python train.py rawwav 1dcnn xl 64
python train.py rawwav 1dcnn l 128

python train.py mfcc_10_40_40 ddnn xl 1280
python train.py mfcc_10_40_20 arm_gru xl 128
python train.py mfcc_10_40_20 arm_lstm xl 128
python train.py mfcc_10_40_20 arm_crnn l 128
python train.py mfcc_10_40_20 arm_dscnn xl 128
python train.py mfcc_40_25_10 gg4 3 128

python train.py logsepctrogram_25_18.75 arm_crnn xl 128
python train.py logspectrogram_25_18.75 gg4 1 128
Python train.py logspectrogram_25_18.75 joint t 128
python train.py logspectrogram_25_18.75 arm_dscnn t 128
python train.py logspectrogram_25_18.75 resnet 18t 128

python train.py logmelsepctrogram_40_25_10 resnet 18 128
python train.py logmelsepctrogram_40_25_10 resnet 34 128
python train.py logmelsepctrogram_40_25_10 arm_crnn l 128
python train.py logmelsepctrogram_40_25_10 arm_crnn xl 128
python train.py logmelsepctrogram_40_25_10 arm_dscnn l 128
python train.py logmelsepctrogram_40_25_10 gg4 2 128
python train.py logmelsepctrogram_40_25_10 gg4 s 128
python train.py logmelsepctrogram_40_25_10 joint s 128

python second_layer_stacker.py
