#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from util import *
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.metrics import accuracy_score
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt

df = get_training_label_wavfilename(fullset = True)
np.random.seed(SEED)
df = shuffle(df)
y = df.label.map(full_label_num_mapping)
test_df = get_test_wavefilename()

models = [
    "d-rawwav_m-1dcnn_s-1",                                               # 0.85
    "d-rawwav_m-1dcnn_s-xl",                                              # ?
    "d-rawwav_m-1dcnn_s-l",                                               # 0.80

    "d-mfcc_10_40_40_m-ddnn_s-xl",
    "d-mfcc_10_40_20_m-arm_gru_s-xl",                                     # 0.84
    "d-mfcc_10_40_20_m-arm_lstm_s-xl",                                    # ?
    "d-mfcc_10_40_20_m-arm_crnn_s-l",                                     # 0.85
    "d-mfcc_10_40_20_m-arm_dscnn_s-xl",                                   # 0.86
    "d-mfcc_40_25_10_m-gg4_s-3",                                          # 0.86

    "d-logspectrogram_25_18.75_m-arm_crnn_s-xl" ,                         # 0.86
    "d-logspectrogram_25_18.75_m-gg4_s-1" ,                               # 0.87
    "d-logspectrogram_25_18.75_m-join_s-t" ,                              # 0.87
    "d-logspectrogram_25_18.75_m-arm_dscnn_s-t",                          # ?
    "d-logspectrogram_25_18.75_m-resnet_s-18t",                           # ?

    "d-logmelsepctrogram_40_25_10_m-resnet_s-18",                         # 0.87
    "d-logmelsepctrogram_40_25_10_m-resnet_s-34",                         # 0.87
    "d-logmelsepctrogram_40_25_10_m-arm_crnn_s-l" ,                       # 0.87
    "d-logmelsepctrogram_40_25_10_m-arm_crnn_s-xl" ,                      # 0.87
    "d-logmelsepctrogram_40_25_10_m-arm_dscnn_s-l",                       # 0.87
    "d-logmelsepctrogram_40_25_10_m-gg4_s-2" ,                            # 0.88
    "d-logmelsepctrogram_40_25_10_m-gg4_s-3" ,                            # 0.88
    "d-logmelsepctrogram_40_25_10_m-join_s-s",                            # 0.86
]

predicts = pd.concat([
        pd.read_csv(inp)[full_labels].idxmax(axis = 1)
        for inp in ["../models/{}/train_meta_probs.csv".format(model)
        for model in models]]
    , axis = 1)

predicts.columns = models

def get_num_aggreement(df, col1, col2):
    return sum(df[col1] == df[col2])/df.shape[0]

mat = pd.DataFrame(np.zeros((len(models), len(models))), columns = models, index = models)
for model1, model2 in combinations_with_replacement(models, 2):
    mat[model1][model2] = get_num_aggreement(predicts, model1, model2)

mat += mat.T
mat.values[[np.arange(mat.shape[0])]*2] = 1
print(np.round(mat.values, 2))
plt.matshow(mat)
plt.show()
plt.close()

def generate_submission(models):
    X = pd.concat([pd.read_csv(inp)[full_labels] for inp in ["../models/{}/train_meta_probs.csv".format(model) for model in models]], axis = 1)
    X_test = pd.concat([pd.read_csv(inp)[full_labels] for inp in ["../models/{}/test_meta_probs.csv".format(model) for model in models]], axis = 1)
    col_names = ["{}_{}".format(i, j) for i in ["model_{}".format(k) for k in range(len(models))] for j in full_labels]
    X.columns, X_test.columns = col_names, col_names
    folds = get_folds()

    print("===Ridge===")
    ridge_cv = RidgeClassifierCV(alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 100], cv = folds).fit(X,y)
    print("best alpha value is: {}".format(ridge_cv.alpha_))
    ridge_model = RidgeClassifier(alpha = ridge_cv.alpha_).fit(X, y)
    print(accuracy_score(y, ridge_model.predict(X)))
    test_df['label'] = pd.Series(ridge_model.predict(X_test)).map(full_num_label_mapping)
    test_df['label'] = test_df['label'].map(lambda x: "unknown" if x not in labels else x)
    test_df.to_csv("ridge_on_{}_models.csv".format(len(models)), index = False)

low_cor_mat = mat[mat.mean(axis = 1) < mat.mean().mean()]

generate_submission(mat.index.tolist())
generate_submission(low_cor_mat.index.tolist())
