#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

from util import *
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

def main():
    df = get_training_label_wavfilename(fullset = True)
    np.random.seed(SEED)
    df = shuffle(df)
    splitter = StratifiedKFold(n_splits = NUM_FOLDS, shuffle = True, random_state = 2014)
    layer1_folds = list(splitter.split(df, df.label))
    pickle.dump(layer1_folds, open('../input/folds.pkl', 'wb'))

if __name__ == '__main__':
    main()
