# tensorflow-speech-recognition-challenge

## #48 Solution to [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) on small foot-print keyword spotting with neural networks.

## Instructions:
To sort of reproduce my result:
1. clone the repo   
2. download competition data and unzip to `input` directory .
3. run `run.sh` .

I used a 1070 workstation and a couple of compute instances with K80 on Google Cloud Platform. For some model I used different machine to train different folds, collect them and run `inference.py` to get oof and test predictions.

**Note:** reproducibility is not 100% guaranteed, but you should be able to get models with 88% accuracy on public leaderboard and 89% accuracy on private leaderboard.

## Inputs:
I have three different input features, MFCC, logspectrogram and logmelspectrogram.

## Models:
For first layer models, I used the model archituctures described in Arms' paper [*Hello Edge: Keyword Spotting on Microcontrollers*](https://arxiv.org/abs/1711.07128) with a bit tweak of mine own. Also tried out 1d cnn and resnet.  

List of first layer neural network archituctures:
1. Fully connected network / MLP .
2. 1D/2D VGG style Convolutional neural network .
3. Recurrent neural network . (I wasn't using Cudnn RNN at the time, so limited experiments were done)
4. Convolutional recurrent neural network .  
5. Depth seperatable convolutional neural network  
6. Joint model which concatenate intermediate layers from CNN and RNN .
7. ResNet-18 and ResNet-34 .(My implementation is wrong)

My second layer stacker is a simple Ridge Classifier, and the stacking only improved very slightly over the best model.

## Closing:
Overall, it is sort of my first time tackling a deep learning competition, and I learned a lot during which.
