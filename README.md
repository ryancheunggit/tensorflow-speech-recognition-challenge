# tensorflow-speech-recognition-challenge

## #48 Solution to [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) on small foot-print keyword spotting with neural networks.

To sort of reproduce my result:
1. clone the repo   
2. download competition data and unzip to `input` directory . 
3. run `run.sh` . 

**Note:** reproducibility is not 100% guaranteed, but you should be able to get models with 88% accuracy on public leaderboard and 89% accuracy on private leaderboard.

For first layer models, I used the model archituctures described in Arms' paper [*Hello Edge: Keyword Spotting on Microcontrollers*](https://arxiv.org/abs/1711.07128) with a bit tweek of mine own. Also tried out 1d cnn and resnet.  

My second layer stacker is a simple Ridge Classifier, and the stacking only improved very slightly over the best model. 

List of neural network archituctures:
1. Fully connected network / MLP . 
2. 1D/2D VGG style Convolutional neural network . 
3. Recurrent neural network   
4. Convolutional recurrent neural network .  
5. Depth seperatable convolutional neural network  
6. Joint model which concatenate intermediate layers from CNN and RNN . 
7. ResNet-18 and ResNet-34 . 
