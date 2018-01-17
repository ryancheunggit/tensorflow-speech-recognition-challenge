# tensorflow-speech-recognition-challenge

#48 Solution to https://www.kaggle.com/c/tensorflow-speech-recognition-challenge on small foot-print keyword spotting with neural networks.

**Note:** reproducibility is not 100% guaranteed, but you should be able to get models with 88% public accuracy and 89% private accuracy.

For first layer models, I used the model archituctures described in Arms' paper *Hello Edge: Keyword Spotting on Microcontrollers*[https://arxiv.org/abs/1711.07128] with a bit tweek of mine own. Also tried out 1d cnn and resnet.  

My second layer stacker is a simple Ridge Classifier, and the stacking only improved very slightly over the best model. 

To sort of reproduce my result:
1. clone the repo   
2. download competition data and unzip to `input` directory . 
3. run `run.sh` . 

