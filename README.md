# Car-Brand-Classification
A PyTorch model for a car brand classification based on ResNet152. The cars dataset contains 16,185 images of 196 classes of cars. (11,185 for training; 5,000 for testing)

### Hardware
- Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz
- NVIDIA GeForce RTX 2080 Ti

### Environment
- Microsoft win10
- Python 3.7.3
- Pytorch 1.7.0
- CUDA 10.2

### Install Packages
- pandas, matplotlib
```
pip install pandas
pip install matplotlib
```
- pytorch: see https://pytorch.org/get-started/locally/

### Data Preparation
Download the given dataset from [kaggle](https://www.kaggle.com/c/cs-t0828-2020-hw1/data).
```
dataset
  +- testing_data / testing_data
  +- training_data / training_data
  +- training_labels.csv
```

And run command `python data_prepare.py` to reorganize the train and valid data structure as below:
```
train/
├── class1
│   ├── aaa.jpg
│   ├── bbb.jpg
│   └── ccc.jpg	
├── class2
│   ├── ddd.jpg
│   ├── eee.jpg
│   └── fff.jpg	
│        .
│        .
│        .
└── classN
    ├── xxx.jpg
    ├── yyy.jpg
    └── zzz.jpg	
```

### Training
- split the training data into 9:1 for train and valid
- set the parameters for ResNet152
- change the desired number of epochs
- start training the model

### Run the program
1. create your working directory and run command `git clone https://github.com/chia56028/Car-Brand-Classification.git`
2. put the organized training dataset into the cloned folder and run command `python data_prepare.py` to do the data preparation
3. run command `python hw1.py` to train

※ get more info by `python hw1.py --help`
```
usage: hw1.py [-h] [-r WORKING_DIR] [-tr TRAINING_DIR] [-te TESTING_DIR]
              [-l LABEL_PATH] [-n MODEL_NAME] [-t IS_TRAIN] [-p IS_PREDICT]
              [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -r WORKING_DIR, --root WORKING_DIR
                        path to dataset
  -tr TRAINING_DIR, --train_dir TRAINING_DIR
                        path to training set
  -te TESTING_DIR, --test_dir TESTING_DIR
                        path to testing set
  -l LABEL_PATH, --label LABEL_PATH
                        path to label file
  -n MODEL_NAME, --model_name MODEL_NAME
                        name the model
  -t IS_TRAIN, --train IS_TRAIN
                        train
  -p IS_PREDICT, --predict IS_PREDICT
                        predict
  -d DEVICE, --device DEVICE
```

### References
- [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#transfer-learning-for-computer-vision-tutorial)
- [Deep CARs— Transfer Learning With Pytorch](https://towardsdatascience.com/deep-cars-transfer-learning-with-pytorch-3e7541212e85)