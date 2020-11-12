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
Download the given dataset and reorganize the train and valid data structure as below:
```
Training_data/
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
