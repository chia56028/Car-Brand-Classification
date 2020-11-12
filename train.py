import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

import time
import copy
import os
import glob

from pathlib import Path

# [train] #


class Trainer():
    def __init__(self, working_dir, training_dir, testing_dir, label_path, model_name, epoch, learning_rate):
        self.root_dir = working_dir
        self.train_dir = self.root_dir+training_dir
        self.test_dir = self.root_dir+testing_dir
        self.label_path = self.root_dir+label_path

        self.dataset_sizes = None
        self.dataloaders = None

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomPerspective(distortion_scale=0.5),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomPerspective(distortion_scale=0.5),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]),
        }

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.load_model()

        self.model_name = model_name

        self.epoch = epoch
        self.learning_rate = learning_rate

    def load_model(self):
        # load pretrained model
        self.model = models.resnet152(pretrained=True)
        num_in_feature = 2048

        for param in self.model.parameters():
            param.require_grad = False

        hidden_layers = None    # [1050, 500]

        classifier = self.bulid_classifier(num_in_feature, hidden_layers, 196)
        print('\n[Classifier]')
        print(classifier)

        self.model.fc = classifier

    def train(self):
        train_on_gpu = torch.cuda.is_available()

        if not train_on_gpu:
            print('Cuda is not available.')
        else:
            print('Cuda is available!')

        # datapath = './cs-t0828-2020-hw1/'
        label_df = pd.read_csv(self.label_path)

        batch_size = 32
        dataset = datasets.ImageFolder(
            self.train_dir, transform=self.data_transforms['train'])

        valid_size = int(0.1*len(dataset))
        train_size = len(dataset) - valid_size
        self.dataset_sizes = {'train': train_size, 'valid': valid_size}

        train_dataset, valid_datset = torch.utils.data.random_split(
            dataset, [train_size, valid_size])

        # load datasets into dataloader
        self.dataloaders = {'train': DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True),
                            'valid': DataLoader(dataset=valid_datset, batch_size=batch_size, shuffle=False)}

        print('\n[Dataset Infomation]')
        print('Total Number of Sample:', len(dataset))
        print('Number of Sample in Train:', len(train_dataset))
        print('Number of Sample in Valid:', len(valid_datset))
        print('Number of Classes:', len(dataset.classes))
        print('First Class:', dataset.classes[0])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load pretrained model
        model = models.resnet152(pretrained=True)
        num_in_feature = 2048

        for param in model.parameters():
            param.require_grad = False

        hidden_layers = None    # [1050, 500]

        classifier = self.bulid_classifier(num_in_feature, hidden_layers, 196)
        print('\n[Classifier]')
        print(classifier)

        model.fc = classifier
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=self.learning_rate, momentum=0.9)
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='max',
                                                     patience=3,
                                                     threshold=0.9)

        epochs = self.epoch
        # model.load_state_dict(torch.load(self.model_name))
        model.to(device)
        model, train_results, valid_results = self.train_model(
            model, criterion, optimizer, sched, epochs, self.model_name)

    # create custom classifier

    def bulid_classifier(self, num_in_features, hidden_layers, num_out_features):
        classifier = nn.Sequential()
        if hidden_layers is None:
            classifier.add_module('fc0', nn.Linear(num_in_features, 196))
        else:
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            classifier.add_module('fc0', nn.Linear(num_in_features,
                                                   hidden_layers[0]))
            classifier.add_module('relu0', nn.ReLU())
            classifier.add_module('drop0', nn.Dropout(.6))

            for i, (h1, h2) in enumerate(layer_sizes):
                classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
                classifier.add_module('reLU'+str(i+1), nn.ReLU())
                classifier.add_module('drop'+str(i+1), nn.Dropout(.5))

            classifier.add_module('output', nn.Linear(hidden_layers[-1],
                                                      num_out_features))

        return classifier

    # train model
    def train_model(self, model, criterion, optimizer, sched, num_epochs=5,
                    model_name='save_model.pt', device='cuda'):
        start = time.time()
        train_results = []
        valid_results = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # sched.step()
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / \
                    self.dataset_sizes[phase]

                # calculate average time over an epoch
                # elapshed_epoch = time.time() - start/
                # print('Epoch {}/{} - completed in: {:.0f}m {:.0f}s'.format(
                #     epoch+1, num_epochs,elapshed_epoch // 60,
                #     elapshed_epoch % 60))

                if(phase == 'train'):
                    train_results.append([epoch_loss, epoch_acc])
                elif(phase == 'valid'):
                    # sched.step(epoch_acc)
                    valid_results.append([epoch_loss, epoch_acc])
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model (Early Stopping) and Saving our model,
                # when we get best accuracy
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # print('Validation loss decreased ({:.6f} --> {:.6f}). \
                    #       Saving model ...'.format(valid_loss_min, valid_loss))

                    model_save_name = model_name
                    path = F"./{model_save_name}"
                    torch.save(model.state_dict(), path)

            print()

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, train_results, valid_results

    def predict(self):
        # Load trained model
        self.model.load_state_dict(torch.load(self.model_name))
        self.model.to(self.device)

        with torch.no_grad():
            self.model.eval()

            dataset = datasets.ImageFolder(
                self.test_dir, transform=self.data_transforms['test'])
            print(dataset.classes)

            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=False, num_workers=2)

            image_names = []
            for index in testloader.dataset.imgs:
                # image_names.append(index[0].split('/')[-1])
                image_names.append(Path(index[0]).stem)

            results = []
            # results.append(image_names)

            # class_names = dataset.classes
            for inputs, labels in testloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                for i in predicted:
                    results.append(int(i)+1)

        print("Predictions on Test Set:")
        df = pd.DataFrame({'Id': image_names, 'Predicted': results})
        pd.set_option('display.max_colwidth', None)

        # sort result
        label_df = pd.read_csv(self.label_path)
        sector = label_df.groupby('label')
        labels = [i[0] for i in sector]

        result_df = df[['Id', 'Predicted']]

        for i in range(196):
            result_df['Predicted'] = result_df['Predicted'].replace(
                [i+1], labels[i])

        result_df = result_df.rename(
            columns={'Id': 'id', 'Predicted': 'label'})
        print(result_df.head(30))
        result_df.to_csv('./results.csv', index=False)
