import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import requests
from io import BytesIO
import io
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import argparse
import math
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import imageio
import argparse
import os, random


def data_preparation():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms  ={
    'transforms_train':transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),

    'transforms_test' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

     'transforms_validation' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])}

# TODO: Load the datasets with ImageFolder
    image_datasets = {'datasets_train' : datasets.ImageFolder(data_dir +'/train',transform=data_transforms['transforms_train']),
    'datasets_test' : datasets.ImageFolder(data_dir + '/test', transform=data_transforms['transforms_test']),
    'datasets_valid' : datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['transforms_validation'])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train_loader' : torch.utils.data.DataLoader(image_datasets['datasets_train'], batch_size=64, shuffle=True),
    'test_loader' : torch.utils.data.DataLoader(image_datasets['datasets_test'], batch_size=32),
    'valid_loader' : torch.utils.data.DataLoader(image_datasets['datasets_valid'], batch_size=32)}
    return image_datasets['datasets_train'], dataloaders['train_loader'], dataloaders['test_loader'], dataloaders['valid_loader']
                    
def pretrainedModel(modelName):
    model = None
    if modelName == "vgg16":
        model = models.vgg16(pretrained=True)
    if modelName == "vgg13":
        model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    return model
             
def classifierDefinition():
      classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(0.5)),
            ('hidden_layer1', nn.Linear(25088, 1024)),
            ('relu1', nn.ReLU()),
            ('hidden_layer2', nn.Linear(1024, 512)),
            ('relu2',nn.ReLU()),
            ('hidden_layer3',nn.Linear(512,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
        return classifier          


def trainAndValidate(model, train_loader, epoch, printE, criterion, optimizer, device='cpu'):
     printEvery = printE
     round = 0
     model.train()

     for e in range(epoch):
         running_loss = 0
         for i, (input1, label1) in enumerate(train_loader):
             round += 1
             if torch.cuda.is_available() and in_arg.gpu == 'yes':
                input1, label1 = input1.to('cuda'), label1.to('cuda')   
             else:
                input1, label1 = Variable(input1), Variable(label1)
             optimizer.zero_grad()
             output1 = model.forward(input1)
             loss = criterion(output1, label1)
             loss.backward()
             optimizer.step()
             running_loss += loss.item()
             if round % printEvery == 0:
                 model.eval()
                 lost = 0
                 accuracy=0
                 for ii, (input2,label2) in enumerate(dataloaders['valid_loader']):
                     optimizer.zero_grad()
                     if torch.cuda.is_available() and in_arg.gpu == 'yes':
                        input2, label2 = input2.to('cuda'), label2.to('cuda')
                     else:
                        input2, label2 = Variable(input2), Variable(label2)
                     #model.to('cuda')
                     with torch.no_grad():    
                         output2 = model.forward(input2)
                         lost = criterion(output2,label2)
                         ps = torch.exp(output2).data
                         equality = (label2.data == ps.max(1)[1])
                         accuracy += equality.type_as(torch.cuda.FloatTensor()).mean()

                 print("Epoch: {}/{}... ".format(e+1, epoch),
                   "Loss: {:.4f}".format(running_loss/printEvery),
                   "Validation Lost {:.4f}".format(lost),
                   "Accuracy: {:.4f}".format(accuracy))
                 running_loss = 0

  
def AccuracyCheck(test_loader,model):    
    correct = 0
    total = 0
    model.to('cuda')
    model.eval()
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to('cuda'), label.to('cuda')
            output = model(input)
            _, predict = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predict == label).sum().item()

    print('Accuracy on the test images: %d %%' % (100 * correct / total))
                    
def save_checkpoint(model, optimizer):
    model.class_to_idx =  image_datasets['datasets_train'].class_to_idx
    torch.save({'model' :'vgg16',
          'hidden_layer': 4096,
          'class_to_idx':model.class_to_idx,
          'classifier' : classifier,
          'optimizer': optimizer.state_dict()},
          'chp.pth')  
    print(" ---------------------------------- ")     
    print(" model has been saved ")
    print(" close application ")
    print(" ---------------------------------- ")  

parser = argparse.ArgumentParser(description='train a model program.')
parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu')
parser.add_argument('--arch', type=str, default='vgg16', help='architecture [available: densenet, vgg16, vgg13]', required=True)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=str, default='512', help='hidden units for fc layers (comma separated)')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
parser.add_argument('--saved_model_path' , type=str, default='flower102_checkpoint.pth', help='path of your saved model')
args = parser.parse_args()    

                
print("Training will be as followed:")
print("GPU: " + str(args.gpu))
print("LR: " + str(args.lr))
print("dataset directory: " + str(args.data_dir))
print("epochs: " + str(args.epochs))
print("Pretrained model: " + args.arch)
print("Hidden units: " + str(args.hidden_units))                    
print("path of cat_to_name : " + str(args.cat_to_name))
print("Saving model to: " + str(args.saved_model_path))  
print(" Start application ")     
                    
datasets_train,train_loader,test_loader,valid_loader = data_preparation()   
model=pretrainedModel("vgg16")    
classifier=classifierDefinition()   
model.classifier = classifier
if args.gpu:
        model.to('cuda')
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), args.lr )   

trainAndValidate(model, train_loader, args.epochs, 40, criterion, optimizer,  args.gpu)  
AccuracyCheck(test_loader,model)  
save_checkpoint(model, optimizer)                    
