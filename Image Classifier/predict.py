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
import argparse
import io
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import math
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import imageio
import os, random


def loadCheckpoint(path):
    checkpoint = torch.load(path)
    modelfun = getattr(models, 'vgg16')(pretrained=True)
    model=modelfun
    chPModel = checkpoint['model']
    hidden_layer = checkpoint['hidden_layer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.to('cuda')
    model.optimizer=checkpoint['optimizer']
    return model, class_to_idx
def LabelMapping(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def downloadModel():
    model = models.vgg16(pretrained=True)
    return model

def process_image(image):

    image = Image.open(image)

    resize=256
    width, height= image.size
    if width > height:
        image = image.resize((resize, height))
        
    else:
        image = image.resize((width, resize))
        
    image = image.crop((resize/2 - 224/2,
                       resize/2 - 224/2,
                       resize/2 + 224/2,
                       resize/2 + 224/2))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    imageArray = np.array(image) / 255
    imageNorm = (imageArray - mean) / std
    FinalImage = imageNorm.transpose((2, 0, 1))
    return FinalImage

def loadImage(path):

    img = Image.open(path)
    output = process_image(img)
    return img, output

def imshow(image, ax=None,title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    image = image.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5,gpu,cat_to_name):
    model.eval()
    if torch.cuda.is_available() and args.gpu == 'yes':
        model = model.cuda()
    else:
        model = model.cpu()
        
    img = process_image(image_path)
    imgTensor = torch.from_numpy(img)
    
    if torch.cuda.is_available() and args.gpu == 'yes':
        inputs = Variable(tensor.float().cuda())
    else: 
        inputs = Variable(tensor.float().cpu())
        
    imgTorch = inputs.unsqueeze(0)
    with torch.no_grad():
        output = model.forward(imgTorch.cuda())
    pro = torch.exp(output).data.topk(topk)
    probs = pro[0].cpu()
    classes = pro[1].cpu()
    class_to_idxx = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classesList = list()
    cat_to_name=LabelMapping(cat_to_name)
    for label in classes.numpy()[0]:
        classesList.append(class_to_idxx[label])
    classesNames = np.array([cat_to_name[x] for x in classesList])

    print(" image " +
          classesNames + "  class  " + probs.numpy()[0] + " probability")
    
 
parser = argparse.ArgumentParser(description='classes of a certin flower')
parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu')
parser.add_argument('--image_path', type=str, help='path of image to be predicted')
parser.add_argument("--model", type=str, required=True,default='vgg16')
parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
parser.add_argument('--saved_model_path' , type=str, default='flower102_checkpoint.pth', help='path of your saved model')
parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')

args = parser.parse_args()
    
print("Start Prediction")    

model, class_to_idx = loadCheckpoint(args.model)
img, output=loadImage(args.image_path)
predict(img, model, args.topk,args.gpu,args.cat_to_name)