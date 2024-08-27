# machine learning libraries
import torch
from torchvision import datasets, transforms
from torch import nn, optim
# import torchvision, which is a complementary tool for image importing in pytorch
import torchvision 
# import numpy for math stuff
import numpy as np
# import matplotlib for nice plotting
import matplotlib.pyplot as plt
# import time so we can see how long training takes!
from time import time
# import seaborn for other nice plotting
import seaborn as sn
# import pandas for data management (particularly for large sets of data)
import pandas as pd

import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from glob import glob
from PIL import Image
from nn_inputs import * 

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pixels that make up detector exterior - comment out lambda in `transforms` if not desired
box = np.zeros([194, 89, 1])
box = np.pad(box, [(28, 28), (9,9), (0,0)], mode='constant', constant_values=1)
box = np.concatenate((box,box), axis=1)
box = np.pad(box, [(0, 0), (20,16), (0,0)], mode='constant', constant_values=1)
box = np.array(box, dtype = np.float32)
torchbox = torch.from_numpy(box)
torchbox = torchbox.permute(2,0,1)

# custom dataset class
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
# custom dataset class with detector exterior subtracted
#transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Lambda(lambda x : x - torchbox)])

class CustomDataset(Dataset):
    def __init__(self,img_paths,img_labels,size_of_images):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.size_of_images = size_of_images
        if len(self.img_paths) != len(self.img_labels):
            raise InvalidDatasetException(self.img_paths,self.img_labels)
            
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self,index):
        PIL_IMAGE = Image.open(self.img_paths[index]).resize(self.size_of_images)
        TENSOR_IMAGE = transform(PIL_IMAGE)
        label = self.img_labels[index]
        
        return TENSOR_IMAGE,label
    
label_map = {0:"NC",
             1:"CC"
            }


class dataloaders(Dataset):
    
    def __init__(self, nc_image_folders, cc_image_folders, testsize_per_channel, datasize = None):
        self.nc_image_folders = nc_image_folders
        self.cc_image_folders = cc_image_folders
        self.testsize_per_channel = testsize_per_channel
        self.datasize = datasize
    def NC_CC(self):
        pathNC, labelNC = [], []
        pathCC, labelCC = [], []
        paths, labels = [], []

        for myfilepath in self.nc_image_folders:
            for nc_path in glob(f"{myfilepath}/*"):
                pathNC.append(nc_path)
                labelNC.append(0)

        for myfilepath in self.cc_image_folders:
            for cc_path in glob(f"{myfilepath}/*"):
                pathCC.append(cc_path)
                labelCC.append(1)
                
        if self.datasize == None: 
            self.datasize = len(pathNC) - self.testsize_per_channel
        return pathNC, labelNC, pathCC, labelCC
    
    def training(self):
        pathNC, labelNC, pathCC, labelCC = self.NC_CC()
        datasize = len(pathNC) - testsize_per_channel
        # combine training data for different channels with corresponding labels
        paths = pathNC[0:self.datasize] + pathCC[0:self.datasize]
        labels = labelNC[0:self.datasize] + labelCC[0:self.datasize]
        dataset = CustomDataset(paths,labels,(250,250))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                           shuffle=True)
        return paths, labels, train_loader
    
    def testing(self):
        pathNC, labelNC, pathCC, labelCC = self.NC_CC()
        # combine testing data channels and corresponding labels 
        paths_test = pathNC[self.datasize + 1: self.datasize + 1 + self.testsize_per_channel] + pathCC[self.datasize + 1:self.datasize + 1 + self.testsize_per_channel]
        labels_test = labelNC[self.datasize + 1:self.datasize + 1 + self.testsize_per_channel] + labelCC[self.datasize + 1:self.datasize + 1 + self.testsize_per_channel]
        test_dataset = CustomDataset(paths_test, labels_test, (250,250))
        validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                shuffle=True)
        return paths_test, labels_test, validation_loader
    
    
### get dataloaders with input data
interaction_data = dataloaders(nc_image_folders, cc_image_folders, testsize_per_channel)
pathNC, labelNC, pathCC, labelCC = interaction_data.NC_CC()
paths_test, labels_test, validation_loader = interaction_data.testing()
paths, labels, train_loader = interaction_data.training()

    
"""
pathNC, labelNC = [], []
pathCC, labelCC = [], []
paths, labels = [], []

label_map = {0:"NC",
             1:"CC"
            }

for myfilepath in nc_image_folders:
    for nc_path in glob(f"{myfilepath}/*"):
        pathNC.append(nc_path)
        labelNC.append(0)

for myfilepath in cc_image_folders:
    for cc_path in glob(f"{myfilepath}/*"):
        pathCC.append(cc_path)
        labelCC.append(1)
        
datasize = len(pathNC) - testsize_per_channel
# combine training data for different channels with corresponding labels
paths = pathNC[0:datasize] + pathCC[0:datasize]
labels = labelNC[0:datasize] + labelCC[0:datasize]


# combine testing data channels and corresponding labels 
paths_test = pathNC[datasize + 1: datasize + 1 + testsize_per_channel] + pathCC[datasize + 1:datasize + 1 + testsize_per_channel]
labels_test = labelNC[datasize + 1:datasize + 1 + testsize_per_channel] + labelCC[datasize + 1:datasize + 1 + testsize_per_channel]

# initialize datasets (training and testing) with 250 x 250 pixels as well as make dataloaders
dataset = CustomDataset(paths,labels,(250,250))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                           shuffle=True)

test_dataset = CustomDataset(paths_test, labels_test, (250,250))
validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                shuffle=True)
"""


# define class to flatten a layer
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 256 * 2 * 2)
    
# define model architecture 
model = nn.Sequential(nn.Conv2d(1,32,kernel_size=3,stride=2,padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(64), 
                            nn.MaxPool2d(2,2),
                            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(128),
                            nn.MaxPool2d(2,2),
                            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
                            nn.BatchNorm2d(256), 
                            nn.MaxPool2d(2,2),
                            Flatten(),
                            nn.Linear(256 * 2 * 2,512),
                            nn.Linear(512,2),
                            nn.LogSoftmax(dim=1))



## evaluate model on testing data for confusion matrix 
def eval_for_confmat(validation_loader, model = model, device = device, criterion = None):
    total_val_loss = 0.0
    total_true = 0
    total = 0
    actual = []
    predicted = []

    with torch.no_grad():
        model.eval()
        for data_,target_ in validation_loader:
            data_ = data_.to(device)
            target_ = target_.to(device)

            outputs = model(data_)
            loss = criterion(outputs,target_).item()
            _,preds = torch.max(outputs,dim=1)
            total_val_loss += loss
            true = torch.sum(preds == target_).item()
            predicted.append(np.array(preds))
            actual.append(np.array(target_))
            total_true += true
            total += target_.size(0)

    validation_accuracy = round(100 * total_true / total,2)
    return actual, predicted


# compute confusion matrix 
def comp_confmat(actual, predicted):
    actual = np.hstack(actual)
    predicted = np.hstack(predicted)
    # extract the different classes
    classes = np.unique(actual)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):

           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return confmat

# plot confusion matrix
def plot_confusion_matrix(confusion_matrix, savepic):
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in ['CC', 'NC']],
                  columns = [i for i in ['CC', 'NC']])
    plt.figure(figsize = (10,7))
    ax = sn.heatmap(df_cm, annot=True,cmap="OrRd")
    ax.set(ylabel="Truth", xlabel="Predicted")
    plt.suptitle(f"Confusion matrix of model on {np.sum(confusion_matrix)} tests")
    ax.xaxis.tick_top()
    plt.savefig(savepic)
    plt.show()
