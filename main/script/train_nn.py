#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
#from handwrite_functions import *
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
"""
#torch.manual_seed(6)
torch.manual_seed(15)
IMAGE_LOC = "/Users/sierra/Dropbox/datsets/nu_mu/data_8_7/QES_1"
BATCH_SIZE = 128
EPOCH_NUMBER = 20
datasize = 4000
testsize_per_channel = 100
png_header = "test"
plot_frequency = 1
"""

parser = argparse.ArgumentParser(description='Train a convolutional neural network for neutrino interactions. ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", type = int, default = 5, help = "Set random seed")
#parser.add_argument("--image_dir", type = str, default = "/home/sthoma31/neutrino_interaction_images/nu_mu_700/data/QES", help = "Location that contains interaction folders CC and NC")
parser.add_argument("--cc_folder", type = str, default = "/home/sthoma31/neutrino_interaction_images/nu_mu_700/data/QES/CC1", nargs='*', help = "Name of folder containing CC interactions")
parser.add_argument("--nc_folder", type = str, default = "/home/sthoma31/neutrino_interaction_images/nu_mu_700/data/QES/NC1", nargs='*', help = "Name of folder containing NC interactions")
parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size when training")
#parser.add_argument("--training_data_size", type = int, default = 4000, help = "Size of training data")
parser.add_argument("--epoch", type = int, default = 20, help = "Number of epochs when training")
parser.add_argument("--testing_data_size", type = int, default = 100, help = "Size of testing data for each channel ")
parser.add_argument("--png_header", type = str, default = "trial", help = "Header name for PNG files")
parser.add_argument("--plot_freq", type = int, default = 5, help = "Plot confusion matrices every {plot_freq} times")

args = parser.parse_args()
torch.manual_seed(args.seed)
#IMAGE_LOC = args.image_dir
BATCH_SIZE = args.batch_size
EPOCH_NUMBER = args.epoch
#datasize = args.training_data_size
testsize_per_channel = args.testing_data_size
png_header = args.png_header
plot_frequency = args.plot_freq

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

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
    
pathNC = []
labelNC = []

pathCC = []
labelCC = []

paths, labels = [], []

label_map = {0:"NC",
             1:"CC"
            }

for myfilepath in args.nc_folder:
    for nc_path in glob(f"{myfilepath}/*"):
        pathNC.append(nc_path)
        labelNC.append(0)

for myfilepath in args.cc_folder:
    for cc_path in glob(f"{myfilepath}/*"):
        pathCC.append(cc_path)
        labelCC.append(1)
    
print("Real NC lengths are ", len(pathNC), len(labelNC))
print("Real CC lengths are ", len(pathCC), len(labelCC))
datasize = len(pathNC) - testsize_per_channel
print(f"Maximum length is: {datasize}")

paths = pathNC[0:datasize] + pathCC[0:datasize]
labels = labelNC[0:datasize] + labelCC[0:datasize]
dataset = CustomDataset(paths,labels,(250,250))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                           shuffle=True)


paths_test = pathNC[datasize + 1: datasize + 1 + testsize_per_channel] + pathCC[datasize + 1:datasize + 1 + testsize_per_channel]
labels_test = labelNC[datasize + 1:datasize + 1 + testsize_per_channel] + labelCC[datasize + 1:datasize + 1 + testsize_per_channel]

test_dataset = CustomDataset(paths_test, labels_test, (250,250))
validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                shuffle=True)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 256 * 2 * 2)
    
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
                              #nn.Unflatten(1, (-1, 256 * 2 * 2)), #x = x.view(-1, 256 * 2 * 2)
                            nn.Linear(256 * 2 * 2,512),
                            nn.Linear(512,2),
                            nn.LogSoftmax(dim=1))

def eval_for_confmat(validation_loader, model = model):
    total_val_loss = 0.0
    total_true = 0

    actual = []
    predicted = []

    # When we're not working with gradients and backpropagation
    # we use torch.no_grad() utility.
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
            #print(preds)
            predicted.append(np.array(preds))
            #print(target_)
            actual.append(np.array(target_))
            total_true += true

    validation_accuracy = round(100 * total_true / total,2)
    #print(f"Validation accuracy: {validation_accuracy}%")
    #print(f"Validation loss: {round(total_val_loss,2)}%")
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
    



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(),lr=1e-4)


TRAIN_LOSS = []
TRAIN_ACCURACY = []

for epoch in range(1,EPOCH_NUMBER+1):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for data_,target_ in train_loader:
        # We have to one hot encode our labels.
        target_ =target_.to(device)
        data_ = data_.to(device)
        
        # Cleaning the cached gradients if there are
        optimizer.zero_grad()
        
        # Getting train decisions and computing loss.
        outputs = model(data_)
        loss = criterion(outputs,target_)
        
        # Backpropagation and optimizing.
        loss.backward()
        optimizer.step()
        
        # Computing statistics.
        epoch_loss = epoch_loss + loss.item()
        _,pred = torch.max(outputs,dim=1)
        correct = correct + torch.sum(pred == target_).item()
        total += target_.size(0)
    
    # Appending stats to the lists.
    TRAIN_LOSS.append(epoch_loss)
    TRAIN_ACCURACY.append(100 * correct / total)
    print(f"Epoch {epoch}: Accuracy: {100 * correct/total}, Loss: {epoch_loss}")
    if epoch % plot_frequency == 0:
        actual, predicted = eval_for_confmat(validation_loader, model = model)
        confmat = comp_confmat(actual, predicted)
        plot_confusion_matrix(confmat, f"{png_header}_{epoch}.png")
        model.train()


fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(range(EPOCH_NUMBER),TRAIN_LOSS)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")

ax[1].plot(range(EPOCH_NUMBER),TRAIN_ACCURACY)
ax[1].set_ylabel("Validation accuracy (%)")
ax[1].set_xlabel("Epoch")
plt.savefig(f"{png_header}_loss.png")
plt.show()


images,labels = next(iter(validation_loader))
type(labels)


fig, axis = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        npimg = images[i].numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        label = label_map[int(labels[i])]
        ax.imshow(npimg, cmap = "Greys_r")
        ax.set(title = f"{label}")

torch.random.initial_seed()

