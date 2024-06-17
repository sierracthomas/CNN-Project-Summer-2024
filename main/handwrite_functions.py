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
# import cv2 for optional image processing
import cv2 as cv

# import seaborn for other nice plotting
import seaborn as sn
# import pandas for data management (particularly for large sets of data)
import pandas as pd


# compute confusion matrix 
def comp_confmat(actual, predicted):

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
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
    plt.figure(figsize = (10,7))


    ax = sn.heatmap(df_cm, annot=True,cmap="OrRd")
    ax.set(xlabel="Truth", ylabel="Predicted")
    plt.suptitle("Confusion matrix of model on 10,000 tests")
    ax.xaxis.tick_top()
    plt.savefig(savepic)


def denoise_img(custom_image_path, output_image_path):
    img = cv.imread(custom_image_path)
    dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.axis(False);
    plt.show()
    plt.imshow(dst)
    plt.axis(False);
    plt.savefig(output_image_path)

