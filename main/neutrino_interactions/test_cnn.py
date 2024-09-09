from nn_functions import *
from nn_inputs import *

# load model
model = torch.load('../recent.pt')
model.eval()

criterion = nn.CrossEntropyLoss()


"""images,labels = next(iter(validation_loader))
fig, axis = plt.subplots(3, 5, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        npimg = images[i].numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        label = label_map[int(labels[i])]
        ax.imshow(npimg, cmap = "Greys_r")
        ax.set(title = f"{label}")
plt.close()"""


actual, predicted, outputs, filepaths = eval_for_confmat(validation_loader, model = model, criterion = criterion)
confmat = comp_confmat(actual, predicted)
print(2 * testsize_per_channel)
print(len(filepaths))
print("-----------------")
print(len(np.hstack(filepaths)))
plot_confusion_matrix(confmat, f"{png_header}.png")

filepaths = np.hstack(filepaths)
actual = np.hstack(actual)
my_outputs = np.vstack(outputs)

with open("accuracy.csv", "w") as my_file:
    for i in range(0, 2 * testsize_per_channel - 1):
        my_file.write(f"{round(my_outputs[i][0], 4):.4f}, {round(my_outputs[i][1], 4):.4f}, {actual[i]}, {filepaths[i]} \n")
        #probs = torch.nn.functional.softmax(torch.Tensor(my_outputs[i]), dim=1)
        #print(probs)
        #conf, classes = torch.max(probs, 1)



## read CSV and plot

import pandas as pd
import matplotlib

accuracy = np.genfromtxt("accuracy.csv", delimiter=",")
font = {'family' : 'normal',
        'size'   : 28}

matplotlib.rc('font', **font)

plt.figure(figsize=[15, 5])
num_bins = 100
start = 0.
stop = 1.
bins = np.linspace(start, stop, num_bins + 1)


df = pd.read_csv('accuracy.csv', names=["confidence 0", "confidence 1", "actual", "location"])

for i in bins:
    df[f"{i}"] = 0
for i in bins:
    df.loc[df["confidence 1"] >= i, f"{i}"] = 1
    
# accuracy

accuracy = []
for i in bins:
    accuracy.append(len(df.loc[(df["actual"] == 1) & (df[f"{i}"] == 1)][f"{i}"].tolist())/np.sum(df[f"{i}"].tolist()))


# efficiency: fraction of true events that are accepted
actual_total = len(df.loc[(df["actual"] == 1)])
efficiency = []
for i in bins:
    efficiency.append(np.sum(df.loc[(df[f"{i}"] == 1) & (df["actual"] == 1)][f"{i}"].tolist())/len(df.loc[(df["actual"] == 1)]["actual"].tolist()))
    


plt.figure(figsize = [15,10])
plt.plot(bins, accuracy, label=r"Accuracy ( $\geq$ bin)", linewidth = 5)



#plt.suptitle(r"Accuracy ( $\geq$ bin)")
plt.xlabel(r"Confidence value that event is $\nu_\mu$CC")
#plt.ylabel(r"$\frac{\mathrm{number right}}{\mathrm{number guessed at confidence}}$")

#plt.legend()
#plt.show()

# efficiency 
#plt.figure(figsize=[15, 5])

plt.plot(bins, efficiency, label=r"Efficiency ( $\geq$ bin)", linewidth = 5)


plt.xlabel(r"Confidence value that event is $\nu_\mu$CC")
#plt.ylabel(r"$\frac{\mathrm{Guessed \,\,test} \nu_\mu \mathrm{CC}}{\mathrm{total} \nu_\mu \mathrm{CC}}$")
plt.suptitle(r"Efficiency/accuracy")
plt.ylabel(r"fraction")
#plt.xscale("log")
plt.ylim(0, 1.0)
plt.legend()
plt.ylim(0.0,1.1)
plt.savefig("efficiency_accuracy.png")
plt.close()


## plot images for each confusion block

import re


def example_plots(bad_images, bad_images_truth, predicted, truth_val, plot_title):
    example_images = CustomDataset(bad_images, bad_images_truth, (250,250))
    example_loader = torch.utils.data.DataLoader(example_images, batch_size=len(bad_images), shuffle=True)
    images, labels, img_path = next(iter(example_loader))
    fig, axis = plt.subplots(2, int(len(bad_images)/2), figsize=(35, 20))
    for i, ax in enumerate(axis.flat):
        with torch.no_grad():
            npimg = images[i].numpy()
            npimg = np.transpose(npimg, (1, 2, 0))
            label = label_map[int(labels[i])]
            ax.imshow(npimg - box, cmap = "Greys_r")
            last_num = re.findall(r'\d+', img_path[i])[-1]
            ax.set(title = f"Predicted {label_map[predicted]}, Truth {label_map[truth_val]}\n {last_num}")
            ax.axis("off")
            #ax.set_xlabel(f"{img_path}")
            
    plt.savefig(f"{plot_title}.png")
    plt.close()


# expected NuMu CC with 100% confidence but actually NC
bad_images = df.loc[(df[f"{bins[-1]}"] == 1) & (df["actual"] == 0)]['location'].tolist()
bad_images = [i.replace(" ", "") for i in bad_images]
bad_images_truth = [0] * len(bad_images)

example_plots(bad_images, bad_images_truth, 0, 1, "test")


def locate_and_plot(predicted, truth, confidence_val = 0.5):
    bad_images = df.loc[(df[f"confidence {predicted}"] >= confidence_val) & (df["actual"] == truth)]['location'].tolist()
    actual_size = len(bad_images)
    bad_images = [i.replace(" ", "") for i in bad_images]
    if actual_size < 11:
        sample_size = actual_size
    else:
        sample_size = 10
    
    bad_images = bad_images[:sample_size]
    bad_images_truth = [truth] * len(bad_images)

    example_plots(bad_images, bad_images_truth, predicted, truth, f"Predicted_{label_map[predicted]}_Truth_{label_map[truth]}_at_{confidence_val}_confidence_total_{actual_size}")

conf = 0.99
locate_and_plot(1, 1, confidence_val = conf)
locate_and_plot(0, 0, confidence_val = conf)
locate_and_plot(1, 0, confidence_val = conf)
locate_and_plot(0, 1, confidence_val = conf)
