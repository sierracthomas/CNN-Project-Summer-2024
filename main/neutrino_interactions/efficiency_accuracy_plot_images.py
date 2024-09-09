import numpy as np
import matplotlib.pyplot as plt
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
