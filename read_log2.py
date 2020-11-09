import os
import re
import array
import matplotlib.pyplot as plt
import numpy as np


TrainLoss = []
TrainAcc1 =[]
TrainAcc5 = []
ValLoss = []
ValAcc1 =[]
ValAcc5 = []
names = ['Train Loss: ','Train Acc 1:  ','Train Acc 5:  ','Valid Loss: ','Valid Acc 1:  ','Valid Acc 5:  ']
titles = ['Training Loss','Top-1 Training Accuracy ','Top-5 Training Accuracy','Validation Loss','Top-1 Validation Accuracy','Top-5 Validation Accuracy']
nb = 29
epch = 20
models = np.arange(nb)
epochs = np.arange(epch)

t=0
for n in names:
    i=-1
    plt.figure()
    for filename in os.listdir('.'):
        if filename.endswith('.log'):
            with open(os.path.join('.', filename)) as f:
                i=i+1
                TrainLoss=np.empty((nb, epch))
                content = f.read()
                #print(i)
                #print(filename)
                str = re.compile(n+"[0-9]+\.[0-9]+")
                tmp = (str.findall(content))
                #print(tmp)
                for j, val in enumerate(tmp):
                    TrainLoss[i][j] = (val[len(n):])
                
                plt.plot(epochs, TrainLoss[i],label=models[i])
                #print(TrainLoss[i])
                #usernames = re.findall(r"Train Loss:\.(.*?)| Train Acc", content)
                #'Part 1\.(.*?)Part 3'


    plt.xlabel('Number of epochs')
    plt.title(titles[t])
    plt.legend()
    #plt.show()
    plt.savefig(titles[t]+'.png',dpi=200)
    t=t+1