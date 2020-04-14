import numpy as np
import time
import math

from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


## Some setting
learning_rate = 0.01
epoch_time = 50
batch = 18
pic_size = 64


class ScratchNet(nn.Module):
    def __init__(self):
        super(ScratchNet, self).__init__()
        self.hidden1 = nn.Linear(12288, 500)
        self.hidden2 = nn.Linear(500, 500)
        self.hidden3 = nn.Linear(500, 50)
        self.output = nn.Linear(50, 2)
 
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.selu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        
        return x


## transform method
transform = T.Compose([
    T.Resize(pic_size), 
    T.CenterCrop(pic_size), 
    T.ToTensor(), 
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) 
])

## Read data
train_dataset = ImageFolder('D:/cat_set2/train', transform=transform)
print(train_dataset.class_to_idx)
print('train set num', len(train_dataset.imgs))
print(train_dataset[0][0].size())

train_loader = DataLoader(train_dataset, batch_size=batch, 
                    shuffle=True, num_workers=0, 
                    drop_last=False)

val_dataset = ImageFolder('D:/cat_set2/val', transform=transform)
# print(val_dataset.class_to_idx)
print('validation set num', len(val_dataset.imgs))
# print(val_dataset[0][0].size())
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset.imgs), 
                    shuffle=True, num_workers=0, 
                    drop_last=False)

test_dataset = ImageFolder('D:/cat_set2/tset', transform=transform)
# print(test_dataset.class_to_idx)
print('test set num', len(test_dataset.imgs))
# print(test_dataset[0][0].size())
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset.imgs), 
                    shuffle=True, num_workers=0, 
                    drop_last=False)

## Initial model
model = ScratchNet()
print(model)

optimizer = t.optim.SGD(model.parameters(), lr = learning_rate)
loss_func = t.nn.CrossEntropyLoss()

## Start training
print('star training!')
tic = time.time() 

for epoch in range(epoch_time):
    ## record loss   
    loss_average = np.zeros(1)
    for step, (batch_x, batch_y) in enumerate(train_loader):         
        #print('step=', step)
        #print(batch_x.size())
        batch_x = batch_x.view(-1, 12288)
        prediction = model(batch_x)
        loss = loss_func(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cpu = loss.cpu().data.numpy()
        loss_average = np.add(loss_average, loss_cpu/batch)
    
    print('Epoch=', epoch)
    print('Loss=%.4f' % loss_average)

toc = time.time() 
print('train time: ' + str((toc - tic)) + 'sec')
print('training ok!')



for step, (batch_x, batch_y) in enumerate(test_loader):         
    #print('step=', step)
    #print(batch_x.size())
    batch_x = batch_x.view(-1, 12288)
    y_hat_tensor = model(batch_x)
    y_hat_test = y_hat_tensor.data.numpy()     
    y_hat_test  = np.argmax(y_hat_test, axis=1)
    y_test = batch_y.data.numpy()
    print(y_test)
    print(y_hat_test)
    print("Test set score: %f" % accuracy_score(y_test, y_hat_test))


    