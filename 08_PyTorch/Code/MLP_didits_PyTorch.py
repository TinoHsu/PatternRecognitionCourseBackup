import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch as t
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader

## Check CPU
cpu = t.device('cpu')
## Check GPU
if cuda.is_available():
    cuda0 = t.device('cuda:0')
    
## Some setting
learning_rate = 0.1
epoch_time = 50
batch = 128


## MLP model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(784, 100)
        self.hidden2 = nn.Linear(100, 100)
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        x = self.output(x)
    
        return x


## Load dataset
df = pd.read_csv('mnist_784.csv', header=0)
y = df.iloc[:, -1].values
print(y.shape)
X = df.iloc[:, 0:-1].values
print(X.shape)


## Preprocessing
X = X / 255.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)
print(X.shape)


## Only have CPU
X_train_tensor = t.tensor(X_train, dtype=t.float, device=cpu)
y_train_tensor = t.tensor(y_train, dtype=t.long, device=cpu)
X_test_tensor = t.tensor(X_test, dtype=t.float, device=cpu)
y_test_tensor = t.tensor(y_test, dtype=t.long, device=cpu)


# With GPU
if cuda.is_available():
    X_train_tensor = t.tensor(X_train, dtype=t.float, device=cuda0)
    y_train_tensor = t.tensor(y_train, dtype=t.long, device=cuda0)
    X_test_tensor = t.tensor(X_test, dtype=t.float, device=cuda0)
    y_test_tensor = t.tensor(y_test, dtype=t.long, device=cuda0)
    

## Use dataLoader
torch_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
loader = DataLoader(dataset=torch_dataset, batch_size=batch, 
                    shuffle=True, num_workers=0)


## Initial model
model = Net()
if cuda.is_available():
    model = Net().cuda()
print(model)

optimizer = t.optim.SGD(model.parameters(), lr = learning_rate)
loss_func = t.nn.CrossEntropyLoss()


## Start training
print('star training!')
tic = time.time() 

for epoch in range(epoch_time):
    ## record loss   
    loss_average = np.zeros(1)
    for step, (batch_x, batch_y) in enumerate(loader):         
        # print('step=', step)
        optimizer.zero_grad()
        prediction = model(batch_x)
        loss = loss_func(prediction, batch_y)
        loss.backward()
        optimizer.step()
        loss_cpu = loss.cpu().data.numpy()
        loss_average = np.add(loss_average, loss_cpu/batch)
    if epoch % 5 == 0:
        print('Epoch=', epoch)
        print('Loss=%.4f' % loss_average)

toc = time.time() 
print('train time: ' + str((toc - tic)) + 'sec')
print('training ok!')


## Testing predict
y_test_hat_tensor = model(X_test_tensor)

if cuda.is_available():
    y_test_hat = y_test_hat_tensor.cpu().data.numpy()
else:
    ## Only have CPU
    y_test_hat = y_test_hat_tensor.data.numpy()

## change to index 
y_test_hat = np.argmax(y_test_hat, axis=1)
print(y_test_hat)
print(y_test)
print("Test set score: %f" % accuracy_score(y_test, y_test_hat))

