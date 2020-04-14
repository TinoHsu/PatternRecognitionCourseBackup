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
from torchvision.models import resnet34
from torch.utils import data
from torch.utils.data import DataLoader


cpu = t.device('cpu')
## Check GPU
if cuda.is_available():
    cuda0 = t.device('cuda:0')
    
## Some setting
learning_rate = 0.01
epoch_time = 50
batch = 128


## CNN model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 20 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv12 = nn.Conv2d(20, 10, 1)
        self.conv2 = nn.Conv2d(10, 50, 5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv12(x))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
    
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


## Load dataset
df = pd.read_csv('mnist_784.csv', header=0)
y = df.iloc[:, -1].values
print(y.shape)
X = df.iloc[:, 0:-1].values
print(X.shape)


## Preprocessing
X = X.reshape(X.shape[0], 1, 28, 28)

# (num_data, channel, height, width)
X = X / 255.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)
print(X.shape)


## Only have CPU
X_train_tensor = t.tensor(X_train, dtype=t.float, device=cpu)
y_train_tensor = t.tensor(y_train, dtype=t.long, device=cpu)
X_test_tensor = t.tensor(X_test, dtype=t.float, device=cpu)
y_test_tensor = t.tensor(y_test, dtype=t.long, device=cpu)


## With GPU
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
model = LeNet()
if cuda.is_available():
    model = LeNet().cuda()
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

## change float to index 
y_test_hat = np.argmax(y_test_hat, axis=1)
print(y_test_hat)
print(y_test)
print("Test set score: %f" % accuracy_score(y_test, y_test_hat))

