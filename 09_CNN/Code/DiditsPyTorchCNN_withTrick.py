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
learning_rate = 0.001
epoch_time = 50
batch = 128


## CNN model
class LeNet_modify(nn.Module):
    def __init__(self):
        super(LeNet_modify, self).__init__()
        self.prelu = nn.PReLU()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(800, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = F.max_pool2d(x, (2, 2))   
        x = F.max_pool2d(self.prelu(self.bn2(self.conv2(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout1(self.prelu(self.bn3(self.fc1(x))))
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


## Weights Initialization
def init_weights(m):
    if type(m) == nn.Linear:
        t.nn.init.kaiming_uniform_(m.weight, mode='fan_in')
    if type(m) == nn.Conv2d:
        t.nn.init.kaiming_uniform_(m.weight, mode='fan_in')


## Initial model
model = LeNet_modify()
model.apply(init_weights)
if cuda.is_available():
    model = LeNet_modify().cuda()
    print(model)

optimizer = t.optim.Adam(model.parameters(), lr = learning_rate)
loss_func = t.nn.CrossEntropyLoss()


## Start training
print('star training!')
tic = time.time() 

for epoch in range(epoch_time):
    ## record loss   
    loss_average = np.zeros(1)
    model.train()
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
model.eval() 
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

