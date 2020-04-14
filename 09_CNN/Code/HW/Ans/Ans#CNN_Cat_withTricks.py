import numpy as np
import time
import math

from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torchvision.models as models


import torch as t
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from torchvision.models import alexnet
from torchvision.models import resnet50

## Tensorboard_logger 
from tensorboard_logger import configure, log_value
#Tensorboard_logger setting
configure("runs", flush_secs=2)
# cmd:tensorboard --logdir runs


## release memory
t.cuda.empty_cache()

## Some setting
learning_rate = 0.001
epoch_time = 15
batch = 18 
pic_size = 64
kv_times = 10


class ScartchNet(nn.Module):
    def __init__(self):
        super(ScartchNet, self).__init__()
        #layer1
        self.conv1 = nn.Conv2d(3, 48, 5)
        self.conv1_bn = nn.BatchNorm2d(48)
        #layer2
        self.conv2 = nn.Conv2d(48, 128, 3)
        self.conv2_bn = nn.BatchNorm2d(128)
        #layer3
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        #layer4
        self.fc1 = nn.Linear(128*6*6, 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(p=0.5)
        #layer5
        self.fc2 = nn.Linear(2048, 2048)
        self.fc2_bn = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(p=0.5)
        #layer6
        self.fc3 = nn.Linear(2048, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), 2)
        # print(x.size()[1:])
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2)
        # print(x.size()[1:])
        x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), 2)
        # print(x.size()[1:])
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout1(F.selu(self.fc1_bn(self.fc1(x))))
        x = self.dropout2(F.selu(self.fc2_bn(self.fc2(x))))    
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        # print('num_features', num_features)
        return num_features


## Weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        # t.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        t.nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        # t.nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0))
        # t.nn.init.xavier_normal_(m.weight, gain=1)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        # t.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        t.nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        # t.nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0))
        # t.nn.init.xavier_normal_(m.weight, gain=1)


## transform method
transform = T.Compose([
    T.Resize(pic_size), 
    T.CenterCrop(pic_size), 
    T.ToTensor(), 
    # T.Normalize(mean=[0.485, 0.456, 0.406], 
    #         std=[0.229, 0.224, 0.225])
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
test_loader = DataLoader(test_dataset, batch_size=20, 
                    shuffle=True, num_workers=0, 
                    drop_last=False)


f1_average = np.zeros(1)
for i in range(kv_times):
    ## Initial model
    model = ScartchNet().cuda()
    # model = models.inception_v3(pretrained=True).cuda()
    model.apply(init_weights)
    optimizer = t.optim.ASGD(model.parameters(), lr = learning_rate)
    loss_func = t.nn.CrossEntropyLoss()
    ## Start training
    print('star training!')
    tic = time.time() 

    for epoch in range(epoch_time):
        
        train_loss_average = np.zeros(1)
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):     
            # print('step=', step)
            optimizer.zero_grad()

            if cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            prediction = model(batch_x)
            loss = loss_func(prediction, batch_y)
            loss.backward()
            
            # if t.cuda.device_count() > 1:
            #     optimizer.module.step()
            # else:
            #     optimizer.step()
            
            optimizer.step()
            loss_cpu = loss.cpu().data.numpy()
            train_loss_average = np.add(train_loss_average, loss_cpu)
        train_loss_average = train_loss_average/len(train_dataset.imgs)
        
        val_loss_average = np.zeros(1)
        model.eval() 
        for step, (batch_x, batch_y) in enumerate(val_loader):
            # print('step=', step)
            if cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            
            prediction = model(batch_x)
            loss = loss_func(prediction, batch_y)

            loss_cpu = loss.cpu().data.numpy()
            val_loss_average = np.add(val_loss_average, loss_cpu/len(val_dataset.imgs))
        
        # log_value('loss', loss_average, epoch)
        log_value('trainloss', train_loss_average, epoch)
        log_value('valloss', val_loss_average, epoch)
        
        # if epoch % 1 == 0:
            # print('Epoch=', epoch)
            # print('Train Loss=%.6f' % train_loss_average, ' Valdition Loss=%.6f' % val_loss_average)

    toc = time.time() 
    # print('train time: ' + str((toc - tic)) + 'sec')
    print('training ok!')


    ## Testing predict
    y_test_hat = np.arange(1)
    y_test = np.arange(1)

    model.eval()
    for step, (batch_x, batch_y) in enumerate(test_loader):

        # print('step=', step)
        if cuda.is_available():
            t.cuda.empty_cache()
            batch_x = batch_x.cuda()
        
            test_yhat_tensor = model(batch_x)
            test_yhat = test_yhat_tensor.cpu().data.numpy() 
        else:
            test_yhat_tensor = model(batch_x)
            test_yhat = test_yhat_tensor.data.numpy()
        
        test_yhat = np.argmax(test_yhat, axis=1)

        y_test_hat = np.append(y_test_hat, test_yhat)
        y_test = np.append(y_test, batch_y.data.numpy())

    # print(y_test)
    # print(y_test_hat)
    y_test = np.delete(y_test, 0)
    y_test_hat = np.delete(y_test_hat, 0)

    # print("Test set Accuracy: %f" % accuracy_score(y_test, y_test_hat))
    # print("Test set Recal: %f" % recall_score(y_test, y_test_hat))
    # print("Test set Precision: %f" % precision_score(y_test, y_test_hat))
    print("Test set F1: %f" % f1_score(y_test, y_test_hat))
    print("Test set confusion_matrix:")
    print(confusion_matrix(y_test, y_test_hat))
    f1_average = np.add(f1_average, f1_score(y_test, y_test_hat))
    # tn, fp, fn, tp = confusion_matrix(y_test, y_test_hat).ravel()
    # print(tn, fp, fn, tp)
print('F1 average', f1_average.sum()/kv_times)





