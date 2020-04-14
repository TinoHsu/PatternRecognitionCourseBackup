import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch as t
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader

'''
## Tensorboard_logger 
from tensorboard_logger import configure, log_value
#Tensorboard_logger setting
configure("runs", flush_secs=2)
# cmd:tensorboard --logdir runs
'''
cpu = t.device('cpu')
## Check GPU
if cuda.is_available():
    cuda0 = t.device('cuda:0')
    
## Some setting
learning_rate = 0.0001
epoch_time = 100
batch = 125
# except_length == bptt_timestep
except_length = 150
# k dimensional embeddings
embed_dim = 200

## RNN model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()   
        self.rnn = nn.LSTM(     
            input_size=embed_dim,      
            hidden_size=500,     
            num_layers=1,       
            batch_first=True,
            # True (batch, time_step, input_size)
            # dropout = 0.5
        )
        # weight initial
        nn.init.orthogonal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        self.fc = nn.Linear(500, 2)
        # self.fc1 = nn.Linear(500, 100)
        # self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        rnn_out, (hn, cn) = self.rnn(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   
        # c_n shape (n_layers, batch, hidden_size)
        out = self.fc(rnn_out[:, -1, :])
        # Pick the the last time of r_out output
        # fc1_out = F.relu(self.fc1(rnn_out[:, -1, :]))
        # out = self.fc2(fc1_out)
        
        return out

## Weights Initialization
def init_weights(m):
    if type(m) == nn.Linear:
        t.nn.init.kaiming_uniform_(m.weight, mode='fan_in')


def wordEmbed2Matrix(docs, except_length, word_to_ix, embeds, embed_dim):

    doc_list = []
    for i in range(docs.shape[0]):
        stance = docs[i].split( )
        temp = []  

        for j in range(len(stance)):    
            stance[j] = stance[j].lower()
            if stance[j] in word_to_ix:
                embed = embeds(t.tensor([word_to_ix[stance[j]]], dtype=t.long)).data.numpy()
                #print(embed.shape)
                temp.append(embed.reshape(-1))       
        
        if len(temp) < except_length:
            temp.reverse()
            for j in range(len(temp), except_length):
                embed = np.zeros(embed_dim)
                temp.append(embed.reshape(-1))
            temp.reverse()

        if len(temp) > except_length:
            del temp[except_length:]

        doc_list.append(np.array(temp))

    p = np.array(doc_list)

    return p

## Load dataset
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df_test = pd.read_csv('movie_data_test.csv', encoding='utf-8')
y = df.iloc[:, 0].values
X = df.iloc[:, 1].values
X_test = df_test.iloc[:, 0].values
print('y', y.shape)
print('X', X.shape)
print('X_test', X_test.shape)
# print('sentiment:', y[89])
# print('review:', X[89])
X_combine = np.hstack((X, X_test))
print('X_combine', X_combine.shape)


## Preprocessing
vectorizer = CountVectorizer(stop_words="english", max_features=50000)
# vectorizer = CountVectorizer(stop_words="english")
bow = vectorizer.fit_transform(X_combine)
print('BOW', bow.shape)
word_to_ix = vectorizer.vocabulary_
# print(vectorizer.vocabulary_)


## Word embedding
embeds = nn.Embedding(bow.shape[1], embed_dim)# n words in vocab, k dimensional embeddings
X = wordEmbed2Matrix(X, except_length, word_to_ix, embeds, embed_dim)
X_test = wordEmbed2Matrix(X_test, except_length, word_to_ix, embeds, embed_dim)
# shape (X.shape[0], except_length, embed_dim)


## split daya to train and val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.05, random_state=1, stratify=y)

print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('X_val', X_val.shape)


## Only have CPU
X_train_tensor = t.tensor(X_train, dtype=t.float, device=cpu)
y_train_tensor = t.tensor(y_train, dtype=t.long, device=cpu)
X_test_tensor = t.tensor(X_test, dtype=t.float, device=cpu)
y_fiktion = t.tensor(y, dtype=t.long, device=cpu)
X_val_tensor = t.tensor(X_val, dtype=t.float, device=cpu)
y_val_tensor = t.tensor(y_val, dtype=t.long, device=cpu)


## With GPU
if cuda.is_available():
    X_train_tensor = t.tensor(X_train, dtype=t.float, device=cuda0)
    y_train_tensor = t.tensor(y_train, dtype=t.long, device=cuda0)
    X_test_tensor = t.tensor(X_test, dtype=t.float, device=cuda0)
    y_fiktion = t.tensor(y, dtype=t.long, device=cuda0)
    X_val_tensor = t.tensor(X_val, dtype=t.float, device=cuda0)
    y_val_tensor = t.tensor(y_val, dtype=t.long, device=cuda0)


## Use dataLoader
torch_dataset_train = data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=torch_dataset_train, batch_size=batch, 
                    shuffle=True, num_workers=0)

torch_dataset_test = data.TensorDataset(X_test_tensor, y_fiktion)
test_loader = DataLoader(dataset=torch_dataset_test, batch_size=1250, 
                    shuffle=False, num_workers=0)


## Initial model
model = RNN()
model.apply(init_weights)
if cuda.is_available():
    model = RNN().cuda()
    print(model)

optimizer = t.optim.Adam(model.parameters(), lr = learning_rate)
loss_func = t.nn.CrossEntropyLoss()


## Start training
print('star training!')
tic = time.time() 

for epoch in range(epoch_time):
    ## record loss 
    train_loss_average = np.zeros(1)
    model.train() 
    for step, (batch_x, batch_y) in enumerate(train_loader):         
        # print('step=', step)

        optimizer.zero_grad()
        prediction = model(batch_x)
        loss = loss_func(prediction, batch_y)
        loss.backward()
        optimizer.step()

        loss_cpu = loss.cpu().data.numpy()
        train_loss_average = np.add(train_loss_average, loss_cpu/batch)
    
    # log_value('trainloss', train_loss_average, epoch)
    
    if epoch % 10 == 0:
        print('Epoch=', epoch)
        print('Train Loss=%.7f' % train_loss_average)

toc = time.time() 
print('train time: ' + str((toc - tic)) + 'sec')
print('training ok!')


## validation model 
model.eval() 
y_val_hat_tensor = model(X_val_tensor)

if cuda.is_available():
    y_val_hat = y_val_hat_tensor.cpu().data.numpy()
else:
    ## Only have CPU
    y_val_hat = y_val_hat_tensor.data.numpy()

## change float to index 
y_val_hat = np.argmax(y_val_hat, axis=1)
# print(y_val_hat)
# print(y_val)
print("val set score: %f" % accuracy_score(y_val, y_val_hat))
print("val set confusion_matrix:")
print(confusion_matrix(y_val, y_val_hat))


## Testing 
model.eval() 
outcome = []
for step, (batch_x, batch_y) in enumerate(test_loader): 
    prediction = model(batch_x)

    if cuda.is_available():
        y_test_hat = prediction.cpu().data.numpy()
    else:
        ## Only have CPU
        y_test_hat = prediction.data.numpy()
    
    y_test_hat = np.argmax(y_test_hat, axis=1)
    # print(y_val)
    b = y_test_hat.tolist()
    for item in b:
        outcome.append(item)

test_out = np.asarray(outcome)
test_out = test_out.T
print(test_out.shape)

df_Submission = pd.read_csv('sampleSubmission.csv', encoding='utf-8')
df_Submission = df_Submission.drop(['sentiment'], axis=1)
df_test_out = pd.DataFrame(test_out, columns=['sentiment'])
df_out = pd.concat([df_Submission, df_test_out], axis=1)
df_out.to_csv('sampleSubmission.csv', index=False, encoding='utf-8')
    