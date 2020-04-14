import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import time


# Implementing an adaptive linear neuron in Python
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression (as we will see later), 
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


def decision_measure(prediction, label, method='for_if'):

  num_tp = 0
  num_tn = 0
  num_fp = 0
  num_fn = 0

  if method == 'for_if':
    for i in range(label.shape[0]):
      if label[i] == 1:
        if prediction[i] == 1:
          num_tp = num_tp + 1
        if prediction[i] == -1:
          num_fn = num_fn + 1
      if label[i] == -1:
        if prediction[i] == 1:
          num_fp = num_fp + 1
        if prediction[i] == -1:
          num_tn = num_tn + 1
  
  if method == 'mat_operat':
      comparison = np.concatenate((label.reshape(label.shape[0], 1), prediction.reshape(label.shape[0], 1)), axis =1)
      variety = np.array([[1], [2]])
      result = np.dot(comparison, variety)
      num_tp = label.shape[0] - np.count_nonzero(result-3)
      num_tn = label.shape[0] - np.count_nonzero(result+3)
      num_fp = label.shape[0] - np.count_nonzero(result-1)
      num_fn = label.shape[0] - np.count_nonzero(result+1)

  acc = (num_tp+num_tn)/(num_tp+num_tn+num_fp+num_fn)
  # pre = num_tp/(num_tp+num_fp)
  # ...
  return num_tp, num_tn, num_fp, num_fn, acc


def k_fold_CV(X, y, num_iter, time_kfCV):

  order = np.arange(y.shape[0])
  half_test_len = int(y.shape[0]/time_kfCV/2)
  
  pre_test = order[num_iter*half_test_len : (num_iter+1)*half_test_len]
  post_test = order[50+num_iter*half_test_len : 50+(num_iter+1)*half_test_len]
  
  test_ord = np.concatenate((pre_test, post_test))
  # print(test_ord)
  train_ord = np.delete(order, [test_ord])
  # print(train_ord)

  return train_ord, test_ord


# Reading-in the Iris data
df = pd.read_csv('iris.csv', header=None)

# select setosa and versicolor
y = df.iloc[1:101, 4].values
y = np.where(y == 'setosa', -1, 1)
# print('y', y)
# extract sepal length and petal length
X = df.iloc[1:101, [0, 2]].values
# change str to float
X = X.astype(np.float)

ada1 = AdalineGD(n_iter=13, eta=0.0001)
ada1.fit(X, y)
y_hat = ada1.predict(X)
# print(y)
# print(y_hat)


#Q1
tp, tn, fp, fn, acc = decision_measure(y_hat, y, method='mat_operat')
print('TP=', tp)
print('FN=', fn)
print('TN=', tn)
print('FP=', fp)
print('ACC=', acc)
print('\n')

#sklearn
from sklearn.metrics import confusion_matrix
cmtn, cmfp, cmfn, cmtp  = confusion_matrix(y_true=y, y_pred=y_hat).ravel()
print('confmat TN FP FN TP =', cmtn, cmfp, cmfn, cmtp)
print('\n')

#Q2&3
time_kfCV = 5
for kf_iter in range(time_kfCV):

  print('K=', kf_iter+1)

  train_fold_order, test_fold_order = k_fold_CV(X, y, kf_iter, time_kfCV)
  X_train = X[train_fold_order]
  y_train = y[train_fold_order]
  X_test = X[test_fold_order]
  y_test = y[test_fold_order]

  ada2 = AdalineGD(n_iter=16, eta=0.0001)
  ada2.fit(X_train, y_train)
  y_hat_test = ada2.predict(X_test)

  tp, tn, fp, fn, acc = decision_measure(y_hat_test, y_test, method='mat_operat')

  print('TP=', tp)
  print('FN=', fn)
  print('TN=', tn)
  print('FP=', fp)
  print('ACC=', acc)