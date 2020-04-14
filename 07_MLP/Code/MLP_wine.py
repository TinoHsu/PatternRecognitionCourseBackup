import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

## load data
wine = datasets.load_wine()
X = wine.data[:, 0:]
y = wine.target
# print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

## classifier SVM
svm = SVC(kernel='linear', C=1.0, max_iter=10, 
                     tol=1e-4, verbose=False, random_state=1)
svm.fit(X_train_std, y_train)
print('SVM')
print("Training set score: %f" % svm.score(X_train_std, y_train))
print("Test set score: %f" % svm.score(X_test_std, y_test))


## classifier MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
                    learning_rate_init=0.1)
mlp.fit(X_train_std, y_train)
print('MLP')
print("Training set score: %f" % mlp.score(X_train_std, y_train))
print("Test set score: %f" % mlp.score(X_test_std, y_test))
