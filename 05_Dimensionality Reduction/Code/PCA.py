import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import f1_score

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

## load data
iris = datasets.load_iris()
X = iris.data[:, [0, 1, 2, 3]]
y = iris.target
# wine = datasets.load_wine()
# X = wine.data[:, 1:]
# y = wine.target
## split data to train and tset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

## Steps to perform PCA
# standardize the dataset.
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# construct the covariance matrix
cov_mat = np.cov(X_train_std.T)
# decompose the covariance matrix 
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)
print('\nEigenvectors \n%s' %eigen_vecs)
# construct a projection matrix W
w = np.column_stack((eigen_vecs[:, i]
                        for i in range(2)))
print('Matrix W:\n', w)
# transform X' = X．W
X_train_pca = X_train_std.dot(w)
X_test_pca = X_test_std.dot(w)

## plot X_train_pca
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

## Using the result of PCA to classifier
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
# plot result 
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
# plot test result 
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
# F_1 score
y_hat = lr.predict(X_test_pca)
f1 = f1_score(y_test, y_hat, average='micro') 
print('f1 score (PCA) =', "%.2f" % f1)


## Using SK to perform PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='randomized')
# pca = PCA(n_components='mle', svd_solver='auto')
X_train_pca_sk = pca.fit_transform(X_train_std)
X_test_pca_sk = pca.transform(X_test_std)
print('X'+"'"+' Dimension=', X_train_pca_sk.shape)
lr_sk = LogisticRegression()
lr_sk = lr_sk.fit(X_train_pca_sk, y_train)
y_hat_sk = lr_sk.predict(X_test_pca_sk)
f1_sk = f1_score(y_test, y_hat_sk, average='micro') 
print('f1 score (SK PCA)=', "%.2f" % f1_sk)

## Using all feature and compare
lr_all_fea = LogisticRegression()
lr_all_fea = lr_all_fea.fit(X_train_std, y_train)

y_hat_all_fea = lr_all_fea.predict(X_test_std)
f1_all_fea = f1_score(y_test, y_hat_all_fea, average='micro') 
print('f1 score(All feature)=', "%.2f" % f1_all_fea)




