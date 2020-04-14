import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns


## Read wine data
df = pd.read_csv('wine.csv', header=0, 
dtype={"Alcohol":"float64", "Malic acid":"float64",
"Ash":"float64", "Alcalinity of ash":"float64",
"Magnesium":"float64", "Total phenols":"float64",
"Flavanoids":"float64", "Nonflavanoid phenols":"float64",
"Proanthocyanins":"float64", "Color intensity":"float64",
"Hue":"float64", "OD280/OD315 of diluted wines":"float64",
"Proline ":"float64"})


## Using all features
y = df.iloc[:, 0].values
print(y.shape)
# print(y)
X = df.iloc[:, 1:].values
print(X.shape)
# print(X)

## Preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=3, svd_solver='randomized')
# X_train_pca_sk = pca.fit_transform(X_train_std)
# X_test_pca_sk = pca.transform(X_test_std)
# print('X'+"'"+' Dimension=', X_train_pca_sk.shape)


## Linear SVM
svm_lin = SVC(kernel='linear', C=10.0, random_state=1)
svm_lin.fit(X_train_std, y_train)
y_hat_svm_lin = svm_lin.predict(X_test_std)

print('svm_lin cm', confusion_matrix(y_test, y_hat_svm_lin))
print('svm_lin precision', precision_score(y_test, y_hat_svm_lin, average='micro'))


## Kernel SVM
svm_k = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm_k.fit(X_train_std, y_train)
y_hat_svm_k = svm_k.predict(X_test_std)

print('svm_k cm', confusion_matrix(y_test, y_hat_svm_k))
print('svm_k precision', precision_score(y_test, y_hat_svm_k, average='micro'))


## KNN
knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)
y_hat_knn = knn.predict(X_test_std)

print('knn cm', confusion_matrix(y_test, y_hat_knn))
print('knn precision', precision_score(y_test, y_hat_knn, average='micro'))


## Decision tree
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
tree.fit(X_train, y_train)
y_hat_tree = tree.predict(X_test)

print('tree cm', confusion_matrix(y_test, y_hat_tree))
print('tree precision', precision_score(y_test, y_hat_tree, average='micro'))

## Random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=100, 
                                random_state=1,
                                n_jobs=4)
forest.fit(X_train, y_train)
y_hat_forest = forest.predict(X_test)

print('forest cm', confusion_matrix(y_test, y_hat_forest))
print('forest precision', precision_score(y_test, y_hat_forest, average='micro'))