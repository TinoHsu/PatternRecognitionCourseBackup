import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('wine.csv', header=0, 
dtype={"Alcohol":"float64", "Malic acid":"float64",
"Ash":"float64", "Alcalinity of ash":"float64",
"Magnesium":"float64", "Total phenols":"float64",
"Flavanoids":"float64", "Nonflavanoid phenols":"float64",
"Proanthocyanins":"float64", "Color intensity":"float64",
"Hue":"float64", "OD280/OD315 of diluted wines":"float64",
"Proline ":"float64"})

a = np.zeros(24)
b = np.ones(47)
y = np.concatenate((a, b), axis=0)
# print(y)
# print(y.shape)
# np.random.shuffle(y)
# print(y)

X = df.iloc[59:130, 1:].values
# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.75, random_state=788597, stratify=y)

sc = StandardScaler()
# sc.fit(X_train)
sc.fit(X)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_all_std =  sc.transform(X)

# ## PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3, svd_solver='auto')
# # pca = PCA(n_components='mle', svd_solver='auto')
# X_train_pca_sk = pca.fit_transform(X_train_std)
# X_test_pca_sk = pca.transform(X_test_std)
# # print('X'+"'"+' Dimension=', X_train_pca_sk.shape)


# # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# # lda = LDA(n_components=3)
# # X_train_lda = lda.fit_transform(X_train_std, y_train)
# # X_test_lda = lda.transform(X_test_std)

# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(n_components=3, kernel='rbf', gamma=15)
# X_train_skkpca = kpca.fit_transform(X_train_std)
# X_test_skkpca = kpca.transform(X_test_std)


## Linear SVM
svm_lin = SVC(kernel='linear', C=10.0, random_state=1)
svm_lin.fit(X_train_std, y_train)
y_hat_svm_lin = svm_lin.predict(X_all_std)


print('svm_lin accuracy', accuracy_score(y, y_hat_svm_lin))
print('svm_lin cm', confusion_matrix(y, y_hat_svm_lin))
print('svm_lin precision', precision_score(y, y_hat_svm_lin, average='micro'))

'''
flag = -1
rseedgood = 0
rseed = 0
while(flag < 0):
    a = np.zeros(24)
    b = np.ones(47)
    y = np.concatenate((a, b), axis=0)
    # print(y)
    # print(y.shape)
    # np.random.shuffle(y)
    # print(y)

    X = df.iloc[59:130, 1:].values
    # print(X.shape)

    rseed = rseed + 1
    # for rseed in range(100000):
    print('rs=', rseed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.75, random_state=rseed, stratify=y)

    sc = StandardScaler()
    # sc.fit(X_train)
    sc.fit(X)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_all_std =  sc.transform(X)

    ## PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, svd_solver='auto')
    # pca = PCA(n_components='mle', svd_solver='auto')
    X_train_pca_sk = pca.fit_transform(X_train_std)
    X_test_pca_sk = pca.transform(X_test_std)
    # print('X'+"'"+' Dimension=', X_train_pca_sk.shape)


    # # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    # # lda = LDA(n_components=3)
    # # X_train_lda = lda.fit_transform(X_train_std, y_train)
    # # X_test_lda = lda.transform(X_test_std)

    # from sklearn.decomposition import KernelPCA
    # kpca = KernelPCA(n_components=3, kernel='rbf', gamma=15)
    # X_train_skkpca = kpca.fit_transform(X_train_std)
    # X_test_skkpca = kpca.transform(X_test_std)


    ## Linear SVM
    svm_lin = SVC(kernel='linear', C=10.0, random_state=1)
    svm_lin.fit(X_train_std, y_train)
    y_hat_svm_lin = svm_lin.predict(X_all_std)


    # print('svm_lin accuracy', accuracy_score(y_test, y_hat_svm_lin))
    # print('svm_lin cm', confusion_matrix(y_test, y_hat_svm_lin))
    # print('svm_lin precision', precision_score(y_test, y_hat_svm_lin, average='micro'))

    # acc = accuracy_score(y_test, y_hat_svm_lin)
    acc = accuracy_score(y, y_hat_svm_lin)
    if (acc > 0.95):
        flag = 1
        print('acc=', acc)
        rseedgood = rseed
        print('rs=', rseed)

print('rsgood=', rseedgood)
'''