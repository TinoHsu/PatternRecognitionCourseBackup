import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

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
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


def spiral(seed=1984):

    np.random.seed(seed)
    n = 200  
    dim = 2  
    class_num = 3  

    x = np.zeros((n*class_num, dim))
    t = np.zeros(n*class_num, dtype=np.int)

    for j in range(class_num):
        for i in range(n):
            rate = i / n
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = n*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix] = j

    return x, t


## load spiral  
X, y = spiral()
print('X', X.shape) 
print('y', y.shape)  


## plot spiral
point_n = 200
class_n = 3
markers = ['o', 'x', '^']
for i in range(class_n):
    plt.scatter(X[i*point_n:(i+1)*point_n, 0], X[i*point_n:(i+1)*point_n, 1], s=40, marker=markers[i])
plt.show()


## Standardized
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)


## classifier SVM
svm = SVC(kernel='linear', C=1.0, max_iter=-1, 
                     tol=1e-4, verbose=False, random_state=1)
svm.fit(X_std, y)
print('SVM')
print("Training set score: %f" % svm.score(X_std, y))
plot_decision_regions(X=X_std, y=y,
                      classifier=svm,)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()



from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
                    learning_rate_init=0.1)
mlp.fit(X_std, y)
print('MLP')
print("Training set score: %f" % mlp.score(X_std, y))
plot_decision_regions(X=X_std, y=y,
                      classifier=mlp,)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
