import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

def plot_clu_result(X_data, clu_y):

    plt.scatter(X_data[clu_y == 0, 0], X_data[clu_y == 0, 1],
                c='lightblue', marker='o', s=40,
                edgecolor='black', 
                label='cluster 1')
    plt.scatter(X_data[clu_y == 1, 0], X_data[clu_y == 1, 1],
                c='r', marker='o', s=40,
                edgecolor='black', 
                label='cluster 2')
    plt.scatter(X_data[clu_y == 2, 0], X_data[clu_y == 2, 1],
                c='g', marker='o', s=40,
                edgecolor='black',
                label='cluster 3')
    plt.scatter(X_data[clu_y == 3, 0], X_data[clu_y == 3, 1],
                c='b', marker='o', s=40,
                edgecolor='black', 
                label='cluster 4')
    plt.scatter(X_data[clu_y == 4, 0], X_data[clu_y == 4, 1],
                c='purple', marker='o', s=40,
                edgecolor='black', 
                label='cluster 5')
    plt.scatter(X_data[clu_y == 5, 0], X_data[clu_y == 5, 1],
                c='y', marker='o', s=40,
                edgecolor='black', 
                label='cluster 6')
    plt.legend()
    plt.tight_layout()
    plt.show()


## plot data
X = np.load('clusterable_data.npy')
plt.scatter(X.T[0], X.T[1], c='b')
plt.tight_layout()
plt.show()


## kmeans 
from sklearn.cluster import KMeans
#...
#y_km = ...
# plot result
plot_clu_result(X, y_km)


## kmeans++
#...

## Agglomerative 
#...

## DBSCAN
#...







