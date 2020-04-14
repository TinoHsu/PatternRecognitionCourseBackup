from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(123)
variables = ['X', 'Y', 'Z']
X = np.random.random_sample([50, 3])*10
df = pd.DataFrame(X, columns=variables
                        )
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(X.shape[0]):
    xs = X[i,0]
    ys = X[i,1]
    zs = X[i,2]
    ax.scatter(xs, ys, zs, c='black', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, 
                             affinity='euclidean', 
                             linkage='complete')
cluster_labels = ac.fit_predict(X)
print('Cluster labels: %s' % cluster_labels)

fig = plt.figure()
ab = fig.add_subplot(111, projection='3d')
typemap = [('r', 'o'), ('g', 'o'), ('b', 'o')]
for i in range(X.shape[0]):
    cK = typemap[cluster_labels[i]][0]
    mK = typemap[cluster_labels[i]][1]
    xs = X[i,0]
    ys = X[i,1]
    zs = X[i,2]
    ab.scatter(xs, ys, zs, c=cK, marker=mK)
ab.set_xlabel('X Label')
ab.set_ylabel('Y Label')
ab.set_zlabel('Z Label')
plt.show()


from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

row_clusters = linkage(df.values, method='complete', metric='euclidean')
print(row_clusters)
pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1)
                    for i in range(row_clusters.shape[0])])

row_dendr = dendrogram(row_clusters, 
                       labels=cluster_labels,
                       )
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()