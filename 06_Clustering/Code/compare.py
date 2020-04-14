import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=14)
    plt.text(-0.3, 0.5, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=20)

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

data = np.load('clusterable_data.npy')
## plot data
plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()


## k-means
plot_clusters(data, cluster.KMeans, (), {'n_clusters':6})
plt.tight_layout()
plt.show()

## agglomerative
plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})
plt.tight_layout()
plt.show()

## DBSCAN
plot_clusters(data, cluster.DBSCAN, (), {'eps':0.025})
plt.tight_layout()
plt.show()

# import hdbscan
# plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':15})
