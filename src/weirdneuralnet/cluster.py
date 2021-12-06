

from sklearn import cluster
import cupy as np

def kmeans(vec, num_clusters):
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=10, max_iter=300).fit(vec)
    return kmeans.labels_, kmeans.cluster_centers_


#see: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering_metrics.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-metrics-py
def agglom(vec, num_clusters, link_type):
    #for linkage in ("ward", "average", "complete", "single"):
    clustering = cluster.AgglomerativeClustering(linkage=link_type, n_clusters=num_clusters)
    clustering.fit(vec)

    return clustering.labels_

def dbscan(vec):
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

#TODO: need to play with this, vectorize properly with labels
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length