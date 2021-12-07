

from sklearn import cluster
from sklearn.neighbors import NearestCentroid
import cupy as np

def kmeans(vec, num_clusters):
    #TODO: need to find border points too
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=10, max_iter=300).fit(vec)
    return kmeans.labels_, kmeans.cluster_centers_


#see: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering_metrics.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-metrics-py
def agglom(vec, num_clusters, link_type):
    #for linkage in ("ward", "average", "complete", "single"):
    agglo = cluster.AgglomerativeClustering(linkage=link_type, n_clusters=num_clusters)
    y_predict = agglo.fit_predict(vec)
    clf = NearestCentroid()
    clf.fit(vec, y_predict)
    #TODO: is this returning indices or vectors for centroids? need to standardize
    # across clustering funcs
    return agglo.labels_, clf.centroids_

def optics(vec):
    optic = cluster.OPTICS(min_samples=vec.shape[0]+1).fit(vec)
    #TODO: also return border indices
    return optic.lables_, np.nonzero(optic.predecessor_==-1)

def dbscan(vec):
    # because dbscan clusters can be of any shape, computing class centroids does not
    # make sense. Instead, there are 3 classes of point: core, border, and oulier.
    # To satisfy my strategy, I'll need to use all border points & some subset of 
    # the core points.
    #TODO: optimize parameters: http://www.sefidian.com/2020/12/18/how-to-determine-epsilon-and-minpts-parameters-of-dbscan-clustering/
    #TODO: also return core & border indices
    db = cluster.DBSCAN(eps=0.3, min_samples=vec.shape[0]+1).fit(vec)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_

    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    return db.labels_, db.core_sample_indices_

def centroids(X, labels, num_classes):
    num_features = X.shape[0]
    centroids = np.zeros((num_features, num_classes))
    for cls in range(num_classes):
        centroids[:, cls] = X[:,np.nonzero(labels==cls)[0]].mean(axis=1)
    return centroids

def calc_distances(X):
    distances = np.empty(X.shape)
    for i in range(X.shape[1]):
        distances[:,i] = np.abs(np.broadcast_to(X[:,i], X.T.shape).T - X).sum(axis=0)
    return distances