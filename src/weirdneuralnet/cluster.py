from sklearn import cluster
from sklearn.neighbors import NearestCentroid
import cupy as np

# TODO: investigate implementing these in cupy rather than using sklearn


def kmeans(vec, num_clusters):
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=10, max_iter=300).fit(
        vec.get()
    )
    return np.array(kmeans.labels_), np.array(kmeans.cluster_centers_)


# see: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering_metrics.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-metrics-py
def agglom(vec, num_clusters, link_type):
    # for linkage in ("ward", "average", "complete", "single"):
    agglo = cluster.AgglomerativeClustering(
        linkage=link_type, n_clusters=num_clusters)
    v = vec.get()
    y_predict = agglo.fit_predict(v)
    clf = NearestCentroid()
    clf.fit(v, y_predict)
    return np.array(agglo.labels_), np.array(clf.centroids_)


CLUSTER_FUNCS = {
    "kmeans": lambda x, n: kmeans(x, n),
    "agglom ward": lambda x, n: agglom(x, n, "ward"),
    "agglom average": lambda x, n: agglom(x, n, "average"),
    "agglom complete": lambda x, n: agglom(x, n, "complete"),
    "agglom single": lambda x, n: agglom(x, n, "single"),
}


def centroids(X, labels, num_classes):
    centroids = np.zeros((X.shape[0], num_classes))
    for cls in range(num_classes):
        centroids[cls, :] = X[labels == cls, :].mean(axis=0)
    return centroids


# def calc_distances(X):
#     distances = np.empty(X.shape)
#     for i in range(X.shape[0]):
#         distances[i, :] = np.abs(np.broadcast_to(X[i, :], X.shape) - X).sum(axis=1)
#     return distances


def calc_distances(p0, points):
    return np.square(p0 - points).sum(axis=1)


def get_furthest(p0, points, num_get):
    return np.argpartition(calc_distances(p0, points), 1)[-1 * num_get:].astype(int)


# TODO: impl a better ways to get edge points
def get_far_points(centroids, point_set, point_labels, num_far=1):
    edge_points = np.zeros(len(centroids) * num_far)
    for idx, centroid in enumerate(centroids):
        print(f"index: {idx}")
        edge_points[idx: idx + num_far] = get_furthest(
            centroid, point_set[point_labels == idx], num_far
        )
    return edge_points.astype(int)


####
# under construction
####
def optics(vec):
    optic = cluster.OPTICS(min_samples=vec.shape[0] + 1).fit(vec.get())
    return np.array(optic.lables_), np.nonzero(np.array(optic.predecessor_) == -1)


def dbscan(vec):
    # because dbscan clusters can be of any shape, computing class centroids does not
    # make sense. Instead, there are 3 classes of point: core, border, and oulier.
    # To satisfy my strategy, I'll need to use border points & a subset of core points.
    # TODO: optimize parameters: http://www.sefidian.com/2020/12/18/how-to-determine-epsilon-and-minpts-parameters-of-dbscan-clustering/
    # TODO: also return core & border indices
    db = cluster.DBSCAN(eps=0.3, min_samples=vec.shape[0] + 1).fit(vec)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_

    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    return db.labels_, db.core_sample_indices_
