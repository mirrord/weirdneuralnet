
from .node_utils import binarize, debinarize
from .cluster import *
import cupy as np

#TODO: these should return matrices and/or indices, not training!

#NOTE: this is here for posterity. It sucks big time!
def prime_typea(network, X, cluster_type, num_classes, epochs, batch_size=-1):
    # priming A: pretrain with blind clusters
    clust = CLUSTER_FUNCS[cluster_type]
    lables, _ = clust(X, num_classes)
    bin_labels = binarize(lables, num_classes)
    return network.train(X, bin_labels, epochs, batch_size)

def prime_typeb(network, X, cluster_type, num_classes, epochs, batch_size=-1):
    # priming B: pretrain with blind cluster centroids & edge points
    #NOTE: at present, we only grab twice the number of classes in points.
    # I should probably parameterize this.
    clust = CLUSTER_FUNCS[cluster_type]
    new_data = np.zeros((num_classes*2, X.shape[1]))
    new_labels = np.zeros(num_classes*2)
    labels, new_data[:num_classes] = clust(X, num_classes)
    new_labels[:num_classes] = np.arange(num_classes)
    new_labels[num_classes:] = np.arange(num_classes)
    #NOTE: I'm using "far" points for now to see how they do.
    # I suspect this will not be good enough.
    edge_idxs = get_far_points(new_data[:num_classes], X, labels)
    new_data[num_classes:] = X[edge_idxs]
    bin_labels = binarize(new_labels, num_classes)
    return network.train(new_data, bin_labels, epochs, batch_size)

def prime_typec(network, X, labels, cluster_type, num_classes, epochs, nested_clusters=3, batch_size=-1):
    # priming C: pretrain with nested cluster centroids & edge points from expected dataset
    clust = CLUSTER_FUNCS[cluster_type]
    new_num_samples = num_classes*nested_clusters*2
    new_data = np.zeros((new_num_samples, X.shape[1]))
    new_labels = np.zeros(new_num_samples)
    labels = debinarize(labels)
    for i in range(num_classes): #for each class...
        #find representative samples (centroids) in that class
        idx = 2*i*nested_clusters
        next_idx = idx+(2*nested_clusters)
        class_samples = X[labels==i]
        nested_labels, nested_centroids = clust(class_samples, nested_clusters)
        new_data[idx:idx+nested_clusters] = nested_centroids
        #and get outliers in those representative samples
        edge_idxs = get_far_points(nested_centroids, class_samples, nested_labels)
        new_data[idx+nested_clusters:next_idx] = class_samples[edge_idxs]
        new_labels[idx:next_idx] = i
    bin_labels = binarize(new_labels, num_classes)
    return network.train(new_data, bin_labels, epochs, batch_size)