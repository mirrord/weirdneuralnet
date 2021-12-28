
from .node_utils import binarize, debinarize
from .cluster import *
import cupy as np

# subset selection functions return subset indices
# subset construction functions return (points, labels)

############## Construction functions ##############
#NOTE: prime type A is here for posterity. It sucks big time!
def prime_typea(X, cluster_type, num_classes):
    # priming A: pretrain with blind clusters
    clust = CLUSTER_FUNCS[cluster_type]
    lables, _ = clust(X, num_classes)
    bin_labels = binarize(lables, num_classes)
    return X, bin_labels

def prime_typeb(X, cluster_type, num_classes, num_far_points=1):
    # priming B: pretrain with blind cluster centroids & edge points
    #NOTE: at present, we only grab twice the number of classes in points.
    # I should probably parameterize this.
    num_points_per_class = num_far_points+1
    clust = CLUSTER_FUNCS[cluster_type]
    new_data = np.zeros((num_classes*num_points_per_class, X.shape[1]))
    new_labels = np.zeros(num_classes*num_points_per_class)
    labels, new_data[:num_classes] = clust(X, num_classes)
    new_labels[:num_classes] = np.arange(num_classes)
    new_labels[num_classes:] = np.arange(num_classes)
    #NOTE: I'm using "far" points for now to see how they do.
    # I suspect this will not be good enough.
    edge_idxs = get_far_points(new_data[:num_classes], X, labels, num_far_points)
    new_data[num_classes:] = X[edge_idxs]
    bin_labels = binarize(new_labels, num_classes)
    return new_data, bin_labels

def prime_typec(X, labels, cluster_type, num_classes, nested_clusters=3, num_far_points=1):
    # priming C: pretrain with nested cluster centroids & edge points from expected dataset
    clust = CLUSTER_FUNCS[cluster_type]
    num_points_per_class = (num_far_points+1)*nested_clusters
    new_num_samples = num_classes*num_points_per_class
    new_data = np.zeros((new_num_samples, X.shape[1]))
    new_labels = np.zeros(new_num_samples)
    labels = debinarize(labels)
    for i in range(num_classes): #for each class...
        #find representative samples (centroids) in that class
        idx = i*num_points_per_class
        next_idx = idx+num_points_per_class
        class_samples = X[labels==i]
        nested_labels, nested_centroids = clust(class_samples, nested_clusters)
        new_data[idx:idx+nested_clusters] = nested_centroids
        #and get outliers in those representative samples
        edge_idxs = get_far_points(nested_centroids, class_samples, nested_labels, num_far_points)
        new_data[idx+nested_clusters:next_idx] = class_samples[edge_idxs]
        new_labels[idx:next_idx] = i
    bin_labels = binarize(new_labels, num_classes)
    return new_data, bin_labels

###########################

############# Selection Functions ##############
def random_equal_selection(labels, num_classes, samples_per_class):
    '''randomly select a certain number of items in each class.'''
    # labels should be debinarized
    new_num_samples = num_classes*samples_per_class
    sample_idxs = np.zeros(new_num_samples)
    for i in range(num_classes):
        class_samples = np.nonzero(labels==i)
        sample_idxs[i*samples_per_class:(i+1)*samples_per_class] = \
            class_samples[np.random.shuffle(np.arange(len(class_samples)))[:samples_per_class]]
    return sample_idxs