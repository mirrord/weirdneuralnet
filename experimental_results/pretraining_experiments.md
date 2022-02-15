# Pretraining with Artificial and Partial Artificial Subset Selection
I conducted an experiment to examine the possibility of pretraining with extremely small subsets of available data constructed via a variety of methods. 

## Method
The subset selection methods were as follows:
1. random subset pretraining with 6 items per class
2. pretrain using replacement labels constructed from kmeans clustering on data
3. pretrain using replacement labels from kmeans, but only centroids of the groups and the farthest datapoints from the centroid within that group
4. pretrain using a subset constructed from clustering each label group independently (k=3), then selecting the centroids and farthest points within those intra-clusters

To test these methods, I first constructed 100 models (Xavier initialization) using a standard feed-foward fully connected neural network with 1 hidden layer of size 128 and the sigmoid activation for all layers. These 100 models were saved in their initialization state in order to provide a meaningful comparison between pretraining methods. The dataset used was the MNIST handwritten digit set.

All pretraining methods were evaluated with between 0 and 180 epochs of pretraining (inclusive; experimental methods were evaluated beginning with 20 pretraining rounds instead of 0). Models subjected to pretraining were saved at every 20 rounds of pretraining, then were trained using traditional batch training (b=5000) and the number of training epochs required to reach convergence was recorded. For the purposes of the experiment, convergence was defined as 90% accuracy on training data.

## Results
While some small improvement was noted for method (3), methods (1) and (2) showed negligible benefit. When compared to random subset training, even the method (3) improvements were shown to be insufficient for any real-world application. 
![reference test: random subset pretraining](random_subset_avg_50.png?raw=true)
![experimental method results](averages.png?raw=true)


## Next steps
Training method (2) was originally intended as a baseline comparison. However, this was a poor choice for several reasons. A much better comparison will be found using random subset selection using the same number of datapoints as dictated by (3) and (4).
The above experiment was somewhat naive in construction, as a more meaningful result would be mean number of epochs to find some convergence target. I will therefore re-run the experiments above with this goal and update this document with the results.