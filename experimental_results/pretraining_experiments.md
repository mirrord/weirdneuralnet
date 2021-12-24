# Pretraining with Artificial and Partial Artificial Subset Selection
I conducted an experiment to examine the possibility of pretraining with extremely small subsets of available data constructed via a variety of methods. 

## Method
The subset selection methods were as follows:
1. no subsets, train using full-batch with no pretraining
2. pretrain using replacement labels constructed from kmeans clustering on data
3. pretrain using replacement labels from kmeans, but only centroids of the groups and the farthest datapoints from the centroid within that group
4. pretrain using a subset constructed from clustering each label group independently (k=3), then selecting the centroids and farthest points within those intra-clusters

To test these methods, I first constructed 100 models (Xavier initialization) using a standard feed-foward fully connected neural network with 1 hidden layer of size 128 and the sigmoid activation for all layers. These 100 models were saved in their initialization state in order to provide a meaningful comparison between pretraining methods. The dataset used was the MNIST handwritten digit set.

All pretraining methods were evaluated with 10, 20, and 50 epochs of pretraining and a total of 100 epochs of full-batch training (90, 80, and 50 epochs of full-batch respectively).

## Results

### Speed
Pretraining methods (3) and (4) were observed to run signficantly faster than traditional training due to their much lighter load. Method (3) used only 20 points for each training epoch, while (4) used 60, compared to the full 50,000 point training set. Some overhead was required to construct the initial clusters required to select the points, however this overhead was small in comparison to the training time.

### Accuracy
After 100 epochs, the baseline models (1) found an accuracy of approximately 50% on the validation set [image needed]. Pretraining (2) did not perform significantly above 10% for any experiment.
Neither pretraining (3) or (4) saw a significant drop in performance between the training accuracy and validation accuracy, indicating that these pretraining methods preserved generalization.
Pretraining (3) found a small decrease in accuracy for 10 and 20 pretraining epochs, around 5% [images needed]. Using 50 pretraining epochs saw around a 20% decrease [image needed].
Pretraining (4) found virtually no decrease in accuracy for 10 epochs, a very small decrease for 20 (around 2%), and about 10% for 50 epochs [images needed].

## Next steps
Training method (2) was originally intended as a baseline comparison. However, this was a poor choice for several reasons. A much better comparison will be found using random subset selection using the same number of datapoints as dictated by (3) and (4).
The above experiment was somewhat naive in construction, as a more meaningful result would be mean number of epochs to find some convergence target. I will therefore re-run the experiments above with this goal and update this document with the results.