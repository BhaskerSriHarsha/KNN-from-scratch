# KNN-from-scratch
Implementing K-Nearest Neighbors from scratch in Python

This is just for practice.

**knn.py** contains the code for KNN classifier. **test rig.py** file contains a test case to run the KNN classifier code on MNIST Digits and MNIST Fashion datasets.
The classifier is functional and is in ready to use state. 3 functions, namely predict_sample, predict and evaluate have been implemented into the classifier.

TO DO upgrades:
1. statistics.mode throws exceptions when there is more than 1 mode. Have to replace it with something else.
2. Currently only Manhattan distance was implemented, Euclidean distance also needs to be added.
3. The algorithm is very slow, running it parallely on multiple systems can be tried.
