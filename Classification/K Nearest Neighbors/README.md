# K Nearest Neighbors
Python implementation of a K Nearest Neighbors model from scratch.

## Algorithm Description
[K-Nearest-Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is a supervised machine learning algorithm. When a new data feature vector is introduced, the algorithm finds the _k_ nearest points to that new data point, and takes the majority vote of the label to be the prediction. The [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance) measure is used to quantify distance between points.

## Euclidean Distance
The Euclidean Distance can be calculated in any number of dimensions. It is the square root of the sum of square differences between the dimensions of the two points.
<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT-BNnXJs2WFM-hvledmFTsECBmQ1ssxkLnucrp3sG8yrXA8VAN" width=250>
</p>

## Implementation
The class contains methods for fitting and making predictions on data. The number of neighbors can be specified; typically an odd number of neighbors is used to avoid ties in majority votes. 

```python
from KNN import KNN

X_train = np.array([[1, 1], [1, 2], [2, 3], [3, 4], [3, 6]])
y_train = np.array([0, 0, 0, 1, 1])
X_test = np.array([[1, 2], [1, 4], [2, 2], [3, 1], [3, 8]])
y_test = np.array([1, 0, 1, 0, 1])

knn = KNN(neighbors=3, standardize=True)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy: ", np.sum(preds != y_test)/len(y_test))
```
