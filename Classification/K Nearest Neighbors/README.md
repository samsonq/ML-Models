# KNN from scratch
Python implementation of a K Nearest Neighbors model from scratch.

## Algorithm Description
[K nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is a supervised machine learning algorithm. When a new data feature vector is introduced, the algorithm finds the _k_ nearest points to that new data point, and takes the majority vote of the label to be the prediction. The [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) measure is used to quantify distance between points.

## Euclidean Distance
<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT-BNnXJs2WFM-hvledmFTsECBmQ1ssxkLnucrp3sG8yrXA8VAN" width=250>
</p>

## Implementation
[Point](https://github.com/senavs/knn-from-scratch/blob/master/model/point.py) is a class to represent a point in cartesian plane. You are able to sum, subtract, multiply, divide and calculate distance between two points.
``` python
from model.point import Point

p1 = Point([7, 4, 3])
p2 = Point([17, 6, 2])
```
[KNearestNeighbors](https://github.com/senavs/knn-from-scratch/blob/master/model/knn.py) is the model class. Only the methods are allowed: `fit` and `predict`. Look into `help(KNearestNeighbors)` for more infomraiton.
```python
from model.knn import KNearestNeighbors

knn = KNearestNeighbors(k=3)
knn.fit(x_train, y_train)

predict = knn.predict(x_predict)
```

## Apply KNearestNeighbors from scratch in dataset
To show the package working, I created a jupyter notebook with [iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html). Take a look into [here](https://github.com/senavs/knn-from-scratch/blob/master/notebook/knn-iris_dataset.ipynb).
