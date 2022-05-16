## Source Code Practice for K Nearest Neighbors
# Highlights:
# distance method: euclidean distance
from cmath import sqrt
import numpy as np
from collections import Counter


class KNN:
    def __init__(self) -> None:
        self.x_train = None
        self.y_train = None
    
    def train(self, x_train,y_train):
        """model training process

        Parameters
        ----------
        x : two dimensional array
            training data points axis values
        y : one dimensional array or list
            training data points designated label/value
        """

        self.x_train = x_train
        self.y_train = y_train

    def get_distance(self, point_1,point_2):
        """distance method: euclidean distance:

        Parameters
        ----------
        point_1 : list or array
            axis value for point 1 
        point_2 : list or array
            axis value for point 2 

        Returns
        -------
        float
            euclidean distance value for the two data points
        """
        axis_distance = point_1-point_2
        
        return np.dot(axis_distance, axis_distance)

    def predict(self, x, k = 'None', type = 'regression'):
        """_summary_

        Parameters
        ----------
        x : one-dimensional numpy array of list
            axis value for the new data point
        k : int or None
            parameter
        type : str, optional
            'regression' or 'classification' problem, by default 'regression'

        Returns
        -------
        float or int
            predicted label value or class for the new data point 
        """
        # auto assignment of k if non specified
        if k == 'None':
            k = int(len(x)**0.5)

        # calculate the distance and its label in tuple pairs
        distance_labels = [(self.get_distance(x,train_point), train_label)
        for train_point, train_label in zip(self.x_train, self.y_train)]

        # sort and filter to the K nearest neighbors by distance values
        neighbors = sorted(distance_labels)[:k]
        neighbors_labels = [label for _, label in neighbors]
        
        # for regression problem take average of the distance values from the k nearest neighbors
        if type == 'regression':
            return sum(neighbors_labels)/k
        else:
            return Counter(neighbors_labels).most_common()[0][0]


# test the algorithm
x_train = np.array([[0.1, 0.2, 1, 2, 4],
                    [0.2, 1, 0.5, 2, 4],
                    [2, 0.2, 2,  3, 1]])
y_train = np.array([1, 2, 1, 3, 2])

KNN_test = KNN()
KNN_test.train(x_train, y_train)

KNN_test.predict(np.array([0.3, 0.2, 3, 2, 1])) # if k is not specified, auto select k
KNN_test.predict(np.array([0.3, 0.2, 3, 2, 1]), 2) # if k is specified

# more advanced implementation (skipped here):
# cross validation to find parameter k
    

