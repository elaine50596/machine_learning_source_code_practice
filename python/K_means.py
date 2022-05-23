## Source Code Practice for K means clustering
# Objective: find K clusters from the "unlabeled" data
# eg: segment users purchasing behaviors

import numpy as np
import random

class K_means:
    def __init__(self) -> None:
        pass
    
    def initialize_centroids(self, data, k):
        centroids = []
        axes_min = np.apply_along_axis(min, axis = 0, arr = data)
        axes_max = np.apply_along_axis(max, axis = 0, arr = data)

        for i in range(k):
            centroid = [random.uniform(axes_min[i], axes_max[i]) for i in range(len(axes_min))]
            centroids.append(centroid)
        
        return centroids

    def update_centroids(self, data, labels, k):
        centroids = []
        for i in range(k):
            index_subset = [label==i for label in labels]
            data_subset = data[index_subset,:]

            centroids.append(list(np.apply_along_axis(np.mean, axis = 0, arr = data_subset)))

        return centroids

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
        axis_distance = np.array(point_1)-np.array(point_2)
        
        return np.dot(axis_distance, axis_distance)

    def assign_cluster(self, data, centroids):
        labels = []
        for point in data:
            centroids_distance = [self.get_distance(point, centroid) for centroid in centroids]
            labels.append(np.argmin(centroids_distance))

        return labels

    def should_stop(self, old_centroids, centroids, threshold = 1e-5):
        total_distance = 0
        for old_centroid, centroid in zip(old_centroids, centroids):
            centroid_distance = self.get_distance(old_centroid, centroid)
            total_distance+=centroid_distance
        
        if total_distance <= threshold:
            return True
        else:
            return False

    def clustering(self, data, k):
        centroids = self.initialize_centroids(data, k)

        while True:
            old_centroids = centroids
            labels = self.assign_cluster(data, centroids)
            centroids = self.update_centroids(data, labels, k)

            if self.should_stop(old_centroids, centroids):
                break
        
        return labels

# simple data example at 2-dimensional space
data = np.array([[1,2], 
                [2,1], 
                [3,2], 
                [4,2], 
                [5,1], 
                [3,7], 
                [2,4]])

K_means_test = K_means()
cluster_labels= K_means_test.clustering(data, 2)

import matplotlib.pyplot as plt
plt.scatter(data[:,0], data[:,1], c = cluster_labels)


