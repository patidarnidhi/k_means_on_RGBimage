import numpy as np
import random as random
import math as math
from numpy import linalg as LA
from utils import error

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        
        #Computing random means
        means = []
        index = [random.sample(range(0,len(X)), self.num_clusters)]
        for i in index :
            means.append(X[i])
        means = means[0]
        

        for itr in range(max_iter):
            # Assign each sample to the closest prototype
            A= [[0]*len(X) for i in range(self.num_clusters)]
            for i in range(len(X)):
                index = -1
                min = math.inf
                for j in range(0,len(means)) :
                    dis = LA.norm(X[i]-means[j])
                    if(dis < min):
                        index = j
                        min = dis
                A[index][i]=1
            
            
            
            # Update prototypes
            upd_mean = []
            for i in range(0,self.num_clusters):
                count = 0
                mean = np.zeros(len(X[0]))
                for j in range(0,len(X)):
                    if(A[i][j]==1):
                        mean+=X[j]
                        count+=1 
                try:
                    mean = mean*(1/count)
                except:
                    mean = 0
                upd_mean.append(mean)
                
            
            if (np.array_equal(means,upd_mean) or itr==max_iter-1 or error(np.array(means).reshape(self.num_clusters,3),np.array(upd_mean).reshape(self.num_clusters,3))<=self.epsilon):
                self.cluster_centers = upd_mean
                break
                
            means = upd_mean  

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point

        # Compute the distance between each data point and each cluster center
        dists = np.zeros((len(X), self.num_clusters))
        for i in range(len(X)):
            for j in range(self.num_clusters):
                dists[i, j] = LA.norm(X[i] - self.cluster_centers[j])

        # Assign each data point to the closest cluster based on the minimum distance
        cluster_labels = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            cluster_labels[i] = np.argmin(dists[i])

        return cluster_labels


    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)


    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        closest_cluster_centers = []
        closest_cluster_idx = self.predict(X)
        for i in range(len(X)):
            closest_cluster_centers.append(self.cluster_centers[closest_cluster_idx[i]])

        return np.array(closest_cluster_centers)