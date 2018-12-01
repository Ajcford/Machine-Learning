import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

### Random blobs
##centres = random.randrange(2,5)
##X, y = make_blobs(n_samples=30, centers=centres, n_features=2)

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3]])
##plt.show()

colors = 10*["g","r","c","b","k"]

class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step = 100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    # Every data point starts as a centroid with a bandwidth (or radius) around it
    # With each iteration, the mean of all the data points in the bandwidth is calculated
    # and becomes the new centroid. Cluster centre is moved to a point of convergence.
    def fit(self, data):
        if self.radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step
        
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        # [::-1] reverse's the list
        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                
                for featureset in data:
                    # If feature is within the bandwidth
##                    if np.linalg.norm(featureset - centroid) < self.radius:
##                        in_bandwidth.append(featureset)
                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:
                        distance = 0.000000001
                    weight_index = int(distance/self.radius) # How many "radius steps"
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    to_add = (weights[weight_index]**2) * [featureset]
                    in_bandwidth += to_add

                # Average data point in in_bandwidth
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid)) # Convert array to tuple
            
            uniques = sorted(list(set(new_centroids)))

            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass # If it's itself, ignore
                    # If two centroids are close enough, merge them into one
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass
                    

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimised = True

            for i in centroids:
                # If previous centroid is equal to current centroid, break loop
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimised = False
                
                if not optimised:
                    break
            if optimised:
                break
        self.centroids = centroids

        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)
             

    def predict(self,data):
        distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
## End class


clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color = color, s=100)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color="k", marker='*', s=150)

plt.show()


















