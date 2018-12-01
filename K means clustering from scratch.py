# Based on youtube.com/sentdex 's SVM Machine Learning tutorials
# Rewritten and commented by Andrew
###################################


import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

# Training data
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

##plt.scatter(X[:,0], X[:,1], s=100)
##plt.show()
##
colors = 2*["g","r","c","b","k"]

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    # Starts with k number of centroids. All points are classified by which centroid they are closest too.
    # The mean of each class of data points becomes the new centroid. Repeat until the centroids do not move.
    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            # Assign first k data points of data to centroids[]
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {} # Contains centroids and classifications

            # Erase the classfications dictionary
            for i in range(self.k):
                self.classifications[i] = []

            # Note: featureset = [1,2] and centroid =   ,in first loop,
            for featureset in data: 
                # Distance is a list populated with k no. of values (ie. 0,1)
                # and finds magnitude of vector featureset -
                # distances[0] -> distance to first centroid, distances[1] -> distance from [0] to [1]
                
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances)) # Return index of minimum value in list
                self.classifications[classification].append(featureset) # Append featureset onto end

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimised = True

            # If the previous and current centroid's distance change is within a certain tolerance, break loop.
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100) < self.tol:
                    optimised = False

            if optimised:
                break;

    # Return the classification of the data point
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

## End class

clf = K_Means()
clf.fit(X)

# Plot the centroids
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                color="k", s=100)

# Plot the features
for classification in clf.classifications:
    colour = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker="x", color=colour, s=100)

# Predict random datapoints and display on graph
unknowns = np.array([[1, 3],
              [0, 10],
              [2, 5 ],
              [1, 3],
              [2, 7],
              [6,2]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0],unknown[1], marker="*", color=colors[classification], s=100)

print('Crosses are train data and stars are predictions')
plt.axis('equal')
plt.show()








