# Based on youtube.com/sentdex 's SVM Machine Learning tutorials
# Rewritten and commented by Andrew
###################################

# The support vector machine performs a binary classification by finding
# the optimial hyperplane that maximises the margin between two classes.
# Data can then be classified as A (above) or B (below) the hyperplane.

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'g', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # Optimise and train model
    def fit(self, data):
        self.data = data
        # Dictionary framework ->  { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1], # transformations to multiply through when optimising
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = [] # Single list of every datapoint
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data) # Maximum value from data
        self.min_feature_value = min(all_data) # Minimum value from data
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]
                      # 0.001.. Point of expense, uses a lot of CPU

        b_range_multiple = 5 # Extremely expensive
        b_multiple = 5 # Don't need to take small steps like we do with w
        latest_optimum = self.max_feature_value*10

        # Loop though step_sizes and data to find w and b
        # such that yi(xi.w + b) ~= 1  (approximately)

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimised = False # Bool remains false until global minimum is found.
            while not optimised:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # Contraint in SVM: yi(xi.w + b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    break
                        if found_option:
                            # np.linalg.norm finds magnitude of vector
                            # Remember dict is as follows -> { ||w||: [w,b] }
                            # Assign the w_t and b value to the dictionary
                            # using the magnitude of the vector as it's index
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimised = True
                    print('Optimised a step')
                else:
                    w = w - step #Matrix subtraction, step is a scalar

            # Assending order of all instances of the dictionary: w, b
            norms = sorted([n for n in opt_dict]) 
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0] # Dictionary of w magnitudes
            self.b = opt_choice[1] # Dictionary of b values
            latest_optimum = opt_choice[0][0]+step*2

        # Print point accuracy
        print('Optimisation accuruacy:')
        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))
    ## End fit


    def predict(self, features):
        # Classification = sign( x.y + b), as per the definition
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    # For visualising the data and hyperplane, doesn't effect SVM
    def visualize(self): # Plots all the data, data_dict is dictionary of data points
        [[self.ax.scatter(x[0],x[1],s=100,c=self.colors[i]) for x in data_dict[i]] for i in data_dict] #self.colors[i]

        # Hyperplane = x.w + b
        # We want  v = x.w + b
        # positive support vector = 1, negative support vector = -1, decision boundary = 0
        # Find two points that fit these conditions to draw the hyperplane.
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # Positive support vector hyperplane ->  (w.x+b) = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])

        # Negative support vector hyperplane ->  (w.x+b) = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # Decision boundary ->  (w.x+b) = 0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2])

        plt.show()

## End Class



# Main dataset
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8]]),
             1:np.array([[5,1],
                         [6,-1],
                         [7,3]])}

# Run program
svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],[1,3],[3,3],[5,7],[5,4],[6,-5],[3,5]]
for p in predict_us:
    svm.predict(p) # Stars on graph are prediction
print('Dots are trained data and stars are predicted.')
svm.visualize()




