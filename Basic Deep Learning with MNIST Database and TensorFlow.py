# Based on youtube.com/sentdex 's Neural Networl tutorials
# Rewritten and commented by Andrew
###################################

# Aim: create a neural network model that identifies the number written 
# in a 28x28 pixel image using the mnist dataset.

#### Outline of how this deep learning model works. ####

# Input and hidden layers are assigned random weights and biases
# Each hidden layer's activation is calculated with sigmoid(prev_activation[] * weights[]) + biases[])
# Compare output of first iteration with intended output using cost function: SUM((actual - output )^2)
# Minimise the cost with the Adam Optimiser function, adjusting the weights and biases accordingly
# Loop 10 times

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # Feed batch of 100 features through network at a time

x = tf.placeholder('float',[None, 784]) # Features, 28*28 = 784
y = tf.placeholder('float') # Label

def neural_network_model(data):

    # Generates a tensor containing random numbers within a normal distribution of shape [784,500]
    hidden_1_layer = { 'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                       'biases':tf.Variable(tf.random_normal([n_nodes_hl1])) }

    hidden_2_layer = { 'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                       'biases':tf.Variable(tf.random_normal([n_nodes_hl2])) }

    hidden_3_layer = { 'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                       'biases':tf.Variable(tf.random_normal([n_nodes_hl3])) }

    output_layer = { 'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                       'biases':tf.Variable(tf.random_normal([n_classes])) }

    # Calculate the next activation number: (activation[] * weights[]) + biases[]
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # Normalise

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
    optimiser = tf.train.AdamOptimizer().minimize(cost)
    howMany_epochs = 10 # Cycles feed forward + backprop.
    # Basically one full cycle of the data plus the altering of weights/biases
    
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())

        ### Trains network
        for epoch in range(howMany_epochs):
            epoch_loss = 0 # Cost after each training interation

            # total no. of samples / batch size, how many times to cycle in a epoch
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size) # Train batch
                _, c = sess.run( [optimiser, cost], feed_dict = {x: epoch_x, y: epoch_y} ) # Optimise
                epoch_loss += c # Sum up costs
            print('Epoch', epoch, 'completed out of', howMany_epochs, 'loss: ', epoch_loss)

        # Returns bool value if prediction == y (indexes with largest values are equal)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)





