import time
import numpy as np
import scipy.special
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

        # Initialize random weights
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # Input and target to 2-d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Calculate input signals, included in the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals coming out of the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate the signals included in the hidden layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals coming out of the hidden layer
        final_outputs = self.activation_function(final_inputs)

        # calculate the error
        output_errors = targets - final_outputs

        # separation by weight
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights between the hidden and outer layer
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.00 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # update weights between incoming and hidden
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.00 - hidden_outputs)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        # translate the input data into a two-dimensional array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate the signals for the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate signals emanating from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate the signals for the output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals coming out of the output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 1
output_nodes = 10
learning_rate = 0.01
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
data_file = open('mnist_train.csv', 'r')
data_list = data_file.readlines()
print(data_list)
data_file.close()
t0 = time.time()
epochs = 25
for e in range(epochs):
    for record in data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# Check the network
scorecard = []  # collect the right answers
test_data_file = open('mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
t1 = time.time()
print('Время', t1 - t0)
print(test_data_list)

for record in test_data_list:
    all_values = record.split(',')

    correct_label = int(all_values[0])
    print(correct_label, "Right")

    # normalize the image bits in the range [0.0;1.0]
    inputs = (np.asfarray(all_values[1:]))
    outputs = n.query(inputs)
    # pull out the element to which the network gave the highest priority
    label = np.argmax(outputs)
    print(label, "Answer of neural network")

    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
# show the effectiveness of the developed neural network
scorecard_array = np.asarray(scorecard)
print(scorecard_array.sum() / scorecard_array.size)
