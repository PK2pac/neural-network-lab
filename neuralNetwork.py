import time

import numpy as np
import scipy.special
import matplotlib.pyplot as plt


class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        self.lr = learningRate

        # инициализируем случайные веса
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # вход и целевое переводим в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # рассчитаем сигналы, входящие в скрытый слой
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчитаем сигналы, выходящие из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитаем сигналы, входящие в скрытый слой
        final_inputs = np.dot(self.who, hidden_outputs)

        # рассчитаем сигналы, выходящие из скрытого слоя
        final_outputs = self.activation_function(final_inputs)

        # посчитаем ошибку
        output_errors = targets - final_outputs

        # разделение по весам
        hidden_errors = np.dot(self.who.T, output_errors)

        # обновить веса между скрытым и внешним слоем
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.00 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # обновить веса между входящим и скрытым
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.00 - hidden_outputs)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        # переведем входные данные в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T

        # вычислим сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)

        # вычислим сигналы, выходящие из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # вычислим сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)

        # вычислим сигналы, выходящие из выходного слоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 1
output_nodes = 10
learning_rate = 0.01
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
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

# проверим сеть
scorecard = []  # здесь соберем верные ответы
test_data_file = open('mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
t1 = time.time()
print('Время', t1 - t0)
print(test_data_list)

for record in test_data_list:
    all_values = record.split(',')

    correct_label = int(all_values[0])
    print(correct_label, "правильный ответ")

    # нормализуем биты изображения в диапозон [0.0;1.0]
    inputs = (np.asfarray(all_values[1:]))
    outputs = n.query(inputs)
    # вытащим тот элементик, которому сеть дала наибольший приоритет
    label = np.argmax(outputs)
    print(label, "ответ нс")

    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
# покажем эффективность разработанной нс
scorecard_array = np.asarray(scorecard)
print(scorecard_array.sum() / scorecard_array.size)
