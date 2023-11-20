import numpy as np
import matplotlib.pyplot as plt
import time

class backpropagate:
    def __init__(self, data: list, layers: list):
        self.data = data
        self.input = self.data[0]
        self.output = self.data[1]
        self.layers = [self.input.shape[1]] + layers + [self.output.shape[1]]
        self.weights = {}
        self.loss_list = []
        self.epoch = 0
        for _ in enumerate(self.layers):
            if _[0] < len(self.layers) - 1:
                self.weights["weight{0}".format(_[0])] = np.zeros((_[1], self.layers[_[0] + 1]))
    #sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    #sigmoid derivative function
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    #loss function
    def mean_square_error(self, input, output):
        return ((input - output)**2)/2
    #passing each input through neural network
    def forward_pass(self, input: np.array) -> list:
        result = [np.array([input])]
        for i in range(len(self.layers) - 1):
            input = self.sigmoid(input@self.weights['weight{0}'.format(i)])
            result.append(np.array([input]))
        return result
    #backpropagate a result and get partial derivatives respect to each weight
    def backward_pass(self, result, output):
        update = {}
        derivating = result[-1] - output #first loss derivate
        for _ in reversed(range(len(result) - 1)):
            weight = derivating * self.sigmoid_deriv(result[_ + 1]) #multiplying sigmoid
            update['update{0}'.format(_)] = (weight.T@result[_]).T #finding derivative and update
            derivating = weight@self.weights['weight{0}'.format(_)].T #continue derivation for former layer's weight by multiplying weights
        return update
    #subtracting derivative to weight
    def update_weight(self, update, learning_rate):
        for i in range(len(self.layers) - 1):
            self.weights['weight{0}'.format(i)] = self.weights['weight{0}'.format(i)] - learning_rate * update['update{0}'.format(i)]
        return self.weights
    #training the instance
    def train(self, epoch, learning_rate):
        loss = 0
        self.epoch = epoch
        self.loss_list = [_ for _ in range(epoch)]
        for i in range(epoch):
            loss_array = [_ for _ in range(self.input.shape[0])]
            for j in range(self.input.shape[0]):
                input = self.input[j]
                result = self.forward_pass(input)
                output = self.output[j]
                self.update_weight(self.backward_pass(result, output), learning_rate)
                loss = self.mean_square_error(result[-1], output)
                loss_array[j] = loss
            self.loss_list[i] = np.sum(loss_array) / len(loss_array)
            if i % 10 == 0:
                    print("epoch:", i + 10, 'loss:', self.loss_list[i])
    #plotting the result
    def plot(self):
        plt.plot(range(self.epoch), self.loss_list)
        plt.title('Loss Change')
        plt.xlabel('#Epoch')
        plt.ylabel('Loss')
        plt.show()

if __name__== "__main__":
    data_ready = np.loadtxt('./training.txt')
    np.random.shuffle(data_ready)
    data = [i for i in range(2)]
    data[0] = data_ready[:, :2]
    data[1] = data_ready[:, 2:3]
    layers = [2,3]
    test = backpropagate(data=data, layers=layers)
    print(test.weights)
    print(test.forward_pass(test.input[0]))
    print(test.backward_pass(test.forward_pass(test.input[0]), test.output[0]))
    test.train(100, 0.1)

    