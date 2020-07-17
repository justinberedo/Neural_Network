import numpy as np


class NeuralNetwork():

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        # converting weights to a 3 by 1 matrix from -1 to 1 and mean of 0
        self.synaptic_weights0 = 2 * np.random.random((3, 10)) - 1
        self.synaptic_weights1 = 2 * np.random.random((10, 1)) - 1

    def sigmoid(self, x):
        # applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            [output_l1, output_l2] = self.think(training_inputs)

            # computing error rate for back-propagation
            error_l2 = training_outputs - output_l2

            # performing weight adjustments
            adjustments_l2 = error_l2 * self.sigmoid_derivative(output_l2)

            error_l1 = adjustments_l2.dot(self.synaptic_weights1.T)

            adjustments_l1 = error_l1 * self.sigmoid_derivative(output_l1)

            self.synaptic_weights1 += output_l1.T.dot(adjustments_l2)
            self.synaptic_weights0 += training_inputs.T.dot(adjustments_l1)

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats

        inputs = inputs.astype(float)
        output_l1 = self.sigmoid(np.dot(inputs, self.synaptic_weights0))
        output_l2 = self.sigmoid(np.dot(output_l1, self.synaptic_weights1))
        return [output_l1, output_l2]




if __name__ == "__main__":
    # initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights0)
    print(neural_network.synaptic_weights1)

    # training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[0, 0, 1],
                                [0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

    training_outputs = np.array([[0], [1], [1], [0]])

    # training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights0)
    print(neural_network.synaptic_weights1)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))
    # user_input_four = str(input("User Input Four: "))

    print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
    print("New Output data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    print("Wow, we did it!")