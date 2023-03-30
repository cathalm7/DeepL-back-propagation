"""
The main code for the back propagation assignment. See README.md for details.
"""
import math
from typing import List

import numpy as np

# Sigmoid Function
def sig_func(val):
    return 1 / (1 + np.exp(-val))

#Derivative of Sigmoid Function
def sig_func_g(val):
    return sig_func(val) * (1 - sig_func(val))

class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation.
    """

    @classmethod
    def random(cls, *layer_units: int):
        """Creates a feedforward neural network with the given number of units
        for each layer.

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        pairs = zip(layer_units, layer_units[1:])
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weight matrices.
        The weights correspond to transformations from one layer to the next, so
        the number of layers is equal to one more than the number of weight
        matrices.

        :param layer_weights: A list of weight matrices
        """
        # list of weight matrices
        self.weights = layer_weights 
        # list of activation functions
        self.activations = []

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """
        # Start for input matrix
        self.activations = [input_matrix]
        # For each weight matrices
        for weight in self.weights: 
            # Calculate the output of each layer
            self.activations.append(sig_func(np.dot(self.activations[-1], weight)))
        return self.activations[-1]

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """
        # If output < 0.5 convert to 0 else 1
        return np.where(self.predict(input_matrix) < 0.5, 0, 1)

    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        This method first performs a pass of forward propagation through the
        network, then applies the following procedure to calculate the
        gradients. In the following description, × is matrix multiplication,
        ⊙ is element-wise product, and ⊤ is matrix transpose.

        First, calculate the error, error_L, between last layer's activations,
        h_L, and the output matrix, y:

        error_L = h_L - y

        Then, for each layer l in the network, starting with the layer before
        the output layer and working back to the first layer (the input matrix),
        calculate the gradient for the corresponding weight matrix as follows.
        First, calculate g_l as the element-wise product of the error for the
        next layer, error_{l+1}, and the sigmoid gradient of the next layer's
        weighted sum (before the activation function), a_{l+1}.

        g_l = (error_{l+1} ⊙ sigmoid'(a_{l+1}))⊤

        Then calculate the gradient matrix for layer l as the matrix
        multiplication of g_l and the layer's activations, h_l, divided by the
        number of input examples, N:

        grad_l = (g_l × h_l)⊤ / N

        Finally, calculate the error that should be backpropagated from layer l
        as the matrix multiplication of the weight matrix for layer l and g_l:

        error_l = (weights_l × g_l)⊤

        Once this procedure is complete for all layers, the grad_l matrices
        are the gradients that should be returned.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """
        # number of layers
        layer = len(self.weights) - 1
        # forward propagation
        h_lastest = self.predict(input_matrix)
        # error
        error_lastest = h_lastest - output_matrix 
        # calculate the last activation function's input matrix
        a_lastest = np.dot(self.activations[layer], self.weights[layer])
        # calculate the last activation function's gradient matrix
        g_lastest = np.multiply(error_lastest, sig_func_g(a_lastest)).T

        g_last = g_lastest
        grad = []
        grad.insert(0, np.dot(g_last, self.activations[layer]).T)

        # Back propagate
        while layer >= 1:
            # Calculate the error that should be backpropagated from layer l
            error_l = np.dot(self.weights[layer], g_last).T
            # calculate the activation function's input matrix for previous layer
            a_last = np.dot(self.activations[layer - 1], self.weights[layer - 1])
            # calculate the gradient matrix for previous layer
            g_last = np.multiply(error_l, sig_func_g(a_last)).T
            
            # update grad
            grad.insert(0, np.dot(g_last, self.activations[layer - 1]).T)
            layer -= 1

        n_input = input_matrix.shape[0]
        gradients = [x / n_input for x in grad]

        return gradients

    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """

        for _ in range(0, iterations):
            # Calculate the gradients for the current set of inputs and expected outputs
            gradients = self.gradients(input_matrix, output_matrix)
            # Iterate over each set of gradients and corresponding weight matrices
            for grad, weight in zip(gradients, self.weights):
                # Update the weight matrix by subtracting the product of the
                # learning rate and the gradients from the current weight matrix
                weight -= learning_rate * grad
