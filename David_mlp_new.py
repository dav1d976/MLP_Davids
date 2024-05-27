import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Created by David Pfliehinger and David Birkel
# ------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)  # every message with the level INFO and above will be printed
logger = logging.getLogger(__name__)  # Set name of the logger


class ActivationFunction:
    def __call__(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")


class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1 - sigmoid_x)


class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Neuron:
    def __init__(self, weights, bias, activation_function):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_function = activation_function

    def feedforward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(z)


class Layer:
    def __init__(self, number_of_neurons, number_of_inputs):
        self.neurons = []
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs = number_of_inputs
        self.outputs = None

    # Create weights and biases for each neuron in the layer
    def populate_neurons(self, activation_function):
        for _ in range(self.number_of_neurons):
            weights = np.random.randn(self.number_of_inputs)
            bias = np.random.randn()

            # Create the neuron within a layer as an instance of the Neuron class
            neuron = Neuron(weights, bias, activation_function())
            self.neurons.append(neuron)  # Add it to the list of neurons which represents the layer

    def feedforward(self, inputs):
        self.outputs = np.array([neuron.feedforward(inputs) for neuron in self.neurons])
        return self.outputs


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, inputs, targets, epochs, learning_rate):
        average_error_by_epoch = []
        # Proces of adjusting weights is repeated for a number of epochs which can be changed
        for epoch in range(epochs):
            total_error = 0
            # Iterate over all the data points in the dataset
            for i in range(len(inputs)):
                x = inputs[i]
                y = targets[i]

                # Forward pass:
                # Create a list of outputs for each layer including the input layer
                outputs = [x]
                for layer in self.layers:
                    x = layer.feedforward(x)
                    outputs.append(x)

                # Calculate error (output layer)
                # The error is calculated as the difference between the output of the network and the target value
                error = outputs[-1] - y
                total_error += np.sum(error ** 2)

                # Backward pass:
                # Formula for delta in the output layer: error * f'(output)
                # The derivative of the activation function get accessed through the last neuron in the last layer
                deltas = [error * self.layers[-1].neurons[0].activation_function.derivative(outputs[-1])]

                # The deltas for the hidden layers are calculated
                # Iterate backwards over the layers starting from the second last layer
                # That is because the delta of the output layer has already been calculated
                for i in reversed(range(len(self.layers) - 1)):
                    layer = self.layers[i]
                    next_layer = self.layers[i + 1]
                    # The last layer of deltas is multiplied with the weights of the next layer
                    # The result is multiplied with the derivative of the activation function of the current layer
                    # The result is the delta for the current layer
                    delta = np.dot(deltas[-1], np.array([neuron.weights for neuron in next_layer.neurons])) * \
                            layer.neurons[0].activation_function.derivative(outputs[i + 1])
                    deltas.append(delta)
                deltas.reverse()  # List of deltas gets reversed to match the order of the layers

                # Update weights and biases
                # Iteration over every layer
                for i in range(len(self.layers)):
                    layer = self.layers[i]
                    # Iteration over every neuron in the layer
                    # For the first hidden layer the input is the input data
                    # For every following layer the input is the output of the previous layer
                    for j, neuron in enumerate(layer.neurons):
                        if i == 0:
                            neuron.weights -= learning_rate * deltas[i][j] * inputs[i]
                        else:
                            neuron.weights -= learning_rate * deltas[i][j] * outputs[i]
                        neuron.bias -= learning_rate * deltas[i][j]

            # An average error for each epoch is calculated and stored in a list
            average_error_by_epoch.append(total_error / len(inputs))

            logger.info(f'Epoch {epoch + 1}/{epochs}, Error: {total_error / len(inputs)}')
        # The list of average errors is returned
        return average_error_by_epoch


class EvaluationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

# --------------------------------------------
# Main
# --------------------------------------------
if __name__ == "__main__":
    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = data.target.reshape(-1, 1)  # Reshape list to vector

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test train split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Transform 1D label-array to one with 3 columns, to match the output layer of the neural network
    y_layered = np.zeros((y_train.size, 3))
    for i in range(y_train.size):
        y_layered[i, y_train[i]] = 1
    y_train_layered = y_layered

    mlp = NeuralNetwork()
    hidden_layer = Layer(10, 4)
    output_layer = Layer(3, 10)

    # Create the layers of the MLP as a list of neurons which are instances of the Neuron class
    hidden_layer.populate_neurons(ReLU)
    output_layer.populate_neurons(Sigmoid)

    # Add the lists of layers to the list which represents the neural network
    mlp.add_layer(hidden_layer)
    mlp.add_layer(output_layer)

    epochs = 100
    learning_rate = 0.01

    if True:
        errors = mlp.train(X_train, y_train_layered, epochs, learning_rate)

        logger.info("Training complete. Testing with one example from test set:")

        y_pred = np.array([mlp.predict(x) for x in X_test])
        y_pred_labels = np.argmax(y_pred, axis=1)

        test_accuracy = EvaluationMetrics.accuracy(y_test.flatten(), y_pred_labels)
        logger.info(f"Accuracy on the test set: {test_accuracy * 100:.2f}%")

        # Plotting the error over epochs
        plt.plot(range(epochs), errors, label='Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error over Epochs')
        plt.legend()
        plt.show()

        # Add an additional data point manually
        logger.info("Do you want to type in an own data point? (yes/no)")
        answer = input(">")
        if answer.lower() == "yes":
            new_data_point = []
            for feature in ["sepal length", "sepal width", "petal length", "petal width"]:
                logger.info(f"Type in a value for the {feature} of the new data point.")
                number = float(input())
                new_data_point.append(number)

            scaled_new_data_point = scaler.transform([new_data_point])  # Scaling of new data point
            scaled_new_data_point = scaled_new_data_point.reshape(-1, 1)

            prediction = mlp.predict(scaled_new_data_point)  # Prediction with the neural network
            predicted_class = np.argmax(prediction)  # Determination of the most likely class
            logger.info(prediction)
            logger.info(predicted_class)
        else:
            logger.info("The Progam will be terminated. Goodbye!")