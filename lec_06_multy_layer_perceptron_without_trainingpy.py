import numpy as npsedgarzgryezeydzth
import logging
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivationFunction:
    def __call__(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

class ReLU(ActivationFunction):
    def __call__(self, x):
        return max(0, x)

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

    def populate_neurons(self, activation_function):
        for _ in range(self.number_of_neurons):
            weights = np.random.randn(self.number_of_inputs)
            bias = np.random.randn()
            neuron = Neuron(weights, bias, activation_function())
            self.neurons.append(neuron)

    def feedforward(self, inputs):
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])

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

        # implementation help
        # for a predetermined number of steps we repeat the following:

                # Forward pass:
                # - - - - - - -
                # we iterate over all layers and calculate each layer's feed forward,
                # starting with the given input. Each layers calculated feed forward
                # response serves as the input to the next layer.

                # error calculation:
                # - - - - - - - - -
                # now we determine the accuracy of the response's prediction
                # by calculating the error vector and the absolute error

                # Backward pass (gradient descent step):
                # - - - - - - - - - - - - - - - - - - -
                # * we iterate ***backwards*** through the layers and start with the
                #   error between prediction and labeled value
                # * by counting i downwards, we pass also backwards through
                #   our previously activation functions
                # * for each neuron inside a layer, we consider all weights
                #   and decrease each weight proportional to
                #    -> the error caused by this neuron
                #    -> its activation
                #    -> the globally given learning rate (descent velocity)
                #   we also decrease each neuron's bias proportional to
                #    -> its error
                #    -> the learning function

            # Store the average error for this epoch

        return average_error_by_epoch


   class EvaluationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

if __name__ == "__main__":
    data = load_iris()
    X = data.data
    y = data.target.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # create a result vector from the iris classification
    y_layered = np.zeros((y_train.size, 3))
    for i in range(y_train.size):
        y_layered[i, y_train[i]] = 1
    y_train_layered = y_layered

    mlp = NeuralNetwork()
    hidden_layer = Layer(10, 4)
    output_layer = Layer(3, 10)
    hidden_layer.populate_neurons(ReLU)
    output_layer.populate_neurons(Sigmoid)
    mlp.add_layer(hidden_layer)
    mlp.add_layer(output_layer)

    epochs = 100
    learning_rate = 0.01

    # change to true once you finished your implementation
    if False:
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
