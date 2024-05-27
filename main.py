import numpy as np
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

    def populate_neurons(self, activation_function):
        for _ in range(self.number_of_neurons):
            weights = np.random.randn(self.number_of_inputs)
            bias = np.random.randn()
            neuron = Neuron(weights, bias, activation_function())
            self.neurons.append(neuron)

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
        # Variable wird initialisiert, die den error pro Epoche in eine Liste speichert
        average_error_by_epoch = []
        # Eine for-Schleife, die die Anzahl der Epochen durchläuft
        for epoch in range(epochs):
            # Variable, die den gesamten Fehler pro Epoche speichert und für jeden neuen Durchlauf auf 0 setzt
            total_error = 0
            # Eine for-Schleife, die die Anzahl der Trainingsdaten durchläuft (120)
            # Es wird immer die aktuelle Trainingsdate + Ihre Sorte aus den Trainingsdaten genommen
            for i in range(len(inputs)):
                x = inputs[i]
                y = targets[i]

                # Forward pass:
                # Der Forward pass wird für jeden Input und für jede Schicht (layer) durchgeführt.
                outputs = [x]
                # print(outputs)
                for layer in self.layers:
                    x = layer.feedforward(x)
                    outputs.append(x)
                    # print(outputs)

                # Calculate error (output layer)
                # der erwartete Output wird mit dem tatsächlichen Output verglichen und quadriert, damit
                # negative Werte positiv werden. Der Fehler wird dann aufsummiert.
                error = outputs[-1] - y
                total_error += np.sum(error ** 2)

                # Backward pass:
                # Zuerst wird der Fehler für die Output-Schicht berechnet mit der Formel: error * f'(output)
                deltas = [error * self.layers[-1].neurons[0].activation_function.derivative(outputs[-1])]

                # Dann wird der Fehler für die vorherigen Schichten berechnet mit Formel:
                # Summe oder Matrixmultiplikation aus (delta (der nachfolgenden Schicht (links nach rechts)) * weights)
                # * f'(output). Das wird dann der Liste deltas hinzugefügt.
                for i in reversed(range(len(self.layers) - 1)):
                    layer = self.layers[i]
                    next_layer = self.layers[i + 1]
                    delta = np.dot(deltas[-1], np.array([neuron.weights for neuron in next_layer.neurons])) * \
                            layer.neurons[0].activation_function.derivative(outputs[i + 1])
                    deltas.append(delta)
                # Im Moment sind die Deltas von der letzten Schicht bis zur ersten Schicht sortiert. Das wird umgedreht.
                deltas.reverse()

                # Update weights and biases
                # die Gewichte und Biases werden für jedes Neuron in jeder Schicht aktualisiert.
                # Nach der Formel: weights = weights - (learning_rate * delta * der Output der vorherigen Schicht)
                # der Bias wird genauso aktualisiert, nur ohne den Output der vorherigen Schicht.
                for i in range(len(self.layers)):
                    layer = self.layers[i]
                    for j, neuron in enumerate(layer.neurons):
                        if i == 0:
                            neuron.weights -= learning_rate * deltas[i][j] * inputs[i]
                        else:
                            neuron.weights -= learning_rate * deltas[i][j] * outputs[i]
                        neuron.bias -= learning_rate * deltas[i][j]

            # Zuletzt wird die Liste für den Fehler pro Epoche mit dem durchschnittlichen Fehler dieser Epoche ergänzt.
            average_error_by_epoch.append(total_error / len(inputs))

            logger.info(f'Epoch {epoch + 1}/{epochs}, Error: {total_error / len(inputs)}')
        # Die Liste den Fehlern jeder einzelnen Epoche wird zurückgegeben.
        # print(average_error_by_epoch)
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
    # print(X)
    # print(y)
    # print(data.target)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)

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

    if True:
        errors = mlp.train(X_train, y_train_layered, epochs, learning_rate)
        # print(errors)

        logger.info("Training complete. Testing with one example from test set:")

        y_pred = np.array([mlp.predict(x) for x in X_test])
        y_pred_labels = np.argmax(y_pred, axis=1)
        # print(y_pred)

        test_accuracy = EvaluationMetrics.accuracy(y_test.flatten(), y_pred_labels)
        logger.info(f"Accuracy on the test set: {test_accuracy * 100:.2f}%")

        # Plotting the error over epochs
        plt.plot(range(epochs), errors, label='Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error over Epochs')
        plt.legend()
        plt.show()

        # ich möchte manuell vier werte eingeben und diese wie beim datensatz in ein array schreiben
        # ich will mithilfe einer for schleife eine array mit vier Zahleneingaben erstellen
        print("Willst du einen eigenen Test machen? (Ja/Nein)")
        answer = input()
        if answer == "Ja":
            new_data_point = []
            for i in range(4):
                print("Geben Sie den Wert für die", i + 1, ". Zahl ein.")
                number = float(input())
                new_data_point.append(number)

            scaled_new_data_point = scaler.transform([new_data_point])  # Skalieren des neuen Datenpunkts
            scaled_new_data_point = scaled_new_data_point.reshape(-1, 1)

            prediction = mlp.predict(scaled_new_data_point)  # Vorhersage mit dem neuronalen Netz
            predicted_class = np.argmax(prediction)  # Bestimmen der wahrscheinlichsten Klasse
            print(prediction)
            print(predicted_class)
        else:
            print("Das Programm wird beendet.")