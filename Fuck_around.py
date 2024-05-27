import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Created by David Pfliehinger and David Birkel
# ------------------------------------------------------------

data = np.array([[3.5, 1.3, 0],
                 [3.4, 1.0, 0],
                 [6.7, 4.5, 1],
                 [6.5, 5.0, 1],
                 [3.8, 0.6, 0],
                 [5.5, 4.5, 1],
                 [3.2, 1.0, 0],
                 [6.0, 5.5, 1]])

features = data[:, :2]
labels = data[:, 2]
labels = labels.reshape(-1, 1)

# Normalize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and validation sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=None)
print(features_test)
print(labels_test)
print("")
# Initialize the weights and biases
w1 = np.random.randn(2, 10)  # two inputs 10 hidden layers
w2 = np.random.randn(10, 1)  # 10 hidden layers 1 output
b1 = np.random.randn(1, 10)
b2 = np.random.randn(1, 1)

# Use ReLU as the activation function for the hidden layer
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

epochs = 1000

# Update the forward propagation and back propagation to use ReLU
for _ in range(epochs):
    # Forward propagation
    z1 = np.dot(features_train, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = 1 / (1 + np.exp(-z2))  # Use sigmoid for the output layer

    # Back propagation
    delta2 = (a2 - labels_train) * (a2 * (1 - a2))  # Derivative of sigmoid
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    delta1 = np.dot(delta2, w2.T) * relu_derivative(z1)  # Derivative of ReLU
    dW1 = np.dot(features_train.T, delta1)
    db1 = np.sum(delta1, axis=0)

    # Update weights
    learning_rate = 0.01
    w1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Validate the model
z1 = np.dot(features_test, w1) + b1
a1 = relu(z1)
z2 = np.dot(a1, w2) + b2
a2 = 1 / (1 + np.exp(-z2))
print(a2)


