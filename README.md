import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR Input and Output
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Set seed
np.random.seed(1)

# Architecture
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# Initialize weights and biases
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Store loss values
losses = []

# Training
for epoch in range(10000):
    # Forward pass
    hl_input = np.dot(X, wh) + bh
    hl_output = sigmoid(hl_input)

    ol_input = np.dot(hl_output, wout) + bout
    predicted_output = sigmoid(ol_input)

    # Compute error and loss
    error = y - predicted_output
    loss = np.mean(np.square(error))
    losses.append(loss)

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)
    error_hidden = d_output.dot(wout.T)
    d_hidden = error_hidden * sigmoid_derivative(hl_output)

    # Update weights and biases
    wout += hl_output.T.dot(d_output) * 0.1
    bout += np.sum(d_output, axis=0, keepdims=True) * 0.1
    wh += X.T.dot(d_hidden) * 0.1
    bh += np.sum(d_hidden, axis=0, keepdims=True) * 0.1

# Final Output
print("Final predicted output:")
print(predicted_output)

# Visualization
plt.plot(losses)
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
