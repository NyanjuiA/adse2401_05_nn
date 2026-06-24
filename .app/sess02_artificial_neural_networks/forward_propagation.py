# Python file to demonstrate Forward Propagation
import numpy as np

def forward_propagation(inputs, weights, biases):
    # Calculate weighted sum of inputs
    weighted_sum = np.dot(inputs, weights) + biases
    # Apply sigmoid activation function
    output = 1 / (1 + np.exp(-weighted_sum))
    return output

# Example inputs
inputs = np.array([0.5, 0.3, 0.2])
weights = np.array([0.4, 0.7, 0.2])
biases = 0.1

# Perform forward propagation
output = forward_propagation(inputs, weights, biases)
print("Forward Propagation Output:", output)