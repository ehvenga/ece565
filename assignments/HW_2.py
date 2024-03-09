'''
 Hari Vengadesh Elangeswaran
 363161
 Homework 2
'''
 
import numpy as np

# Neural Network architecture
n_input_nodes = 7  # Including an extra node to match the 7 nodes requirement
n_nodes_l2 = 5
n_nodes_l3 = 2
n_output_nodes = 1

# Activation function
def logistic(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights randomly
np.random.seed(42)  # For reproducibility
weights_l1_l2 = np.random.rand(n_input_nodes, n_nodes_l2)
weights_l2_l3 = np.random.rand(n_nodes_l2 + 1, n_nodes_l3)  # +1 for bias
weights_l3_output = np.random.rand(n_nodes_l3 + 1, n_output_nodes)  # +1 for bias

# Data
X = np.full((31, n_input_nodes - 1), 3)  # 31 examples, 5 features set to 3, excluding the bias and the extra node
X = np.hstack((X, np.ones((31, 1))))  # Adding bias to the input layer

# Forward propagation
# Layer 2
Z_l2 = np.dot(X, weights_l1_l2)
A_l2 = logistic(Z_l2)
A_l2 = np.hstack((A_l2, np.ones((31, 1))))  # Adding bias to the second layer

# Layer 3
Z_l3 = np.dot(A_l2, weights_l2_l3)
A_l3 = logistic(Z_l3)
A_l3 = np.hstack((A_l3, np.ones((31, 1))))  # Adding bias to the third layer

# Output layer
Z_output = np.dot(A_l3, weights_l3_output)
A_output = logistic(Z_output)

# Print the output
print(A_output)
# print('Shape of Output', A_output.shape)