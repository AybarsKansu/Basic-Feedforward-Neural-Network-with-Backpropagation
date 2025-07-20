import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4x2 input matrix of possible combinations
y = np.array([[0], [1], [1], [0]])  # Corresponding 4x1 output matrix (XOR logic)

np.random.seed(42)

input_weight = np.random.randn(2, 2)    # Weights from input layer to hidden layer
hidden_weight = np.random.randn(2, 1)   # Weights from hidden layer to output layer

hidden_bias = np.zeros([1, 2])      # Bias for hidden layer neurons
output_bias = np.zeros([1, 1])      # Bias for output neuron

epochs = 20000
learning_rate = 0.1

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(t):
    return t * (1 - t)

for epoch in range(epochs):
    hidden_in = np.dot(x, input_weight) + hidden_bias # Compute input values for hidden units
    hidden_out = sigmoid(hidden_in) # Apply sigmoid activation to hidden layer

    output_in = np.dot(hidden_out, hidden_weight) + output_bias # Compute input to output layer
    output_out = sigmoid(output_in) # Apply sigmoid activation to output layer (4x1 matrix)

    error = (y - output_out)
    loss = np.mean(np.square(error))

    if epoch % 1000 == 0:
        print(f"epoch: {epoch} loss: {loss}")

    d_output = error * sigmoid_derivative(output_out) # Error at output layer; shows total error without knowing the source

    error_hidden = d_output.dot(hidden_weight.T)

    d_hidden = error_hidden * sigmoid_derivative(hidden_out)

    hidden_weight += hidden_out.T.dot(d_output) * learning_rate

    input_weight += x.T.dot(d_hidden) * learning_rate

    output_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    hidden_bias += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("\nOutputs after training:")
for i in range(len(x)):
    hidden_output = sigmoid(np.dot(x[i], input_weight) + hidden_bias)
    final_output = sigmoid(np.dot(hidden_output, hidden_weight) + output_bias)
    print(f"Enter: {x[i]}, Guess: {final_output[0][0]:.4f}")