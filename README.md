# Basic Feedforward Neural Network with Backpropagation

This is a simple implementation of a feedforward neural network from scratch using only Python and NumPy.  
It demonstrates the basic concepts of how neural networks learn through forward propagation and backpropagation.

## 🧠 Structure

- **Input Layer:** 2 neurons (for 2 input features)
- **Hidden Layer:** 4 neurons (configurable)
- **Output Layer:** 1 neuron (for binary classification)

All layers use the sigmoid activation function.

---

## 📌 What It Does

This neural network learns to solve a basic binary classification task using the following steps:

### 1. **Forward Propagation**
Calculates the output prediction using randomly initialized weights and biases:
- `Z = W·X + b`
- `A = sigmoid(Z)`

### 2. **Loss Calculation**
The loss is computed using Mean Squared Error (MSE) or Cross-Entropy (if implemented).

### 3. **Backpropagation**
Error is propagated backwards to update weights:
- Uses derivatives of sigmoid
- Updates weights using Gradient Descent

---

## ⚙️ Why Each Step Matters

| Step | Purpose |
|------|---------|
| **Weight Initialization** | Starts learning with small random values to avoid symmetry. |
| **Sigmoid Activation** | Introduces non-linearity so the network can learn complex patterns. |
| **Backpropagation** | Adjusts weights to minimize loss based on the gradient. |
| **Learning Rate** | Controls how large the updates to the weights are. |

---

## 📁 Files

- `neural_network.py` - main implementation
- (Optional) `train_data.csv` - input/output training data (if separated)
- `.gitignore` - to ignore IDE settings and unnecessary folders like `.idea/`

---

## 🚀 How to Run

Make sure you have NumPy installed:

```bash
pip install numpy
```
Then, run:
```bash
python neural_network.py
```
