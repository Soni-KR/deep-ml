import numpy as np

def train_neuron(x: np.ndarray, y: np.ndarray, initial_weights: np.ndarray,
                 initial_bias: float, learning_rate: float, epochs: int):

    mse_values = []

    for i in range(epochs):
        z = x @ initial_weights + initial_bias
        p = 1 / (1 + np.exp(-z))

        mse = np.mean((p - y) ** 2)
        mse_values.append(round(float(mse), 4))

        dz = 2 * (p - y) * p * (1 - p)

        dw = x.T @ dz / len(y)
        db = np.mean(dz)

        initial_weights -= learning_rate * dw
        initial_bias -= learning_rate * db

    return np.round(initial_weights, 4), round(initial_bias, 4), mse_values