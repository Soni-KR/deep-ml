import math

def single_neuron_model(x: list[list[float]], y: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = []
    for row in x:
	    z = sum(row[i] * weights[i] for i in range(len(weights))) + bias
	    p = 1 / (1 + math.exp(-z))
	    probabilities.append(p)
    mse = sum((probabilities[i] - y[i]) ** 2 for i in range(len(y))) / len(y)
    mse = round(mse, 4)
    return probabilities, mse