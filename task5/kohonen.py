import numpy as np


class KohonenNetwork():
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = np.random.rand(n_outputs, n_inputs) * 2 - 1

    def __call__(self, x, learning_rate=0.0, lambd=0.01):
        y = np.matmul(self.W, x)
        if learning_rate > 0.0:
            winner = np.argmax(y)
            distance = np.sqrt(
                np.sum(np.power(self.W - self.W[winner], 2), axis=1))
            gaussian_neighbourhood = np.exp(-(distance * distance) /
                                            (2 * lambd * lambd))
            self.W = self.W + learning_rate * np.reshape(np.repeat(gaussian_neighbourhood, len(x)), (-1, len(x))) * (x - self.W)


from sklearn.datasets import make_moons
import matplotlib
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200, noise=0.05)
kohonen = KohonenNetwork(2, 100)

fig, ax = plt.subplots()
moons, = ax.plot(X[:, 0], X[:, 1], 'b.')
neurons, = ax.plot(kohonen.W[:, 0], kohonen.W[:, 1], 'ro')


def update(frame):
    for x in X:
        kohonen(x, learning_rate=0.01)
    neurons.set_data(kohonen.W[:, 0], kohonen.W[:, 1])
    return moons, neurons


anim = matplotlib.animation.FuncAnimation(fig, update, frames=range(1000), interval=100)
plt.show()
