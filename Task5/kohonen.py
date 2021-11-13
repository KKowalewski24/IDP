import numpy as np


class KohonenNetwork():
    def __init__(self, n_neurons: int, X: np.ndarray, normalize: bool = False):
        self.n_outputs = n_neurons
        self.n_inputs = X.shape[1]
        self.random_generator = np.random.default_rng()
        self.X = X
        self.W = self.random_generator.uniform(np.min(self.X),
                                               np.max(self.X),
                                               size=(self.n_outputs,
                                                     self.n_inputs))
        self.normalize = normalize
        if normalize:
            self.X = X / np.sqrt(
                np.sum(X.astype(np.float32)**2, axis=1, keepdims=True))
            self.W = self.W / np.sqrt(np.sum(self.W**2, axis=1, keepdims=True))
            self.orig_X = X

    def winner(self, X: np.ndarray):
        if self.normalize:
            return np.argmax(np.matmul(self.W, np.transpose(X)), axis=0)
        else:
            W = np.tile(self.W, (len(X), 1, 1))
            X = np.reshape(np.repeat(X, self.n_outputs, axis=0),
                           (len(X), self.n_outputs, self.n_inputs))
            return np.argmin(np.sum((W - X)**2, axis=2), axis=1)

    def training_step(self, learning_rate):
        W = self.W.copy()

        # find and update winners
        winners = self.winner(self.X)
        for winner, x in zip(winners, self.X):
            W[winner] += learning_rate * (x - W[winner])

        # reset total loosers (dead neurons)
        loosers = list(set(range(len(W))) - set(winners))
        W[loosers] = self.random_generator.uniform(np.min(self.X),
                                                   np.max(self.X),
                                                   size=(len(loosers),
                                                         self.n_inputs))
        if self.normalize:
            W = W / np.sqrt(np.sum(W**2, axis=1, keepdims=True))

        self._max_winner_step = np.max(self.W[winners] - W[winners])
        self._n_loosers = len(loosers)
        self.W = W

    def should_stop(self):
        return self._n_loosers == 0 and self._max_winner_step < 0.00001 * (
            np.max(self.X) - np.min(self.X))


if __name__ == "__main__":
    # simple Kohonen visualization
    from sklearn.datasets import make_moons
    import matplotlib
    import matplotlib.pyplot as plt

    X, y = make_moons(n_samples=200, noise=0.1)
    kohonen = KohonenNetwork(20, X, normalize=False)

    fig, ax = plt.subplots()
    moons, = ax.plot(kohonen.X[:, 0], kohonen.X[:, 1], 'b.')
    neurons, = ax.plot(kohonen.W[:, 0], kohonen.W[:, 1], 'ro')

    def update(frame):
        kohonen.training_step(learning_rate=0.1)
        if kohonen.should_stop():
            neurons.set_color('black')
            anim.pause()
        neurons.set_data(kohonen.W[:, 0], kohonen.W[:, 1])
        return moons, neurons

    anim = matplotlib.animation.FuncAnimation(fig,
                                              update,
                                              frames=range(1000),
                                              interval=100)
    plt.show()
