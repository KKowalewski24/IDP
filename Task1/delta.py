import argparse

import numpy as np
from tqdm import tqdm


def train(N, M, K, n, w_min, w_max, p_min, p_max):
    # initialize patterns and weights
    X = np.random.uniform(p_min, p_max, size=(M, N))
    Z = np.random.uniform(p_min, p_max, size=(M,))
    w = np.random.uniform(w_min, w_max, size=(N,))

    # train
    for k in tqdm(range(K)):
        for x, z in zip(X, Z):
            y = np.sum(w * x)
            w = w + n * (z - y) * x

    # test
    for x, z in zip(X, Z):
        y = np.sum(w * x)
        print(f"z = {np.round(z, 4)}, y = {np.round(y, 4)}")


if __name__ == "__main__":
    np.random.seed(47)
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, required=True)
    parser.add_argument('-M', type=int, required=True)
    parser.add_argument('-K', type=int, required=True)
    parser.add_argument('-n', type=float, required=True)
    parser.add_argument('-w', nargs=2, type=float, required=True)
    parser.add_argument('-p', nargs=2, type=float, required=True)
    args = parser.parse_args()

    train(args.N, args.M, args.K, args.n, args.w[0], args.w[1], args.p[0], args.p[1])
