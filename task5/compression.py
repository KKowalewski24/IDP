import argparse
import cv2
import numpy as np
from kohonen import KohonenNetwork


def compress_image(filename, number_of_neurons, crop_size, number_of_crops,
                   learning_rate, normalize):
    data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    random_crops = np.array([
        np.reshape(data[x:x + crop_size, y:y + crop_size], (-1, ))
        for x, y in zip(
            np.random.randint(
                0, data.shape[0] - crop_size, size=(number_of_crops, )),
            np.random.randint(
                0, data.shape[1] - crop_size, size=(number_of_crops, )))
    ])

    kohonen = KohonenNetwork(number_of_neurons,
                             random_crops,
                             normalize=normalize)
    kohonen.training_step(learning_rate=learning_rate)
    while not kohonen.should_stop():
        print(
            f"dead neurons: {kohonen._n_loosers}\t max winner step: {kohonen._max_winner_step}"
        )
        kohonen.training_step(learning_rate=learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--number-of-neurons", type=int, required=True)
    parser.add_argument("--crop-size", type=int, required=True)
    parser.add_argument("--number-of-crops", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    compress_image(args.filename, args.number_of_neurons, args.crop_size,
                   args.number_of_crops, args.learning_rate, args.normalize)
