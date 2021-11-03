import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kohonen import KohonenNetwork


def compress_image(filename, number_of_neurons, crop_size, number_of_crops,
                   learning_rate, normalize):
    # read image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # extract random crops
    random_crops = np.array([
        np.reshape(image[x:x + crop_size, y:y + crop_size], (-1, ))
        for x, y in zip(
            np.random.randint(
                0, image.shape[0] - crop_size, size=(number_of_crops, )),
            np.random.randint(
                0, image.shape[1] - crop_size, size=(number_of_crops, )))
    ])

    # clusterize random crops using kohonen network
    kohonen = KohonenNetwork(number_of_neurons,
                             random_crops,
                             normalize=normalize)
    kohonen.training_step(learning_rate=learning_rate)
    while not kohonen.should_stop():
        print(
            f"dead neurons: {kohonen._n_loosers}\t max winner step: {kohonen._max_winner_step}"
        )
        kohonen.training_step(learning_rate=learning_rate)

    # simulate decoded image view - replace each crop with activated neuron weights
    decoded_image = np.empty_like(image, dtype=np.float32)
    for i in range(0, image.shape[0], crop_size):
        for j in range(0, image.shape[1], crop_size):
            crop = np.reshape(image[i:i + crop_size, j:j + crop_size], (-1, )).astype(np.float32)
            factor = np.sqrt(np.sum(crop**2)) if normalize else 1.0
            crop = crop / factor
            winner = kohonen.winner(crop)
            decoded_image[i:i + crop_size, j:j + crop_size] = np.reshape(kohonen.W[winner], (crop_size, crop_size)) * factor
    cv2.imwrite("output.png", decoded_image)
    plt.imshow(decoded_image, cmap='gray')
    plt.show()

    # calculate compression ratio
    not_compressed_size = image.shape[0] * image.shape[1] * 8
    compressed_size = (image.shape[0] / crop_size) * (
        image.shape[1] / crop_size) * np.ceil(np.log2(
            number_of_neurons)) + crop_size * crop_size * number_of_neurons * 8
    compression_ratio = compressed_size / not_compressed_size
    print(f"Compression ratio: {compression_ratio}")


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
