import math
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from compression import compress_image
from tqdm import tqdm

results_dir = "results/"
filename = "data/boat.png"
crop_sizes = [4, 8]
number_of_neurons = [
    3, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200
]


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    create_directory(results_dir)
    executor = ProcessPoolExecutor()

    for crop_size in crop_sizes:
        n = len(number_of_neurons)
        results = list(
            tqdm(executor.map(compress_image, [filename] * n, number_of_neurons,
                              [crop_size] * n, [10000] * n, [0.01] * n,
                              [False] * n),
                 total=len(number_of_neurons)))

        sorted(results, key=lambda result: result[1])
        df = pd.DataFrame()
        df["number_of_neurons"] = number_of_neurons
        df["compression_ratio"] = [result[1] for result in results]
        df["PSNR"] = [result[2] for result in results]
        decoded_images = [result[0] for result in results]

        # save csv
        df.to_csv(f"{results_dir}crop_size_{crop_size}.csv")

        # plot images
        ncols = int(math.ceil(math.sqrt(len(decoded_images))))
        nrows = int(math.ceil(len(decoded_images) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        for i in axes:
            for j in i:
                j.set_axis_off()
                j.set_frame_on(False)
        for i in range(len(decoded_images)):
            ax = axes[i // ncols][i % ncols]
            ax.imshow(decoded_images[i], cmap="gray")
            ax.set_title(f"{df.loc[i, 'number_of_neurons']}")

        # save images
        for decoded_image, number_of_neurons in zip(decoded_images, df["number_of_neurons"]):
            cv2.imwrite(
                f"{results_dir}crop_size_{crop_size}_neurons_{number_of_neurons}.png", decoded_image
            )

        # plot compression ratio vs PSNR
        fig, ax = plt.subplots()
        ax.plot(df["compression_ratio"], df["PSNR"], '-')
        ax.plot(df["compression_ratio"], df["PSNR"], 'o')
        ax.set_xlabel("compression ratio")
        ax.set_ylabel("PSNR [dB]")

        # plot number of neurons vs PSNR
        fig, ax = plt.subplots()
        ax.plot(df["number_of_neurons"], df["PSNR"], '-')
        ax.plot(df["number_of_neurons"], df["PSNR"], 'o')
        ax.set_xlabel("number of neurons")
        ax.set_ylabel("PSNR [dB]")

        plt.show()
