from typing import List
import numpy as np


class Image():
    def __init__(self, n_rows, n_cols, lines: List[str]):
        self.name = lines[0]
        lines = lines[1:]
        self.data = np.zeros(shape=(n_rows, n_cols), dtype=np.uint8)
        for row in range(n_rows):
            for col in range(n_cols):
                character = lines[row][col]
                if character == "-":
                    self.data[row, col] = 0
                elif character == "#":
                    self.data[row, col] = 1
                else:
                    raise ValueError(character)

    @staticmethod
    def read_from_file(filename):
        with open(filename) as file:
            lines = file.readlines()
        number_of_images = int(lines[0])
        n_cols = int(lines[1])
        n_rows = int(lines[2])
        lines = lines[3:]
        images = []
        for i in range(number_of_images):
            lines = lines[1:]  # remove empty line
            images.append(Image(n_rows, n_cols, lines[:n_rows + 1]))
            lines = lines[n_rows + 1:]
        return images


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    images = Image.read_from_file('test.txt')
    for image in images:
        plt.imshow(image.data)
        plt.show()

# class Madeline():
#     def __init__(self, n_inputs, n_outpus):
#         self.n_inputs = n_inputs
#         self.n_outputs = n_outputs
#         self.weights = np.zeros(size=(n_outputs, n_inputs))
#
#     def __call__(self, image):
#         return np.matmul(self.weights, X)
#
#     def train(self, images):
