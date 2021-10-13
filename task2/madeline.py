from typing import List
import numpy as np


class Image():
    def __init__(self, n_rows, n_cols, lines: List[str]):
        self.name = lines[0][:-1]
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


class Madeline():
    def __init__(self, template_images: List[Image]):
        W = []
        for template_image in template_images:
            W.append(np.reshape(template_image.data, (-1,)) / np.count_nonzero(template_image.data))
        self.W = np.array(W)
        self.labels = [image.name for image in template_images]

    def __call__(self, image):
        return np.matmul(self.W, np.reshape(image.data, (-1)))


if __name__ == "__main__":
    madeline = Madeline(Image.read_from_file("template_images.txt"))
    test_images = Image.read_from_file("test_images.txt")
    for test_image in test_images:
        pred = madeline(test_image)
        print(f"{test_image.name} : {list(zip(madeline.labels, pred))}")
