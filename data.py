from pathlib import Path
import struct
import numpy as np
import torch

import random
import matplotlib.pyplot as plt


def _get_data(root: Path, images_file_name: str, labels_file_name: str, restrict: int = None):
    """
    returns:
        images: torch.Tensor of shape (N, 28*28) of type float32
        labels: torch.Tensor of shape (N,) of type int64
    """
    images_file_path = root.joinpath(images_file_name)
    labels_file_path = root.joinpath(labels_file_name)

    if not images_file_path.is_file():
        raise ValueError(f"Cannot find images file at {images_file_path}")
    if not labels_file_path.is_file():
        raise ValueError(f"Cannot find labels file at {labels_file_path}")

    with labels_file_path.open('rb') as file_labels:
        magic, num = struct.unpack(">II", file_labels.read(8))
        labels = np.fromfile(file_labels, dtype=np.int8)

    with images_file_path.open('rb') as file_images:
        magic, num, rows, cols = struct.unpack(">IIII", file_images.read(16))
        images = np.fromfile(file_images, dtype=np.uint8).reshape(len(labels), rows * cols)

    labels = torch.from_numpy(labels).long()
    images = torch.from_numpy(images).float()

    if restrict is not None:
        images = images[:restrict]
        labels = labels[:restrict]

    assert images.size(0) == labels.size(0)

    return images, labels


def get_train_data(root: Path, restrict: int = None):
    return _get_data(
        root,
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        restrict
    )


def get_test_data(root: Path, restrict: int = None):
    return _get_data(
        root,
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte',
        restrict
    )


def show_sample(images, labels, nb_sample: int = 3):
    """
    preconditions:
        images.size() == (N, 1, 28, 28)
        labels.size() == (N,)
    """
    assert images.size(0) == labels.size(0)

    def plot_img_lbl(img, lbl):
        plt.title(f"Label = {lbl}")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    sample_indices = random.sample(range(images.size(0)), nb_sample)
    num_rows = (nb_sample + 4) // 5
    plt.figure(figsize=(15, num_rows * 3))

    for row in range(num_rows):
        start_idx = row * 5
        end_idx = min(start_idx + 5, nb_sample)
        for idx, i in enumerate(sample_indices[start_idx:end_idx], start=1):
            plt.subplot(num_rows, 5, row * 5 + idx)
            plot_img_lbl(images[i, 0], labels[i])

    plt.show()
