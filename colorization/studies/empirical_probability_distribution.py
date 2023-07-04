# References
    # https://github.com/Time0o/colorful-colorization/blob/9cbbc9fb7518bd92c441e36e45466cfd663fa9db/colorization/cielab.py

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from skimage.color import rgb2lab
import cv2
import numpy as np

from image_utils import (
    load_image,
    _to_pil,
    show_image,
    save_image,
    _figure_to_array
)


def ab_color_space_histogram(data_dir):
    data_dir = Path(data_dir)

    hist = np.zeros(shape=(2 ** 8, 2 ** 8), dtype="uint32")
    for img_path in tqdm(list(data_dir.glob("**/*.jpg"))):
        img = load_image(img_path)
        lab_img = rgb2lab(img).round().astype("int8")

        ab_vals = lab_img[..., 1:].reshape(-1, 2)
        indices = ab_vals + 2 ** 7
        np.add.at(hist, (indices[:, 0], indices[:, 1]), 1)
    return hist


def empirical_probability_distribution(hist):
    copied = hist.copy()
    copied[copied == 0] = 1
    log_scaled = np.log10(copied)
    return log_scaled


def empirical_probability_distribution_plot(hist):
    copied = hist.copy()
    copied[copied == 0] = 1
    log_scaled = np.log10(copied)

    fig, axes = plt.subplots(figsize=(8, 8))
    axes.pcolormesh(np.arange(-128, 128), np.arange(-128, 128), log_scaled, cmap="jet")
    axes.set(xticks=range(-125, 125 + 1, 10), yticks=range(-125, 125 + 1, 10))
    axes.tick_params(axis="x", labelrotation=90, labelsize=8)
    axes.tick_params(axis="y", labelsize=8)
    axes.invert_yaxis()
    axes.grid(axis="x", color="White", alpha=1, linestyle="--", linewidth=0.5)
    axes.grid(axis="y", color="White", alpha=1, linestyle="--", linewidth=0.5)

    heatmap = _figure_to_array(fig)
    heatmap = cv2.pyrDown(heatmap)
    return heatmap


if __name__ == "__main__":
    hist = ab_color_space_histogram("/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages")
    # hist = ab_color_space_histogram("/Users/jongbeomkim/Desktop/workspace/visual_representation_learning/simclr/voc2012_image_views")
    prob_dist_plot = empirical_probability_distribution_plot(hist)
    show_image(prob_dist_plot)
    save_image(
        img=prob_dist_plot,
        path="/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/visual_representation_learning/colorization/studies/voc2012_empirical_probability_distribution.jpg"
    )
