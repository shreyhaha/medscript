"""
features.py - turns an image into numbers for the classic-ML baseline.

HOG = Histogram of Oriented Gradients.
  - Split the image into small cells (e.g. 8x8 pixels).
  - In each cell, measure the direction of the edges (gradients) and build a
    histogram of 9 directions (0-180 degrees).
  - Normalise neighbouring cells together (blocks of 2x2 cells) so the result
    does not depend on how dark the ink is.
  -> One long vector that describes the SHAPE of the strokes.

Why HOG for the baseline: handwriting is basically strokes = edges, and HOG
is exactly an edge-direction descriptor. It was the standard approach before
deep learning (e.g. Dalal & Triggs 2005 for pedestrian detection).
ECE link: gradients are computed with derivative filters (like Sobel), which
is plain 2-D convolution / signal processing.
"""
import numpy as np
from skimage.feature import hog


def hog_features(img, pixels_per_cell=8):
    """img: preprocessed uint8 image -> 1-D feature vector."""
    return hog(
        img.astype(np.float32) / 255.0,
        orientations=9,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def hog_matrix(images, pixels_per_cell=8):
    """List of images -> 2-D array (n_images, n_features)."""
    return np.array([hog_features(im, pixels_per_cell) for im in images])
