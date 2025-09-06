import errno

import cv2
import logging
import os
import time

import numpy as np

from PIL import Image

# ----------------------------------------------------------------------------#

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    if os.path.splitext(path)[1] == '.tif':
        img = np.asarray(Image.open(path), dtype=int)
        img[np.where(img == 0)] = 8
        img[np.where(img == 1)] = 0
        img = np.stack((img,)*3, axis=-1)
        img[:, :, 1] = 0
        img[:, :, 2] = 0
    else:
        img = cv2.imread(path)

    if img is None:
        raise Exception("Image is empty or corrupted", path)

    return img

# ----------------------------------------------------------------------------#

def prepare_image(img, cropping=True, vertical=False):
    # -------------------------------
    start = time.time()
    # -------------------------------

    # Converting to grayscale if it's not already.
    if img.ndim == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Invert image: make text white, background black.
    img_gray = cv2.bitwise_not(img_gray)

    # Threshold to binary.
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to 3-channel for compatibility.
    img = cv2.merge([img_bin, img_bin, img_bin])

    # Optional cropping to content.
    if cropping:
        coords = cv2.findNonZero(img_bin)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]
            cv2.imwrite("04_test2.png", img)
        else:
            logging.warning("No foreground content found for cropping.")

    # Rotate 90 degrees to the left the image (for vertical scripts such as Chinese).
    if vertical:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # -------------------------------
    stop = time.time()
    logging.info("finished after: {diff} s".format(diff=stop - start))
    # -------------------------------

    return img
