import cv2
import numpy as np
from skimage.feature import hog


def extract_hog(image, config):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, tuple(config["image_size"]), interpolation=cv2.INTER_AREA)
    options = config["hog"]
    return hog(
        image,
        orientations=options["orientations"],
        pixels_per_cell=tuple(options["pixels_per_cell"]),
        cells_per_block=tuple(options["cells_per_block"]),
        block_norm="L2-Hys",
    )
