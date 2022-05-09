from PIL import Image
import numpy as np


def import_image(image_file):
    original_img = Image.open(image_file)
    original_img_mat = np.array(original_img)
    original_img_mat = original_img_mat.astype(float)
    return original_img_mat


def create_recovered_image(recov_matrix):
    img = Image.fromarray(recov_matrix)
    img.show()