import numpy as np
import PIL.Image


def pil_to_np(pil_image):
    """
    :param pil_image: PIL Image object to be converted
    :return: Numpy array representing the PIL Image
    """
    return np.array(pil_image)


def np_to_pil(numpy_array):
    """
    :param numpy_array: NumPy array representing an image
    :return: PIL Image object converted from the input NumPy array
    """
    return PIL.Image.fromarray(numpy_array)
