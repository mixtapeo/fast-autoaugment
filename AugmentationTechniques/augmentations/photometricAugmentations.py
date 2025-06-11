from albumentations import FancyPCA, RandomGamma, CLAHE
import PIL
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps

import utils.utils as utils

def brightness(img, v=1.5):
    """
    :param img: PIL image.
    :param v: Factor to adjust brightness level, where v > 1 increases brightness and 0 < v < 1 decreases brightness.
    :return: PIL image with adjusted brightness.
    """
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def color_space(img, v=0.5):
    """
    :param img: PIL image.
    :param v: Factor by which to enhance color, where value of 1.0 gives the original image and 0.0 gives black/white.
    :return: Color-enhanced PIL image.
    """
    return PIL.ImageEnhance.Color(img).enhance(v)


def equalize(img):
    """
    :param img: PIL image.
    :return: Equalized PIL image.
    """
    return PIL.ImageOps.equalize(img)


def contrast(img, v=1.5):
    """
    :param img: PIL image.
    :param v: The factor by which to adjust the contrast, 0.0 gives gray image, 1.0 gives original image and a greater
              values increase contrast.
    :return: PIL image with adjusted contrast.
    """
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def auto_contrast(img):  # Not from one of the papers
    """
    :param img: PIL image.
    :return: PIL image after applying auto-contrast adjustment.
    """
    return PIL.ImageOps.autocontrast(img)


def invert(img):
    """
    :param img: PIL image.
    :return: The inverted PIL image.
    """
    return PIL.ImageOps.invert(img)


def pca_jitter(img, v=0.1):
    """
    :param img: PIL image.
    :param v: Strength of the PCA perturbation (Standard deviation of noise).
    :return: A PIL image with PCA Jittering applied.
    """
    img_np = utils.pil_to_np(img)
    transform = FancyPCA(alpha=v, p=1.0)

    return utils.np_to_pil(transform(image=img_np)["image"])


def gamma_correction(img, v=(80, 120)):
    """
    :param img: PIL image.
    :param v: Tuple specifying the gamma range (as percentage, e.g., (80, 120))
    :return: A PIL image with gamma correction applied
    """
    img_np = utils.pil_to_np(img)
    transform = RandomGamma(gamma_limit=v, p=1.0)
    augmented = transform(image=img_np)["image"]

    return utils.np_to_pil(augmented)


def balanced_contrast(img, clip_limit=(1, 4), tile_grid_size=(8, 8)):
    """
    :param img: PIL image.
    :param clip_limit: Threshold for contrast limiting.
    :param tile_grid_size: Size of grid for histogram equalization.
    :return: PIL image with enhanced contrast.
    """
    img_np = utils.pil_to_np(img)
    transform = CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, p=1.0)
    augmented = transform(image=img_np)["image"]

    return utils.np_to_pil(augmented)
