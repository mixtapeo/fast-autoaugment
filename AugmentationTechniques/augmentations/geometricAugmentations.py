import PIL
import PIL.ImageOps
import PIL.Image


def rotate(img, v=30):
    """
    :param img: PIL image.
    :param v: The angle by which the image should be rotated.
              (Positive for counterclockwise, negative for counterclockwise rotation).
    :return: Rotated PIL image.
    """
    return img.rotate(v)


def flip(img, vertical=True):
    """
    :param img: PIL image.
    :param vertical: A boolean flag indicating whether to flip the image vertically (True) or horizontally (False).
    :return: Flipped PIL image.
    """
    return PIL.ImageOps.flip(img) if vertical else PIL.ImageOps.mirror(img)


def translate(img, x=True, v=0.2):
    """
    :param img: PIL image.
    :param x: A boolean flag indicating whether to translate the image along x (True) or y (False).
    :param v: The percentage by which the image should be translated (Positive for left/up, Negative for right/down).
    :return: Translated PIL image.
    """
    if x:
        v = v * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    else:
        v = v * img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def shear(img, x=True, v=0.2):
    """
    :param img: PIL image.
    :param x: Boolean flag indicating whether shearing is performed along x-axis.
    :param v: Shearing factor.
    :return: Sheared PIL image.
    """
    if x:
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
    else:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
