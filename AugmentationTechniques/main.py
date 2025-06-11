from PIL import Image

import augmentations

imgStr = "img.png"
img = Image.open(imgStr)

modifiedImg = augmentations.balanced_contrast(img)
img.show()
