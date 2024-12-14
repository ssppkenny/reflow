import mydjvulib
import numpy as np
from PIL import Image


def get_image(pageno, filepath):
    bs, w, h = mydjvulib.get_page(pageno, filepath)
    arr = np.frombuffer(bs, dtype=np.int8).reshape((h, w))
    img = Image.fromarray(arr, "L")
    return img


if __name__ == "__main__":
    img = get_image(1, "dvurog.djvu")
    img.save("img.png")
