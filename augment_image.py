from scipy import ndimage
import numpy as np
from PIL import Image


def flip_up_down(image, coord):
    """ Flip the image along the X axis """
    img_ud_flipped = np.flipud(image)
    y, x = coord
    coord = ((img.shape[0] - 1) - y, x)
    return img_ud_flipped, coord


def flip_left_right(img, coord):
    """ Flip the image along the Y axis """
    img_lr_flipped = np.fliplr(img)
    y, x = coord
    width = img.shape[1]
    coord = (y, (width - 1) - x)
    return img_lr_flipped, coord


def rotate_degrees(image, deg, coord):
    """ Rotate the image by a number of degrees """
    rotated = ndimage.rotate(image, deg, reshape=False)
    patch = image[:10, :10, :]
    fill_color = np.mean(patch, axis=(1, 0))
    mask = np.all(rotated == [0, 0, 0], axis=-1)
    rotated[mask] = fill_color

    height, width = image.shape[0], image.shape[1]
    y, x = coord
    xm, ym = width // 2, height // 2
    a = np.radians(deg)
    xr = (x - xm) * np.cos(a) + (y - ym) * np.sin(a) + xm,
    yr = -(x - xm) * np.sin(a) + (y - ym) * np.cos(a) + ym
    return rotated, (xr, yr)


def increase_brightness(img, coord):
    """ Increase brightness by 50% """
    img_brightened = 1.5 * img / 255.0
    return img_brightened, coord


def decrease_brightness(img, coord):
    """ Decrease brightness by 50% """
    img_darkened = 0.5 * img / 255.0
    return img_darkened, coord


def reduce_color(img, coord):
    """ Reduce color vibrance """
    img_32 = img // 32 * 32
    return img_32, coord


def add_gaussian_noise(img, coord):
    """ Add gaussian noise to the channels"""
    mean = 0
    sigma = np.std(img) * 0.1
    gaussian = np.random.normal(mean, sigma, (128, 128, 3))
    noisy_image = img + gaussian
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image, coord


def smooth(img, coord):
    return ndimage.gaussian_filter(img, sigma=1), coord
