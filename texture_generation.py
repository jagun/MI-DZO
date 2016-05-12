# constants
import numpy as np
import argparse

ERR_TRESHOLD = 0.1
MAX_ERR_TRESHOLD = 0.3


class Pixel:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.error = 0
        self.value = 0


class TextureGenerator:
    def __init__(self, image, window_size):
        self.sample_image = image
        self.image = np.array()
        self.windows_size = window_size

    def __init_image__(self):
        # find 3x3
        pass


def has_filled_neighbours(image, pos, window_size):
    flag = False
    for i in xrange(pos.x, pos.x + window_size):
        for j in xrange(pos.y, pos.y + window_size):
            if image[i, j] != 0:
                flag = True
    return flag


def get_unfilled_neighbours(image, window_size):
    """

    :param image:
    :param window_size:
    :return:
    """
    lst = []
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            pos = Pixel(i, j)
            if has_filled_neighbours(image, pos, window_size):
                lst.append(pos)
    np.random.shuffle(lst)
    return lst


def get_neighborhood_window(image, pixel, window_size):
    """

    :param image:
    :param pixel:
    :param window_size:
    :return:
    """
    template = np.array((window_size, window_size))

    for i in xrange(pixel.x, pixel.x + window_size):
        for j in xrange(pixel.y, pixel.y + window_size):
            template[i - pixel.x, j - pixel.y] = image[i, j]

    return template


def get_candidate_list(template, sample_image, window_size):
    # init distance matrix
    ssd = np.zeros((sample_image))

    # construct a bitmask
    mask = bitmask(template, window_size)
    g_mask = gauss_mask(window_size)
    total_wight = np.sum(mask * g_mask)

    pass


def bitmask(template, window_size):
    mask = np.zeros((window_size, window_size))
    for i in xrange(window_size):
        for j in xrange(window_size):
            if template[i, j] != 0:
                mask[i, j] = 1
    return mask

def gauss_2d(x, y, center, sigma):
    """

    :param x: x
    :param y: x
    :param center:
    :param sigma: width of distribution
    :return: gauss
    """
    left = np.power(x - center, 2) / (2 * np.power(sigma, 2))
    right = np.power(y - center, 2) / (2 * np.power(sigma, 2))
    return np.exp(-(left + right))


def gauss_mask(window_size):
    """

    :param window_size: better to be an even number
    :return:
    """
    sample = np.zeros((window_size, window_size))
    center = (window_size / 2) + 1
    for x in xrange(window_size):
        for y in xrange(window_size):
            sample[x, y] = gauss_2d(x, y, center)
    return sample


# numpy arrays
def grow_image(sample_image, image, window_size):
    while True:
        flag = False
        pixelList = get_unfilled_neighbours(image)
        for pixel in pixelList:
            template = get_neighborhood_window(image, pixel, window_size)
            candidates = get_candidate_list(template, sample_image)
            candidate = np.random.choice(candidates)
            if candidate.error < MAX_ERR_TRESHOLD:
                image[pixel.x, pixel.y] = candidate.value
                flag = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    window_size = 11