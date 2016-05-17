import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.feature_extraction import image as img_features


class Pixel:
    """
    Pixel class to store information about pixel
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.error = 0
        self.value = 0
        self.neighbor_count = 0


def get_unfilled_pixel_neighbours_list(image, window_size):
    """
    Returns list of pixels that have non-zero pixels in their neighborhood
    :param image: image to fill
    :param window_size: size of neighborhood window
    :return: list of pixels sorted by neighbor count descending
    """
    lst = []
    half_window = window_size / 2
    for i in xrange(half_w, image.shape[0] - half_w):
        for j in xrange(half_w, image.shape[1] - half_w):
            if image[i, j] != 0:
                continue
            pixel = Pixel(i, j)
            neighborhood = image[pixel.x - half_window:pixel.x + half_window + 1,
                                      pixel.y - half_window:pixel.y + half_window + 1]
            pixel.neighbor_count = np.count_nonzero(neighborhood)
            if pixel.neighbor_count > 0:
                lst.append(pixel)

    np.random.shuffle(lst)

    return sorted(lst, key=lambda x: x.neighbor_count, reverse=True)


def get_candidate_list(template, gauss_mask, patches, win_size):
    """
    Returns list of possible values for current pixel
    :param template: current pixel's window
    :param gauss_mask: 2d gauss mask with center in the middle of window, size win_zize x win_zize
    :param patches: sample image patches, candidates for pixel value
    :param win_size: size of neighborhood window
    :return: list of candidate pixels
    """
    threshold = 0.1

    mask = template != 0
    weight = gauss_mask * mask
    total_weight = np.sum(weight)

    if total_weight == 0:
        total_weight = 1

    # place where magic happens, vectorized multiplication
    # stores distance for each pixel in sample image
    SSD = np.sum(np.multiply((patches - template) ** 2, weight), axis=(1, 2)) / float(total_weight)

    min_err = SSD.min()
    candidates = []
    for i, err in enumerate(SSD):
        if err <= min_err * (1 + threshold):
            candidate = Pixel(0, 0)
            candidate.error = err
            candidate.value = patches[i, win_size / 2, win_size / 2]
            candidates.append(candidate)
    return candidates


def get_gauss_mask(window_size):
    """
    Returns gauss mask of size window_size x window_size
    :param window_size: size of neighborhood window
    :return: 2d numpy array
    """
    gauss = np.zeros((window_size, window_size))
    center = window_size / 2
    sigma = window_size / 6.4
    for x in xrange(window_size):
        for y in xrange(window_size):
            left = np.power(x - center, 2) / (2 * np.power(sigma, 2))
            right = np.power(y - center, 2) / (2 * np.power(sigma, 2))
            gauss[x, y] = np.exp(-(left + right))
    return gauss


def grow_image(sample_image, image, image_size, window_size):
    """
    Fills image from sample
    :param sample_image: sample image to grow texture from
    :param image: target image to fill in, padded with zeros to halde corners
    :param image_size: actual image size to keep track progress
    :param window_size: size of neighborhood window
    :return: filled image
    """
    unfilled = image_size - np.count_nonzero(image)
    max_err_threshold = 0.3
    iteration = 1
    prev = 0

    half_window = window_size / 2
    g_mask = get_gauss_mask(window_size)
    patches = img_features.extract_patches_2d(sample_image, (window_size, window_size))

    while unfilled:
        flag = False

        pixel_list = get_unfilled_pixel_neighbours_list(image, window_size)

        for pixel in pixel_list:
            template = image[pixel.x - half_window:pixel.x + half_window + 1,
                       pixel.y - half_window:pixel.y + half_window + 1]

            candidates = get_candidate_list(template, g_mask, patches, window_size)

            if len(candidates) == 0:
                continue

            candidate = np.random.choice(candidates)
            if candidate.error <= max_err_threshold:
                image[pixel.x, pixel.y] = candidate.value
                flag = True
                unfilled -= 1
                print 'filled %f' % ((image_size - float(unfilled)) / image_size)

        if not flag:
            max_err_threshold *= 1.1

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='sample file path')
    parser.add_argument('window_size', type=int, help='sliding window size')
    parser.add_argument('width', type=int, help='generated texture size')
    parser.add_argument('height', type=int, help='generated texture size')

    args = parser.parse_args()

    window_size = args.window_size
    filename = args.input_file
    half_w = window_size / 2

    size = (args.width, args.height)
    # For us to be able to fill in corners pad image size with half of the window
    # otherwise we will not be able to get template for upper and bottom rows
    # and left / right columns
    image_size = (args.width + half_w * 2, args.height + half_w * 2)

    img = mpimg.imread(filename)

    # rgb case, transform to grayscale
    if len(img.shape) > 2:
        gray_image = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]) / 255.0
        img = gray_image
    else:
        img /= 255.0

    sample = img

    mpimg.imsave('sample.png', sample, cmap=plt.get_cmap('gray'))

    # new image we would like to generate
    target = np.zeros(image_size)

    # seed the image
    rand_x = np.random.randint(0, img.shape[0] - 3)
    rand_y = np.random.randint(0, img.shape[1] - 3)
    seed_size = 3
    seed = sample[rand_x:rand_x + seed_size, rand_y:rand_y + seed_size]
    center_x = (image_size[0] - seed_size) / 2
    center_y = (image_size[1] - seed_size) / 2

    target[center_x:center_x + seed_size, center_y:center_y + seed_size] = seed
    target = grow_image(sample, target, size[0] * size[1], window_size)
    mpimg.imsave('output.png', target[half_w:-half_w, half_w:-half_w], cmap=plt.get_cmap('gray'))