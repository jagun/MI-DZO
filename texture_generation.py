import string
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.feature_extraction import image as img_features


class Pixel:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.error = 0
        self.value = 0
        self.neighbor_count = 0


def get_unfilled_pixel_neighbours_list(image, window_size):
    lst = []
    half_window = window_size / 2
    for i in xrange(half_w, image.shape[0] - half_w):
        for j in xrange(half_w, image.shape[1] - half_w):
            if image[i, j] != 0:
                continue
            pixel = Pixel(i, j)
            neiborhood = image[pixel.x - half_window:pixel.x + half_window + 1,
                                      pixel.y - half_window:pixel.y + half_window + 1]
            pixel.neighbor_count = np.count_nonzero(neiborhood)
            if pixel.neighbor_count > 0:
                lst.append(pixel)

    np.random.shuffle(lst)

    return sorted(lst, key=lambda x: x.neighbor_count, reverse=True)


def get_candidate_list(template, gauss_mask, patches, win_size):
    threshold = 0.1

    mask = template != 0
    weight = gauss_mask * mask
    total_weight = np.sum(weight)

    if total_weight == 0:
        total_weight = 1

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
    unfilled = image_size - np.count_nonzero(image)
    max_err_treshold = 0.3
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
            if candidate.error <= max_err_treshold:
                image[pixel.x, pixel.y] = candidate.value
                flag = True
                unfilled -= 1
                print 'filled %f' % ((image_size - float(unfilled)) / image_size)

            progress = (image_size - float(unfilled)) / image_size
            if progress - prev >= 0.1:
                mpimg.imsave('tmp/progress_%d.png' % iteration, image, cmap=plt.get_cmap('gray'))
                iteration += 1
                prev = progress

        if not flag:
            max_err_treshold *= 1.1

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