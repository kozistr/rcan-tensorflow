import numpy as np
import scipy.misc


def transform(images, inv_type='255'):
    if inv_type == '255':
        images /= 255.
    elif inv_type == '127':
        images = (images / 127.5) - 1.
    else:
        raise NotImplementedError("[-] Only 255 and 127")
    return images.astype(np.float32)


def inverse_transform(images, inv_type='255'):
    if inv_type == '255':    # [ 0  1]
        images *= 255
    elif inv_type == '127':  # [-1, 1]
        images = (images + 1) * (255 / 2.)
    else:
        raise NotImplementedError("[-] Only 255 and 127")

    # clipped by [0, 255]
    images[images > 255] = 255
    images[images < 0] = 0
    return images.astype(np.uint8)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]

    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[0]
        j = idx // size[1]
        img[h * j:h * (j + 1), w * i:w * (i + 1), :] = image
    return img


def split(image, n_patch=16):
    h, w, c = image.shape
    assert h == w

    patch = int(np.sqrt(n_patch))
    patch_size = h // patch

    patch_images = [image[patch_size * j: patch_size * (j + 1), patch_size * i: patch_size * (i + 1), :]
                    for j in range(patch) for i in range(patch)]
    return patch_images


def save_image(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def save_images(images, size, image_path, inv_type='255', use_inverse=False):
    images = inverse_transform(images, inv_type) if use_inverse else images
    return save_image(images, size, image_path)


def img_save(img, path, inv_type='255', use_inverse=False):
    img = inverse_transform(img, inv_type) if use_inverse else img
    return scipy.misc.imsave(path, img)


def rotate(images):
    images = np.append(images, [np.fliplr(image) for image in images], axis=0)  # 180 degree
    images = np.append(images, [np.rot90(image) for image in images], axis=0)   # 90 degree
    return images
