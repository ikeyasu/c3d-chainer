import random

import chainer
import os
import numpy as np
import pickle


def get_dir_list(path):
    return [os.path.join(path, i) for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]


class UCF11Sub(chainer.dataset.DatasetMixin):

    """
    Args:
        image_dir (string): path of image directory
    """
    def __init__(self, image_dir):
        self.base = chainer.datasets.ImageDataset(os.listdir(image_dir), root=image_dir)
        self.crop_size = None
        self.crop_pos = None
        self.flip = False
        self.mask = None
        self.mask_value = 0

    """
    Args:
        crop_size (int): crop size. An image is cropped by rectangle. None is disabled.
        crop_pos ((int, int)): crop starting position. None means clipping center.
        flip (boolean): Random flip. False is disabled.
        mask ((int, int, int, int)): Mask area. (left, top, right, bottom). None is disabled.
        mask_value (int): mask color.
    """
    def init(self, crop_size=None, crop_pos=None, flip=False, mask=None, mask_value=0):
        self.crop_size = crop_size
        self.crop_pos = crop_pos
        self.flip = flip
        self.mask = mask  # (left, top, right, bottom)
        self.mask_value = mask_value

    def __len__(self):
        return len(self.base)

    def size(self):
        _, h, w = self.base[0].shape
        return w, h

    def get_example(self, i):
        image = self.base[i]

        # flip
        if self.flip:
            image = image[:, :, ::-1]

        # crop
        if self.crop_size is None:
            return image
        if self.crop_pos is not None:
            left = self.crop_pos[0]
            top = self.crop_pos[1]
        else:  # center
            _, h, w = image.shape
            left = (w - self.crop_size) // 2
            top = (h - self.crop_size) // 2
        bottom = top + self.crop_size
        right = left + self.crop_size
        image = image[:, top:bottom, left:right]

        # random erase
        if self.mask is not None:
            image[:, self.mask[1]:self.mask[3], self.mask[0]:self.mask[2]].fill(self.mask_value)

        # test
        # import scipy.misc
        # scipy.misc.imsave('outfile_{}.jpg'.format(i), image.transpose(1, 2, 0))
        return image


class UCF11(chainer.dataset.DatasetMixin):
    NUM_OF_CLASSES = 11
    LABELS = ["biking", "diving", "golf", "juggle", "jumping", "riding", "shooting", "spiking", "swing", "tennis",
              "walk_dog"]

    """
    dataset example:
    videos/v_biking_01_01/00001.jpg
    videos/v_biking_01_01/00002.jpg
    """

    def __init__(self, path, mean=None, frames=6, data_aug=True):
        self.dir_list = get_dir_list(path)
        self.mean = mean.astype('f') if type(mean) is np.ndarray else None
        self.frames = frames
        self.data_aug = data_aug

    def __len__(self):
        return len(self.dir_list)

    @staticmethod
    def save_obj(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def get_label_idx(path):
        img_dir = os.path.split(path)[-1]
        for (i, label) in enumerate(UCF11.LABELS):
            if img_dir.find(label) > -1:
                return i
        return 0

    # http://xkumiyu.hatenablog.com/entry/numpy-data-augmentation#Random-Erasing
    @staticmethod
    def random_erasing(w, h, p=0.5, s=(0.02, 0.4), r=(0.3, 3)):
        # Mask or not
        if np.random.rand() > p:
            return None, None
        # choose color to mask
        mask_value = np.random.randint(0, 256)

        # choose mask size in (0.02-0.4) times
        mask_area = np.random.randint(h * w * s[0], h * w * s[1])

        # choose mask aspect ratio
        mask_aspect_ratio = np.random.rand() * r[1] + r[0]

        # choose height/width
        mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
        if mask_height > h - 1:
            mask_height = h - 1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w - 1:
            mask_width = w - 1

        top = np.random.randint(0, h - mask_height)
        left = np.random.randint(0, w - mask_width)
        bottom = top + mask_height
        right = left + mask_width
        return (left, top, right, bottom), mask_value

    def get_example(self, i):
        image_dir = self.dir_list[i]
        base = UCF11Sub(image_dir)
        w, h = base.size()
        crop_size = int(w * 0.7)
        if self.data_aug:
            mask_area, mask_value = UCF11.random_erasing(crop_size, crop_size)
            base.init(
                crop_size=crop_size,
                crop_pos=(random.randint(0, w - crop_size), random.randint(0, h - crop_size)),
                flip=(random.randint(0, 1) == 0),
                mask=mask_area,
                mask_value=mask_value
            )
        else:
            base.init(
                crop_size=crop_size,
                crop_pos=None,
                flip=False,
                mask=None,
                mask_value=0
            )

        if len(base) < self.frames:
            raise "{} does not have {} frames.".format(image_dir, self.frames)
        if self.mean is None:
            data = np.array([base[j] for j in range(self.frames)])
        else:
            data = np.array([base[j] - self.mean for j in range(self.frames)])
        if len(data.shape) < 4:
            raise "{} is invalid.".format(image_dir)
        images = data.transpose((1, 0, 2, 3))

        images *= (1.0 / 255.0)  # Scale to [0, 1]
        return images, UCF11.get_label_idx(image_dir)
