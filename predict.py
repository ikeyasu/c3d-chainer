import argparse
import chainer
import os
import numpy as np
import pickle

import models.VGG
import models.C3D
import chainer.links as L

from datasets.UCF11 import UCF11, UCF11Sub, LABELS

NUM_OF_CLASSES = 11


def get_data(path, mean):
    base = UCF11Sub(path)
    base.init(crop_size=112, crop_pos=None, flip=False, mask=None)
    frames = 9
    data = np.array([base[j] - mean for j in range(frames)])
    images = data.transpose(1, 0, 2, 3)
    images *= (1.0 / 255.0)  # Scale to [0, 1]
    return images

def get_dataset(path, mean):
    return UCF11(path, mean=mean, frames=9, random=False, label_file=None)

def main():
    archs = {
        'vgg3d': models.VGG.VGG3D,
        'c3d': models.C3D.C3D
    }
    parser = argparse.ArgumentParser(description='Chainer ConvolutionND example:')
    parser.add_argument('--image-dir', '-d',
                        help='Image dir path')
    parser.add_argument('--dataset-dir', '-i',
                        help='Dataset dir path')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='vgg3d',
                        help='Architecture (vgg3d, c3d)')
    parser.add_argument('--model', '-e', required=True,
                        help='model file')
    parser.add_argument('--mean', '-m', required=True,
                        help='Mean file (computed by compute_mean.py)')
    args = parser.parse_args()
    model = L.Classifier(archs[args.arch](NUM_OF_CLASSES))
    chainer.serializers.load_npz(args.model, model)
    model.predictor.train = False

    mean = np.load(args.mean)
    if args.image_dir:
        data = get_data(args.image_dir, mean)
        y = model.predictor(data[np.newaxis, :, :, :, :])
        results = np.argmax(y.data, axis=1)
        print(LABELS[results[0]])
    if args.dataset_dir:
        data = get_dataset(args.dataset_dir, mean)
        for i in range(0, len(data), min(16, len(data))):
            y = model.predictor(np.asarray([data[i + j][0] for j in range(min(16, len(data) - i))]))
            results = np.argmax(y.data, axis=1)
            for (j, v) in enumerate(results):
                print("{},{}".format(data.dir_list[i + j], LABELS[v]))


if __name__ == '__main__':
    main()
