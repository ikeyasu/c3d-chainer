import argparse
import chainer
import numpy as np
import cv2

import models.VGG
import models.C3D

from datasets.UCF11 import UCF11, UCF11Sub


def get_data(path, mean):
    base = UCF11Sub(path)
    base.init(crop_size=112, crop_pos=None, flip=False, mask=None)
    frames = 9
    data = np.array([base[j] - mean for j in range(frames)])
    images = data.transpose((1, 0, 2, 3))
    images *= (1.0 / 255.0)  # Scale to [0, 1]
    return images


def get_ucf11_dataset(path, mean):
    return UCF11(path, mean=mean, frames=9, data_aug=False)


def get_data_from_video(path, mean):
    cap = cv2.VideoCapture(path)
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        vid.append(cv2.resize(img, (171, 128)))
    vid = np.array(vid, dtype=np.float32)
    frames = vid[2000:2016, 8:120, 30:142, :]
    if mean is not None:
        frames = np.array([frames[j] - mean for j in range(frames)])
    x = frames.transpose((3, 0, 1, 2))
    return x


def main():
    archs = {
        'vgg3d': models.VGG.VGG3D,
        'c3d': models.C3D.C3D
    }
    parser = argparse.ArgumentParser(description='Chainer ConvolutionND example:')
    parser.add_argument('--image-dir', '-d',
                        help='Image dir path (UCF11 dataset only)')
    parser.add_argument('--dataset-dir', '-i',
                        help='Dataset dir path (UCF11 dataset only)')
    parser.add_argument('--video', '-v',
                        help='Input video file')
    parser.add_argument('--labels',
                        help='Labels file. (No need labels file for UCF11 dataset)')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='vgg3d',
                        help='Architecture (vgg3d, c3d)')
    parser.add_argument('--model', '-e', required=True,
                        help='model file')
    parser.add_argument('--mean', '-m', required=True,
                        help='Mean file (computed by compute_mean.py)')
    args = parser.parse_args()

    if args.labels:
        with open(args.labels, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        labels = UCF11.LABELS
    print("Loaded {} labels.".format(len(labels)))
    model = chainer.links.Classifier(archs[args.arch](len(labels) if args.labels else UCF11.NUM_OF_CLASSES))
    chainer.serializers.load_npz(args.model, model)
    model.predictor.train = False

    mean = np.load(args.mean)
    if args.image_dir:
        data = get_data(args.image_dir, mean)
        y = model.predictor(data[np.newaxis, :, :, :, :])
        results = np.argmax(y.data, axis=1)
        print(labels[results[0]])
    if args.dataset_dir:
        data = get_ucf11_dataset(args.dataset_dir, mean)
        for i in range(0, len(data), min(16, len(data))):
            y = model.predictor(np.asarray([data[i + j][0] for j in range(min(16, len(data) - i))]))
            results = np.argmax(y.data, axis=1)
            for (j, v) in enumerate(results):
                print("{},{}".format(data.dir_list[i + j], labels[v]))
    if args.video:
        x = get_data_from_video(args.video, mean)
        y = model.predictor(np.array([x]))
        pos = np.argmax(y.data, axis=1)[0]
        print('Position of maximum probability: {}'.format(pos))
        print('Maximum probability: {:.5f}'.format(max(y.data[0])))
        print('Corresponding label: {}'.format(labels[pos]))
        # sort top five predictions from softmax output
        top_idxs = np.argsort(y.data[0])[::-1][:5]
        print('\nTop 5 probabilities and labels:')
        _ = [print('{:.5f} {}'.format(y.data[0][i], labels[i])) for i in top_idxs]


if __name__ == '__main__':
    main()
