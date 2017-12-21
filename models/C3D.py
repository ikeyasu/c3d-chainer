from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L


class Block3D(chainer.Chain):

    """A 3d convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, in_channels, out_channels, ksize, pad=1):
        super(Block3D, self).__init__()
        with self.init_scope():
            self.conv = L.ConvolutionND(ndim=3, in_channels=in_channels, out_channels=out_channels, ksize=ksize, pad=pad,
                                        nobias=True)

    def __call__(self, x):
        h = self.conv(x)
        return F.relu(h)


class C3D(chainer.Chain):

    """A C3D network.

    Args:
        class_labels (int): The number of class labels.

    """

    def __init__(self, class_labels=11):
        super(C3D, self).__init__()
        with self.init_scope():
            self.conv1a = Block3D(3, 64, 2)
            self.conv2a = Block3D(64, 128, 2)
            self.conv3a = Block3D(128, 256, 2)
            self.conv3b = Block3D(256, 256, 2)
            self.conv4a = Block3D(256, 512, 2)
            self.conv4b = Block3D(512, 512, 2)
            self.conv5a = Block3D(512, 512, 2)
            self.conv5b = Block3D(512, 512, 2)
            self.fc6 = L.Linear(None, 4096, nobias=True)
            self.fc7 = L.Linear(None, 4096, nobias=True)
            self.fc8 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        h = self.conv1a(x)
        h = F.max_pooling_nd(h, ksize=(1, 2, 2), stride=(1, 2, 2))
        h = self.conv2a(h)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.conv3a(h)
        h = self.conv3b(h)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.conv4a(h)
        h = self.conv4b(h)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.conv5a(h)
        h = self.conv5b(h)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.fc6(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc7(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc8(h)
        return h
        #return F.softmax(h)
