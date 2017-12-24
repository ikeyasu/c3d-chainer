import chainer
import os
import numpy as np
import pickle

from chainer import serializers

import models.VGG
import models.C3D
import chainer.links as L
import caffe_model.caffe_pb2 as caffe

from datasets.UCF11 import UCF11, UCF11Sub, LABELS

NUM_OF_CLASSES = 487 #11


model = L.Classifier(models.C3D.C3D(NUM_OF_CLASSES))
#chainer.serializers.load_npz('mlp.model', model)
model.predictor.train = False

conv_layers_indx = [1, 4, 7, 9, 12, 14, 17, 19]
fc_layers_indx = [22, 25, 28]

p = caffe.NetParameter()
p.ParseFromString(open('caffe_model/conv3d_deepnetA_sport1m_iter_1900000', 'rb').read())

for i in conv_layers_indx:
    layer = p.layers[i]
    print(layer.name)
    weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
    weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
        layer.blobs[0].num,
        layer.blobs[0].channels,
        layer.blobs[0].length,
        layer.blobs[0].height,
        layer.blobs[0].width,
        )
    np.copyto(model.predictor[layer.name].conv.W.data, weights_p)
    np.copyto(model.predictor[layer.name].conv.b.data, weights_b)

for i in fc_layers_indx:
    layer = p.layers[i]
    name = layer.name.replace('-1', '')
    print(name)
    weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
    weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num,
            layer.blobs[0].channels,
            layer.blobs[0].length,
            layer.blobs[0].height,
            layer.blobs[0].width)
    np.copyto(model.predictor[name].W.data, weights_p)
    np.copyto(model.predictor[name].b.data, weights_b)

print('saving the model')
serializers.save_npz('mlp.model', model)
