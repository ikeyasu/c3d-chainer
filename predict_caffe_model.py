import chainer
import os
import numpy as np
import pickle

import models.VGG
import models.C3D
import chainer.links as L
import caffe_model.caffe_pb2 as caffe

from datasets.UCF11 import UCF11, UCF11Sub, LABELS

NUM_OF_CLASSES = 487 #11


model = L.Classifier(models.C3D.C3D(NUM_OF_CLASSES))
#chainer.serializers.load_npz('mlp.model', model)
#model.predictor.train = False

conv_layers_indx = [1, 4, 7, 9, 12, 14, 17, 19]
fc_layers_indx = [22, 25, 28]

p = caffe.NetParameter()
p.ParseFromString(open('caffe_model/conv3d_deepnetA_sport1m_iter_1900000', 'rb').read())

for i in conv_layers_indx:
    layer = p.layers[i]
    weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
    weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
        layer.blobs[0].num,
        layer.blobs[0].channels,
        layer.blobs[0].length,
        layer.blobs[0].height,
        layer.blobs[0].width,
        )
    #print(weights_b.shape)
    #print(model.predictor[layer.name].conv.b.shape)
    np.copyto(model.predictor[layer.name].conv.W.data, weights_p)
    np.copyto(model.predictor[layer.name].conv.b.data, weights_b)

for i in fc_layers_indx:
    layer = p.layers[i]
    weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
    weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num,
            layer.blobs[0].channels,
            layer.blobs[0].length,
            layer.blobs[0].height,
            layer.blobs[0].width)#[0,0,0,:,:].T
    name = layer.name.replace('-1', '')
    print(name)
    print(weights_p.shape)
    print(model.predictor[name].W.shape)
    print(weights_b.shape)
    print(model.predictor[name].b.shape)
    np.copyto(model.predictor[name].W.data, weights_p)
    np.copyto(model.predictor[name].b.data, weights_b)

with open('caffe_model/labels.txt', 'r') as f:
  labels = [line.strip() for line in f.readlines()]
print('Total labels: {}'.format(len(labels)))

import cv2
cap = cv2.VideoCapture('caffe_model/dM06AMFLsrc.mp4')

vid = []
while True:
  ret, img = cap.read()
  if not ret:
    break
  vid.append(cv2.resize(img, (171, 128)))
vid = np.array(vid, dtype=np.float32)
x = vid[2000:2016, 8:120, 30:142, :].transpose((3, 0, 1, 2))
output = model.predictor(np.array([x]))

pos = np.argmax(output.data, axis=1)[0]
print('Position of maximum probability: {}'.format(pos))
print('Maximum probability: {:.5f}'.format(max(output.data[0])))
print('Corresponding label: {}'.format(labels[pos]))

# sort top five predictions from softmax output
top_idxs = np.argsort(output.data[0])[::-1][:5]  # reverse sort and take five largest items
print('\nTop 5 probabilities and labels:')
_ = [print('{:.5f} {}'.format(output.data[0][i], labels[i])) for i in top_idxs]
