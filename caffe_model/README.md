# Prerequirement

```
$ pip install youtube-dl
$ conda config --add channels conda-forge  
$ conda install opencv
```

Ubuntu

```
$ sudo apt-get install protobuf-compiler
```

MacOSX

```
$ brew install protobuf
```

# Downloading

```
$ pushd caffe_model
$ wget http://vlg.cs.dartmouth.edu/c3d/conv3d_deepnetA_sport1m_iter_1900000
$ wget https://raw.githubusercontent.com/gtoderici/sports-1m-dataset/master/labels.txt
$ wget https://raw.githubusercontent.com/facebook/C3D/master/C3D-v1.0/src/caffe/proto/caffe.proto
$ wget --content-disposition https://github.com/axon-research/c3d-keras/blob/master/data/train01_16_128_171_mean.npy.bz2?raw=true
$ bunzip2 train01_16_128_171_mean.npy.bz2
$ youtube-dl -f mp4 https://www.youtube.com/watch?v=dM06AMFLsrc -o dM06AMFLsrc.mp4
$ popd
```

# Compiling caffe.proto

```
$ pushd caffe_model
$ sed -i '1s/^/syntax = "proto2";\n/' caffe.proto # inserting syntax version on the head.
$ protoc --python_out=. caffe.proto
$ popd
```

# Converting caffe model to chainer model

```
$ # on the top of this repository's dir.
$ convert_caffe_model.py
```
