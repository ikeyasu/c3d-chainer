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
$ wget http://vlg.cs.dartmouth.edu/c3d/conv3d_deepnetA_sport1m_iter_1900000
$ wget https://raw.githubusercontent.com/gtoderici/sports-1m-dataset/master/labels.txt
$ wget https://raw.githubusercontent.com/facebook/C3D/master/C3D-v1.0/src/caffe/proto/caffe.proto
$ youtube-dl -f mp4 https://www.youtube.com/watch?v=dM06AMFLsrc -o dM06AMFLsrc.mp4
```

# Compiling caffe.proto

```
$ sed -i '1s/^/syntax = "proto2";\n/' caffe.proto # inserting syntax version on the head.
$ protoc --python_out=. caffe.proto
```