Creating images from UCF11 videos.
--------------------------------

Download UCF11 videos.
(Please refer to [here](http://crcv.ucf.edu/data/UCF_YouTube_Action.php) for details.)

```
$ wget http://crcv.ucf.edu/data/UCF11_updated_mpg.rar
$ mkdir videos
$ pushd videos
$   unar e UCF11_updated_mpg.rar
$ popd
```

Then, convert videos to images.

```
$ ls videos/*.mpg | parallel --no-notice -j8 ./tools/video_to_images.sh  {}
```

Make folders for resized and cropped images.

```
$ ls videos/*/*.jpg | parallel --no-notice -j 200 'echo `dirname {} | sed s/videos/images/`' | uniq | tee dirs
$ cat dirs | xargs mkdir -p
$ rm dirs
```

Resize and crop.

```
$ ls videos/*/*.jpg | parallel -j20 'python tools/resize.py -i {} -o `echo {} | sed s/videos/images/`'
```

Some videos have very small number of frames. These videos cannot use for training/testing.
The following command conts number of files for each video.
Please find videos which have images less than 10, and remove the folders.

```
$ cd images/
$ ls | parallel -j50 'echo `ls -1 {} | wc -l` {}' | sort -n > counts
```

To separate test set, move dirs randomly.

```
$ pushd images
$   ls | shuf | head -n 300 | xargs -I '{}' mv {} ../tests/
$ popd
```
compute mean value of images.

```
$ find images/ | grep .jpg$ > list
$ python tools/compute_mean.py --root . list
$ ls mean.npy
```

If you want to resize,

```
$ find ucf11_240px/ | grep \.jpg$ | parallel -j20 'convert -resize 112X112 {} `echo {} | sed s/240px/112px/`'
```
