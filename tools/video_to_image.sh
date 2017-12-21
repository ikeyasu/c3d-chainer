#!/bin/bash
FILE=$1
mkdir -p ${FILE%.*}
ffmpeg -i "$FILE" -vf fps=5 "${FILE%.*}"/%05d.jpg
