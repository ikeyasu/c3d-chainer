from PIL import Image
import numpy as np
import argparse


def main(path, out_path, crop=32):
  img = Image.open(path, 'r')
  w, h = img.size
  cx = int((w - crop) / 2)
  cy = int((h - crop) / 2)
  clp = img.crop((cx, cy, w - cx, h - cy))
  resized = clp.resize((crop, crop))
  resized.save(out_path, 'JPEG', quality=100, optimize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='clipping and resize')
    parser.add_argument('--input', '-i', default='input.jpg',
                        help='input file')
    parser.add_argument('--output', '-o', default='output.jpg',
                        help='output file')
    parser.add_argument('--crop', '-c', type=int, default=32,
                        help='crop size')
    args = parser.parse_args()
    main(args.input, args.output, args.crop)
