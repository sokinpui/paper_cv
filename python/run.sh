#!/bin/bash

for method in ssim cielab color_histogram color_clustering color_range_hsv; do
  echo Method: $method, running on GPU
  python main.py -d gpu --unit-size 512 512 ../test_image/pure_black.bmp $method
  echo
done

for method in ssim cielab color_histogram color_clustering color_range_hsv; do
  echo Method: $method, running on CPU
  python main.py -d cpu --unit-size 512 512 ../test_image/pure_black.bmp $method
  echo
done
