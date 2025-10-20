import sys
from typing import List

import numpy as np
from PIL import Image


def pad_image_to_block_size(image: Image.Image, block_size: int) -> Image.Image:
    """
    Pads an image with black pixels so that its dimensions are perfectly
    divisible by the block size.
    """
    width, height = image.size
    pad_width = (block_size - width % block_size) % block_size
    pad_height = (block_size - height % block_size) % block_size

    if pad_width == 0 and pad_height == 0:
        return image

    new_width = width + pad_width
    new_height = height + pad_height

    padded_image = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    padded_image.paste(image, (0, 0))
    return padded_image


def split_image(image_path: str, block_size: int) -> List[np.ndarray]:
    """
    Loads an image from the specified path, converts it to RGB, pads it,
    and splits it into a list of blocks of a given size.

    Returns:
        A list of NumPy arrays, where each array represents a block.
        Returns an empty list if an error occurs.
    """
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            padded_img = pad_image_to_block_size(img_rgb, block_size)

            blocks = []
            width, height = padded_img.size
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    box = (x, y, x + block_size, y + block_size)
                    block = padded_img.crop(box)
                    blocks.append(np.array(block))
            return blocks
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An error occurred while processing the image: {e}", file=sys.stderr)
        return []
