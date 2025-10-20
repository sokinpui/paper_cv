from typing import Tuple, List
from PIL import Image
import numpy as np

ImageUnit = Tuple[Tuple[int, int], np.ndarray]

def divide_image_into_units(
    image_path: str, unit_size: Tuple[int, int] = (512, 512)
) -> List[ImageUnit]:
    """
    Divides an image into units of a specified size.

    Args:
        image_path (str): The path to the input image.
        unit_size (tuple): The (width, height) of the units.

    Returns:
        list: A list of tuples, where each tuple contains the coordinates
              (row, col) and the image unit as a numpy array.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return []

    image = image.convert("RGB")
    image_np = np.array(image)
    width, height = image.size
    unit_width, unit_height = unit_size
    image_units = []

    for i in range(height // unit_height):
        for j in range(width // unit_width):
            top = i * unit_height
            left = j * unit_width
            unit = image_np[top : top + unit_height, left : left + unit_width]
            image_units.append(((i, j), unit))

    return image_units
