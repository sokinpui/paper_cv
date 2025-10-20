# Image Block Color Comparator

This program analyzes a BMP image by dividing it into blocks of a specified size (e.g., 512x512) and then calculates the color difference between every possible pair of blocks. The color difference is measured using the CIELAB Delta E 76 formula.

The program is designed to be efficient, utilizing multiprocessing to parallelize the comparison tasks across multiple CPU cores. It also features optional GPU acceleration via CuPy for NVIDIA GPUs, which can significantly speed up the process of calculating the average color of each block.

## Features

-   **Image Splitting**: Divides any BMP image into N x N blocks.
-   **Color Comparison**: Uses the perceptually uniform CIELAB color space to compare the average color of blocks.
-   **Parallel Processing**: Leverages the `multiprocessing` module to run comparisons in parallel.
-   **GPU Acceleration**: Optionally uses `cupy` to accelerate calculations on NVIDIA GPUs.
-   **POSIX-compliant CLI**: A simple and standard command-line interface powered by `argparse`.

## Setup

1.  **Create the project directory and files:**
    Place the provided Python files (`main.py`, `image_processing.py`, `color_comparison.py`, `requirements.txt`) inside a directory named `image_comparator`.

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    The required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **GPU Support (Optional):**
    To use GPU acceleration, you must have an NVIDIA GPU with the appropriate CUDA toolkit installed. Then, install the `cupy` version that matches your CUDA toolkit. For example, for CUDA 11.x:
    ```bash
    pip install cupy-cuda11x
    ```
    For other versions, please refer to the official [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Usage

The script is run from the command line from within the `image_comparator` directory.

### Syntax
```bash
python main.py [image_path] [options]
```

### Arguments

-   `image_path`: (Required) The path to the input BMP image file.

### Options

-   `-s, --block-size <int>`: The height and width of the blocks to divide the image into. (Default: 512)
-   `-p, --processes <int>`: The number of worker processes to use. (Default: number of CPU cores)
-   `--gpu`: Enable GPU acceleration. Requires a compatible NVIDIA GPU and `cupy`.
-   `-h, --help`: Show the help message.

### Example

```bash
# Run with default settings on all available CPU cores
python main.py path/to/my_image.bmp

# Run using 16 processes and a block size of 256
python main.py path/to/my_image.bmp -p 16 -s 256

# Run with GPU acceleration
python main.py path/to/my_image.bmp --gpu
```
