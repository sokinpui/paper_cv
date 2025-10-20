import argparse
import itertools
import multiprocessing
import sys
from functools import partial
from typing import Tuple

from tqdm import tqdm

from color_comparison import (calculate_average_color, calculate_delta_e,
                              is_cupy_available, is_mps_available)
from image_processing import split_image

# A global list to hold image blocks in memory.
# This leverages copy-on-write memory on POSIX systems (Linux, macOS) for
# efficient data sharing with child processes. For Windows, this approach is
# less efficient as data is pickled and sent to each child process.
BLOCKS_DATA = []


def process_pair(indices: Tuple[int, int], device: str) -> Tuple[int, int, float]:
    """
    A worker function designed to be executed in a separate process. It compares
    a single pair of image blocks and returns their color difference.

    Args:
        indices: A tuple containing the indices (i, j) of the blocks to compare
                 from the global BLOCKS_DATA list.
        device: The device to use for calculations ('cpu', 'cuda', 'mps').

    Returns:
        A tuple containing the original indices and the calculated Delta E value.
    """
    i, j = indices
    block1 = BLOCKS_DATA[i]
    block2 = BLOCKS_DATA[j]

    avg_color1 = calculate_average_color(block1, device=device)
    avg_color2 = calculate_average_color(block2, device=device)

    delta_e = calculate_delta_e(avg_color1, avg_color2)

    return i, j, delta_e


def create_argument_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Divide an image into blocks and compare all pairs for color difference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_path", help="Path to the input BMP image.")
    parser.add_argument(
        "-s", "--block-size", type=int, default=512,
        help="The height and width of the blocks to divide the image into."
    )
    parser.add_argument(
        "-p", "--processes", type=int, default=multiprocessing.cpu_count(),
        help="Number of worker processes to use."
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu", action="store_true",
        help="Enable GPU acceleration with CuPy. Requires a compatible NVIDIA GPU and CUDA."
    )
    gpu_group.add_argument(
        "--mps", action="store_true",
        help="Enable GPU acceleration on Apple Silicon (MPS)."
    )
    return parser


def main():
    """The main entry point of the application."""
    parser = create_argument_parser()
    args = parser.parse_args()

    device = "cpu"
    if args.gpu:
        if not is_cupy_available():
            print("Error: GPU acceleration requested, but CuPy is not available.", file=sys.stderr)
            sys.exit(1)
        device = "cuda"
        print("Using CUDA for GPU acceleration.")
    elif args.mps:
        if not is_mps_available():
            print("Error: MPS acceleration requested, but PyTorch with MPS support is not available.", file=sys.stderr)
            sys.exit(1)
        device = "mps"
        print("Using MPS for GPU acceleration on Apple Silicon.")

    print(f"Loading and splitting image: {args.image_path}")
    blocks = split_image(args.image_path, args.block_size)

    if not blocks:
        print("Error: No blocks were created from the image. Exiting.", file=sys.stderr)
        sys.exit(1)

    num_blocks = len(blocks)
    print(f"Image split into {num_blocks} blocks.")

    global BLOCKS_DATA
    BLOCKS_DATA = blocks

    pairs = list(itertools.combinations(range(num_blocks), 2))

    if not pairs:
        print("Not enough blocks to form a pair for comparison. Exiting.")
        return

    print(f"Comparing {len(pairs)} pairs using {args.processes} processes...")

    worker_func = partial(process_pair, device=device)

    with multiprocessing.Pool(processes=args.processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, pairs), total=len(pairs)))

    results.sort(key=lambda x: x[2])

    print("\nTop 5 most similar pairs (lowest color difference):")
    for i, j, diff in results[:5]:
        print(f"Blocks ({i}, {j}): Delta E = {diff:.2f}")

    print("\nTop 5 most different pairs (highest color difference):")
    for i, j, diff in reversed(results[-5:]):
        print(f"Blocks ({i}, {j}): Delta E = {diff:.2f}")


if __name__ == "__main__":
    main()
