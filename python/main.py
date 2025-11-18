import argparse

from core import find_different_units_cpu, find_different_units_gpu


def threshold_0_to_1(x: str) -> float:
    """Argument type checker for a float between 0 and 1."""
    try:
        x_float = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} is not a floating-point literal")

    if not (0.0 <= x_float <= 1.0):
        raise argparse.ArgumentTypeError(f"{x_float} is not in range [0.0, 1.0]")
    return x_float


def threshold_0_to_255(x: str) -> float:
    """Argument type checker for a float between 0 and 255."""
    try:
        x_float = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} is not a floating-point literal")

    if not (0.0 <= x_float <= 255.0):
        raise argparse.ArgumentTypeError(f"{x_float} is not in range [0.0, 255.0]")
    return x_float


def main() -> None:
    """
    Main function to parse arguments and run the image comparison.
    """
    parser = argparse.ArgumentParser(
        description="Compare units of an image."
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--unit-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("WIDTH", "HEIGHT"),
        help="The size of the image units to compare (width height). Default: 512 512.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Directory to save different image unit pairs. If not provided, no pairs will be saved.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device to use for computation ('gpu' or 'cpu'). Default is 'gpu'.",
    )
    parser.add_argument(
        "--analyze-units",
        action="store_true",
        help="Perform and display a statistical analysis of each unit against all others.",
    )

    # Create subparsers for each method
    subparsers = parser.add_subparsers(
        dest="method",
        required=True,
        title="Comparison Methods",
        help="Choose a comparison method. Use 'python main.py <method> --help' for method-specific options.",
    )

    # SSIM method
    parser_ssim = subparsers.add_parser(
        "ssim", help="Structural Similarity Index (SSIM) comparison."
    )
    parser_ssim.add_argument(
        "-t",
        "--threshold",
        type=threshold_0_to_1,
        default=0.9,
        help="SSIM threshold (0.0 to 1.0). Pairs with a score BELOW this are considered different. Default: 0.9.",
    )

    # CIELAB method
    parser_cielab = subparsers.add_parser(
        "cielab", help="Perceptually Uniform Color Spaces (CIELAB) comparison."
    )
    parser_cielab.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=10.0,
        help="CIELAB distance threshold (Delta E*). Pairs with a score ABOVE this are different. Default: 10.0.",
    )

    # Color Histogram method
    parser_color_histogram = subparsers.add_parser(
        "color_histogram", help="Color Histogram comparison using OpenCV."
    )
    parser_color_histogram.add_argument(
        "-t",
        "--threshold",
        type=threshold_0_to_1,
        default=0.9,
        help="Histogram correlation threshold (0.0 to 1.0). Pairs with a score BELOW this are considered different. Default: 0.9.",
    )

    # Color Clustering method
    parser_clustering = subparsers.add_parser(
        "color_clustering", help="Color Clustering (K-means) comparison."
    )
    parser_clustering.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=20.0,
        help="Palette distance threshold. Pairs with a score ABOVE this are different. Default: 20.0.",
    )
    parser_clustering.add_argument(
        "-k",
        "--num-clusters",
        type=int,
        default=5,
        help="Number of dominant colors (clusters) to find. Default: 5.",
    )

    # Color Range HSV method
    parser_color_range_hsv = subparsers.add_parser(
        "color_range_hsv", help="Color Range Detection in HSV color space using OpenCV."
    )
    parser_color_range_hsv.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.2, # A lower L1 distance means more similar, so we look for scores ABOVE this.
        help="Color range distance threshold. Pairs with a score ABOVE this are considered different. Default: 0.2.",
    )
    args = parser.parse_args()
    unit_size = (args.unit_size[0], args.unit_size[1])

    method_params = {}
    if args.method == "color_clustering":
        method_params["k"] = args.num_clusters

    if args.device == "gpu":
        find_different_units_gpu(
            args.image,
            args.threshold,
            unit_size=unit_size,
            method=args.method,
            method_params=method_params,
            output_dir=args.output,
            analyze_units=args.analyze_units,
        )
    else:
        find_different_units_cpu(
            args.image,
            args.threshold,
            unit_size=unit_size,
            method=args.method,
            method_params=method_params,
            output_dir=args.output,
            analyze_units=args.analyze_units,
        )


if __name__ == "__main__":
    main()
