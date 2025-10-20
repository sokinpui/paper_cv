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
        description="Compare 512x512 units of an image."
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device to use for computation ('gpu' or 'cpu'). Default is 'gpu'.",
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

    # MSE method
    parser_mse = subparsers.add_parser("mse", help="Mean Squared Error (MSE) comparison.")
    parser_mse.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=100.0,
        help="MSE threshold. Pairs with a score ABOVE this are different. Default: 100.0.",
    )

    # Color Histogram method
    parser_hist = subparsers.add_parser(
        "color_histogram", help="Color Histogram Intersection comparison."
    )
    parser_hist.add_argument(
        "-t",
        "--threshold",
        type=threshold_0_to_1,
        default=0.9,
        help="Histogram intersection threshold (0.0 to 1.0). Pairs with a score BELOW this are considered different. Default: 0.9.",
    )

    # CIELAB method
    parser_cielab = subparsers.add_parser(
        "cielab", help="CIELAB Delta E comparison."
    )
    parser_cielab.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=2.3,
        help="CIELAB Delta E threshold. A common value is 2.3 (Just Noticeable Difference). Pairs with a score ABOVE this are different. Default: 2.3.",
    )

    # PAD method
    parser_pad = subparsers.add_parser(
        "pad", help="Peak Absolute Difference (PAD) comparison."
    )
    parser_pad.add_argument(
        "-t",
        "--threshold",
        type=threshold_0_to_255,
        default=20.0,
        help="PAD threshold (0-255). Pairs with a score ABOVE this are different. Default: 20.0.",
    )

    args = parser.parse_args()

    if args.device == "gpu":
        find_different_units_gpu(args.image, args.threshold, method=args.method)
    else:
        find_different_units_cpu(args.image, args.threshold, method=args.method)


if __name__ == "__main__":
    main()
