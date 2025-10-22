import torch
from pytorch_msssim import SSIM
from typing import Callable
from kornia.color import rgb_to_lab as kornia_rgb_to_lab


def get_ssim_module() -> SSIM:
    """
    Creates and returns an SSIM module configured for the project's needs.

    Returns:
        SSIM: An instance of the SSIM class.
    """
    return SSIM(data_range=255.0, size_average=False, channel=3, nonnegative_ssim=True)


def compare_ssim(
    tensor1: torch.Tensor, tensor2: torch.Tensor, ssim_module: SSIM
) -> torch.Tensor:
    """
    Computes the SSIM score between two batches of image tensors.

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors.
        tensor2 (torch.Tensor): The second batch of image tensors.
        ssim_module (SSIM): The SSIM module to use for computation.

    Returns:
        torch.Tensor: A tensor containing the SSIM scores for each pair.
    """
    return ssim_module(tensor1, tensor2)


def rgb_to_lab(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of RGB image tensors to the CIELAB color space using kornia.
    Args:
        rgb_tensor (torch.Tensor): Batch of RGB tensors (B, C, H, W) with values in [0, 255].
    Returns:
        torch.Tensor: Batch of CIELAB tensors.
    """
    # kornia expects RGB values in the range [0, 1]
    rgb_normalized = rgb_tensor / 255.0
    return kornia_rgb_to_lab(rgb_normalized)


def compare_cielab(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    lab1 = rgb_to_lab(tensor1)
    lab2 = rgb_to_lab(tensor2)
    delta_e = torch.sqrt(torch.sum((lab1 - lab2) ** 2, dim=1))
    return torch.mean(delta_e, dim=[1, 2])


def compare_mean_color_distance(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Euclidean distance between the mean colors of two batches of image tensors.

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors (B, C, H, W).
        tensor2 (torch.Tensor): The second batch of image tensors (B, C, H, W).

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the mean color distance for each pair.
    """
    mean1 = torch.mean(tensor1, dim=[2, 3])
    mean2 = torch.mean(tensor2, dim=[2, 3])
    distance = torch.sqrt(torch.sum((mean1 - mean2) ** 2, dim=1))
    return distance

def get_diff_mask(
    scores: torch.Tensor, threshold: float, method: str
) -> torch.Tensor:
    """
    Determines which pairs are different based on scores and a threshold.

    Args:
        scores (torch.Tensor): The comparison scores.
        threshold (float): The threshold for difference.
        method (str): The comparison method used.

    Returns:
        torch.Tensor: A boolean tensor indicating different pairs.
    """
    if method in ["ssim"]:
        return scores < threshold
    if method in ["cielab", "mean_color"]:
        return scores > threshold
    raise ValueError(f"Unsupported method: {method}")


def get_comparison_function(
    method: str,
) -> Callable:
    """
    Returns the appropriate comparison function based on the method name.

    Args:
        method (str): The name of the comparison method.

    Returns:
        Callable: The comparison function.
    """
    if method == "ssim":
        return compare_ssim
    if method == "cielab":
        return compare_cielab
    if method == "mean_color":
        return compare_mean_color_distance
    raise ValueError(f"Unsupported method: {method}")
