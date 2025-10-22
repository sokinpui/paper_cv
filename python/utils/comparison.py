from functools import partial
from typing import Callable

import cv2
import numpy as np
import torch
from kornia.color import rgb_to_lab as kornia_rgb_to_lab
from pytorch_msssim import SSIM


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


def compare_color_histogram(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> torch.Tensor:
    """
    Computes the color histogram correlation between two batches of image tensors using OpenCV.

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors (B, C, H, W).
        tensor2 (torch.Tensor): The second batch of image tensors (B, C, H, W).

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the histogram correlation score for each pair.
    """
    imgs1 = tensor1.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    imgs2 = tensor2.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    scores = []
    for i in range(imgs1.shape[0]):
        img1 = imgs1[i]
        img2 = imgs2[i]

        hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        scores.append(score)

    return torch.tensor(scores, dtype=torch.float32, device=tensor1.device)


def _get_dominant_colors(image: np.ndarray, k: int) -> np.ndarray:
    """Helper to get dominant colors from a single image using K-means."""
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers


def _palette_distance(centers1_rgb: np.ndarray, centers2_rgb: np.ndarray) -> float:
    """Helper to compute the distance between two color palettes in LAB space."""
    k = centers1_rgb.shape[0]
    # cv2.cvtColor expects a 3D array (1, k, 3)
    centers1_lab = cv2.cvtColor(
        centers1_rgb.reshape(1, k, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
    ).reshape(k, 3)
    centers2_lab = cv2.cvtColor(
        centers2_rgb.reshape(1, k, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
    ).reshape(k, 3)

    dist_matrix = np.linalg.norm(
        centers1_lab[:, np.newaxis, :] - centers2_lab[np.newaxis, :, :], axis=2
    )

    min_dists_1 = np.min(dist_matrix, axis=1)
    min_dists_2 = np.min(dist_matrix, axis=0)

    score = (np.mean(min_dists_1) + np.mean(min_dists_2)) / 2.0
    return score


def compare_color_clustering(
    tensor1: torch.Tensor, tensor2: torch.Tensor, k: int = 5
) -> torch.Tensor:
    """
    Computes color similarity based on K-means clustering of color palettes.

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors (B, C, H, W).
        tensor2 (torch.Tensor): The second batch of image tensors (B, C, H, W).
        k (int): The number of dominant colors (clusters) to find.

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the palette distance score for each pair.
    """
    imgs1 = tensor1.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    imgs2 = tensor2.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    scores = []
    for i in range(imgs1.shape[0]):
        img1 = imgs1[i]
        img2 = imgs2[i]

        centers1 = _get_dominant_colors(img1, k)
        centers2 = _get_dominant_colors(img2, k)

        score = _palette_distance(centers1, centers2)
        scores.append(score)

    return torch.tensor(scores, dtype=torch.float32, device=tensor1.device)


def get_diff_mask(scores: torch.Tensor, threshold: float, method: str) -> torch.Tensor:
    """
    Determines which pairs are different based on scores and a threshold.

    Args:
        scores (torch.Tensor): The comparison scores.
        threshold (float): The threshold for difference.
        method (str): The comparison method used.

    Returns:
        torch.Tensor: A boolean tensor indicating different pairs.
    """
    if method in ["ssim", "color_histogram"]:
        return scores < threshold
    if method in ["cielab", "mean_color", "color_clustering"]:
        return scores > threshold
    raise ValueError(f"Unsupported method: {method}")


def get_comparison_function(method: str, **kwargs) -> Callable:
    """
    Returns the appropriate comparison function based on the method name.

    Args:
        method (str): The name of the comparison method.
        **kwargs: Additional arguments for specific methods (e.g., 'k' for color_clustering).

    Returns:
        Callable: The comparison function.
    """
    if method == "ssim":
        return compare_ssim
    if method == "mean_color":
        return compare_mean_color_distance
    if method == "color_histogram":
        return compare_color_histogram
    if method == "color_clustering":
        k = kwargs.get("k", 5)
        return partial(compare_color_clustering, k=k)
    raise ValueError(f"Unsupported method: {method}")
