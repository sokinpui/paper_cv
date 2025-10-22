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


def compare_cielab(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Compares two batches of image tensors based on the average color difference
    in the CIELAB color space (Delta E*).

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors (B, C, H, W).
        tensor2 (torch.Tensor): The second batch of image tensors (B, C, H, W).

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the average Delta E*
                      distance for each pair. Higher score means more different.
    """
    lab1 = rgb_to_lab(tensor1)
    lab2 = rgb_to_lab(tensor2)

    mean_lab1 = torch.mean(lab1, dim=[2, 3])
    mean_lab2 = torch.mean(lab2, dim=[2, 3])

    delta_e = torch.linalg.norm(mean_lab1 - mean_lab2, dim=1)
    return delta_e


def compare_color_histogram(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> torch.Tensor:
    """
    Compares the color histogram correlation between two batches of image tensors using OpenCV.

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
    # Downscale the image to a smaller size to speed up k-means
    small_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = small_image.reshape(-1, 3).astype(np.float32)
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
    tensor1: torch.Tensor, tensor2: torch.Tensor, k: int
) -> torch.Tensor:
    """
    Compares two batches of image tensors based on the distribution of pixels
    within predefined HSV color ranges.

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors (B, C, H, W).
        tensor2 (torch.Tensor): The second batch of image tensors (B, C, H, W).
        k (int): The number of dominant colors (clusters) to find.

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the palette distance for each pair.
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


HSV_COLOR_RANGES_DICT = {
    "red": [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [179, 255, 255])],
    "orange": [([10, 100, 100], [20, 255, 255])],
    "yellow": [([20, 100, 100], [35, 255, 255])],
    "green": [([40, 100, 100], [80, 255, 255])],
    "cyan": [([80, 100, 100], [100, 255, 255])],
    "blue": [([100, 100, 100], [140, 255, 255])],
    "purple": [([140, 50, 100], [160, 255, 255])],  # A bit more relaxed S for purple
    "magenta": [([160, 100, 100], [170, 255, 255])],
    "white": [([0, 0, 200], [179, 30, 255])],  # Low S, High V
    "black": [([0, 0, 0], [179, 255, 50])],  # Low V
    "gray": [([0, 0, 50], [179, 50, 200])],  # Low S, Mid V
}


def _get_color_percentages(hsv_image: np.ndarray, total_pixels: int) -> np.ndarray:
    """
    Calculates the percentage of pixels for each predefined HSV color range in an image.

    Args:
        hsv_image (np.ndarray): The HSV image (H, W, C).
        total_pixels (int): The total number of pixels in the image.

    Returns:
        np.ndarray: A numpy array where each element is the percentage of pixels
                    falling into a specific color range.
    """
    percentages = []
    for color_name, ranges in HSV_COLOR_RANGES_DICT.items():
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        count = cv2.countNonZero(combined_mask)
        percentages.append(count / total_pixels)
    return np.array(percentages, dtype=np.float32)


def compare_color_range_hsv(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> torch.Tensor:
    """
    Compares two batches of image tensors based on the distribution of pixels
    within predefined HSV color ranges.

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors (B, C, H, W).
        tensor2 (torch.Tensor): The second batch of image tensors (B, C, H, W).

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the L1 distance
                      between color range percentage vectors for each pair.
                      Lower score means more similar.
    """
    # Convert tensors from (B, C, H, W) to (B, H, W, C) and then to numpy uint8 for OpenCV
    imgs1_rgb = tensor1.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    imgs2_rgb = tensor2.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    scores = []
    for i in range(imgs1_rgb.shape[0]):
        img1_rgb = imgs1_rgb[i]
        img2_rgb = imgs2_rgb[i]

        # Convert RGB to HSV
        hsv1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2HSV)

        total_pixels = hsv1.shape[0] * hsv1.shape[1]

        # Get color percentages for both images
        percentages1 = _get_color_percentages(hsv1, total_pixels)
        percentages2 = _get_color_percentages(hsv2, total_pixels)

        # Calculate L1 distance between the percentage vectors
        distance = np.sum(np.abs(percentages1 - percentages2))
        scores.append(distance)

    # Return scores as a torch tensor on the same device as input tensors
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
    if method in ["cielab", "mean_color", "color_clustering", "color_range_hsv"]:
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
    if method == "cielab":
        return compare_cielab
    if method == "color_histogram":
        return compare_color_histogram
    if method == "color_clustering":
        k = kwargs.get("k", 5)
        return partial(compare_color_clustering, k=k)
    if method == "color_range_hsv":
        return compare_color_range_hsv
    raise ValueError(f"Unsupported method: {method}")
