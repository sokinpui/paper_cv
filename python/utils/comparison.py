import torch
from pytorch_msssim import SSIM
from typing import Callable


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


def compare_mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Computes the MSE score between two batches of image tensors.

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors.
        tensor2 (torch.Tensor): The second batch of image tensors.

    Returns:
        torch.Tensor: A tensor containing the MSE scores for each pair.
    """
    return torch.mean((tensor1 - tensor2) ** 2, dim=[1, 2, 3])


def compare_color_histogram(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> torch.Tensor:
    """
    Computes the color histogram intersection score between two batches of image tensors.

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors.
        tensor2 (torch.Tensor): The second batch of image tensors.

    Returns:
        torch.Tensor: A tensor containing the histogram intersection scores for each pair.
    """
    B, C, H, W = tensor1.shape
    bins = 256
    combined_tensor = torch.cat([tensor1, tensor2], dim=0)
    combined_tensor_int = combined_tensor.long()
    offsets = torch.arange(2 * B, device=tensor1.device) * bins
    hists = []
    for c_idx in range(C):
        channel_data = combined_tensor_int[:, c_idx, :, :].flatten(1)
        channel_data_offset = channel_data + offsets[:, None]
        hist_c = torch.bincount(
            channel_data_offset.flatten(), minlength=2 * B * bins
        ).view(2 * B, bins)
        hists.append(hist_c)
    batch_hists = torch.cat(hists, dim=1)
    batch_hists = batch_hists.float()
    # Add a small epsilon to avoid division by zero for blank images
    batch_hists_sum = torch.sum(batch_hists, dim=1, keepdim=True)
    batch_hists = batch_hists / (batch_hists_sum + 1e-6)

    hist1, hist2 = torch.chunk(batch_hists, 2, dim=0)
    return torch.sum(torch.sqrt(hist1 * hist2), dim=1)


def rgb_to_lab(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of RGB image tensors to the CIELAB color space.
    Args:
        rgb_tensor (torch.Tensor): Batch of RGB tensors (B, C, H, W) with values in [0, 255].
    Returns:
        torch.Tensor: Batch of CIELAB tensors.
    """
    # Normalize to [0, 1]
    rgb_linear = rgb_tensor / 255.0

    # sRGB to linear RGB
    mask = (rgb_linear > 0.04045).float()
    rgb_linear = (((rgb_linear + 0.055) / 1.055) ** 2.4) * mask + (
        rgb_linear / 12.92
    ) * (1 - mask)

    # RGB to XYZ
    # Transformation matrix from sRGB to XYZ (D65 illuminant)
    xyz_from_rgb = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=rgb_tensor.dtype,
        device=rgb_tensor.device,
    )

    B, C, H, W = rgb_linear.shape
    rgb_linear_reshaped = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)
    xyz = (
        torch.matmul(rgb_linear_reshaped, xyz_from_rgb.t())
        .reshape(B, H, W, 3)
        .permute(0, 3, 1, 2)
    )

    # XYZ to CIELAB
    # D65 white point reference
    xn, yn, zn = 95.047, 100.0, 108.883
    xyz_ref = (
        torch.tensor([xn, yn, zn], dtype=rgb_tensor.dtype, device=rgb_tensor.device)
        .view(1, 3, 1, 1)
    )
    xyz_normalized = xyz / xyz_ref

    # f(t) function
    mask = (xyz_normalized > 0.008856).float()  # (6/29)**3
    f_xyz = (xyz_normalized ** (1 / 3.0)) * mask + (
        7.787 * xyz_normalized + 16 / 116.0
    ) * (1 - mask)

    fx, fy, fz = f_xyz[:, 0, :, :], f_xyz[:, 1, :, :], f_xyz[:, 2, :, :]

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fz - fy)

    return torch.stack([L, a, b], dim=1)


def compare_cielab(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    lab1 = rgb_to_lab(tensor1)
    lab2 = rgb_to_lab(tensor2)
    delta_e = torch.sqrt(torch.sum((lab1 - lab2) ** 2, dim=1))
    return torch.mean(delta_e, dim=[1, 2])


def compare_pad(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Peak Absolute Difference (PAD) between two batches of image tensors.
    This finds the maximum absolute difference for any pixel channel across all pixels.

    Args:
        tensor1 (torch.Tensor): The first batch of image tensors (B, C, H, W).
        tensor2 (torch.Tensor): The second batch of image tensors (B, C, H, W).

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the PAD scores for each pair.
    """
    abs_diff = torch.abs(tensor1 - tensor2)
    # Flatten C, H, W dimensions to find the max over all pixels and channels for each item in the batch
    flattened_diff = abs_diff.reshape(tensor1.shape[0], -1)
    max_diff, _ = torch.max(flattened_diff, dim=1)
    return max_diff


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
    if method in ["ssim", "color_histogram"]:
        return scores < threshold
    if method in ["mse", "cielab", "pad", "mean_color"]:
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
    if method == "mse":
        return compare_mse
    if method == "color_histogram":
        return compare_color_histogram
    if method == "cielab":
        return compare_cielab
    if method == "pad":
        return compare_pad
    if method == "mean_color":
        return compare_mean_color_distance
    raise ValueError(f"Unsupported method: {method}")
