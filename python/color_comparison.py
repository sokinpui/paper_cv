import numpy as np
from skimage import color

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    cupy = None
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def is_cupy_available() -> bool:
    """Checks if the CuPy library is installed and available."""
    return CUPY_AVAILABLE


def is_mps_available() -> bool:
    """Checks if PyTorch is installed and MPS is available on Apple Silicon."""
    if not TORCH_AVAILABLE or not torch:
        return False
    return torch.backends.mps.is_available()


def _calculate_average_color_cpu(block: np.ndarray) -> np.ndarray:
    """Calculates the average color of a block using NumPy on the CPU."""
    return np.mean(block, axis=(0, 1))


def _calculate_average_color_gpu(block: np.ndarray) -> np.ndarray:
    """
    Calculates the average color of a block using CuPy on the GPU.
    """
    if not cupy:
        raise RuntimeError("CuPy is not installed or failed to import.")

    block_gpu = cupy.asarray(block)
    avg_color_gpu = cupy.mean(block_gpu, axis=(0, 1))
    return cupy.asnumpy(avg_color_gpu)


def _calculate_average_color_mps(block: np.ndarray) -> np.ndarray:
    """
    Calculates the average color of a block using PyTorch on Apple Silicon GPU (MPS).
    """
    if not torch:
        raise RuntimeError("PyTorch is not installed or failed to import.")

    device = torch.device("mps")
    block_tensor = torch.from_numpy(block).to(device, dtype=torch.float32)
    avg_color_tensor = torch.mean(block_tensor, dim=(0, 1))
    return avg_color_tensor.cpu().numpy()


def calculate_average_color(block: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Calculates the average RGB color of an image block, dispatching to
    the specified device implementation.
    """
    if device == "cuda":
        return _calculate_average_color_gpu(block)
    if device == "mps":
        return _calculate_average_color_mps(block)
    if device == "cpu":
        return _calculate_average_color_cpu(block)
    raise ValueError(f"Unsupported device: {device}")


def calculate_delta_e(avg_color1: np.ndarray, avg_color2: np.ndarray) -> float:
    """
    Calculates the perceptual color difference between two RGB colors using
    the CIELAB Delta E 76 formula.
    """
    # Reshape and normalize from [0, 255] to [0, 1] for scikit-image
    avg_color1_rgb_norm = avg_color1.astype("float64").reshape(1, 1, 3) / 255.0
    avg_color2_rgb_norm = avg_color2.astype("float64").reshape(1, 1, 3) / 255.0

    avg_color1_lab = color.rgb2lab(avg_color1_rgb_norm)
    avg_color2_lab = color.rgb2lab(avg_color2_rgb_norm)

    # Calculate Delta E and extract the single float value
    delta_e = color.deltaE_cie76(avg_color1_lab, avg_color2_lab)
    return float(delta_e[0, 0])
