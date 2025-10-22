import concurrent.futures
import itertools
import multiprocessing
import os
import shutil
import time
from functools import partial
from typing import List, Tuple

import numba
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.comparison import get_comparison_function, get_diff_mask, get_ssim_module
from utils.image_processor import divide_image_into_units


@numba.jit(nopython=True)
def _collect_diff_pairs_from_batch_np(
    batch_indices_np, positions_np, scores_np, diff_indices_np
):
    """
    Collects different pairs from a batch of comparisons using Numba for acceleration.
    This function is designed to be JIT-compiled by Numba in nopython mode.

    Args:
        batch_indices_np (np.ndarray): Numpy array of pair indices for the batch.
        positions_np (np.ndarray): Numpy array of all unit positions.
        scores_np (np.ndarray): Numpy array of scores for the batch.
        diff_indices_np (np.ndarray): Numpy array of indices within the
                                                 batch that are considered different.

    Returns:
        np.ndarray: A numpy array where each row represents a different pair:
                    [idx1, idx2, pos1_row, pos1_col, pos2_row, pos2_col, score]
    """
    num_found = len(diff_indices_np)
    # (idx1, idx2, pos1_row, pos1_col, pos2_row, pos2_col, score)
    results = np.empty((num_found, 7), dtype=np.float64)

    for i in range(num_found):
        idx_in_batch = diff_indices_np[i]

        pair_indices = batch_indices_np[idx_in_batch]
        idx1 = pair_indices[0]
        idx2 = pair_indices[1]

        pos1 = positions_np[idx1]
        pos2 = positions_np[idx2]

        score = scores_np[idx_in_batch]

        results[i, 0] = idx1
        results[i, 1] = idx2
        results[i, 2] = pos1[0]
        results[i, 3] = pos1[1]
        results[i, 4] = pos2[0]
        results[i, 5] = pos2[1]
        results[i, 6] = score

    return results


def _collect_diff_pairs_from_batch_gpu(
    batch_indices_tensor: torch.Tensor,
    positions_tensor: torch.Tensor,
    scores_tensor: torch.Tensor,
    diff_indices_in_batch_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Collects different pairs from a batch of comparisons using PyTorch on GPU.

    Args:
        batch_indices_tensor (torch.Tensor): Tensor of pair indices for the batch (on GPU).
        positions_tensor (torch.Tensor): Tensor of all unit positions (on GPU).
        scores_tensor (torch.Tensor): Tensor of scores for the batch (on GPU).
        diff_indices_in_batch_tensor (torch.Tensor): Tensor of indices within the
                                                     batch that are considered different (on GPU).

    Returns:
        torch.Tensor: A tensor where each row represents a different pair:
                      [idx1, idx2, pos1_row, pos1_col, pos2_row, pos2_col, score] (on GPU)
    """
    if len(diff_indices_in_batch_tensor) == 0:
        return torch.empty((0, 7), dtype=torch.float32, device=batch_indices_tensor.device)

    diff_pair_indices = batch_indices_tensor[diff_indices_in_batch_tensor]
    idx1_tensor = diff_pair_indices[:, 0].to(torch.float32)
    idx2_tensor = diff_pair_indices[:, 1].to(torch.float32)

    pos1_tensor = positions_tensor[idx1_tensor.long()].to(torch.float32)
    pos2_tensor = positions_tensor[idx2_tensor.long()].to(torch.float32)

    diff_scores_tensor = scores_tensor[diff_indices_in_batch_tensor].to(torch.float32)

    return torch.cat(
        [idx1_tensor.unsqueeze(1), idx2_tensor.unsqueeze(1), pos1_tensor, pos2_tensor, diff_scores_tensor.unsqueeze(1)],
        dim=1,
    )


def save_different_pairs(
    different_pairs: List[Tuple[int, int, Tuple[int, int], Tuple[int, int], float]],
    units: List[np.ndarray],
    output_dir: str,
) -> None:
    """
    Saves the different image unit pairs to the specified directory using multiple threads.

    Args:
        different_pairs (list): A list of tuples, where each tuple contains
                                (index1, index2, pos1, pos2, score).
        units (list): The list of all image units (numpy arrays).
        output_dir (str): The directory to save the output pairs.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(
        f"\nSaving {len(different_pairs)} different pairs to '{output_dir}' directory..."
    )

    def _save_single_pair(args):
        i, (idx1, idx2, _, _, _) = args
        pair_dir = os.path.join(output_dir, f"pair_{i+1}")
        os.makedirs(pair_dir, exist_ok=True)
        Image.fromarray(units[idx1]).save(os.path.join(pair_dir, "img1.png"))
        Image.fromarray(units[idx2]).save(os.path.join(pair_dir, "img2.png"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(_save_single_pair, enumerate(different_pairs))

    print("Done saving.")


_worker_units_tensor = None
_worker_positions_np = None
_worker_threshold = None
_worker_method = None
_worker_comparison_func = None


def _init_worker_cpu(
    units_tensor: torch.Tensor,
    positions_np: np.ndarray,
    threshold: float,
    method: str,
    method_params: dict = None,
):
    """Initializes a worker process for CPU-based comparison."""
    global _worker_units_tensor, _worker_positions_np, _worker_threshold, _worker_method, _worker_comparison_func
    _worker_units_tensor = units_tensor
    _worker_positions_np = positions_np
    _worker_threshold = threshold
    _worker_method = method
    comparison_func = get_comparison_function(method, **(method_params or {}))
    if method == "ssim":
        ssim_module = get_ssim_module()
        _worker_comparison_func = lambda t1, t2: comparison_func(t1, t2, ssim_module)
    else:
        _worker_comparison_func = comparison_func


def _compare_batch_worker_cpu(
    batch_indices: List[Tuple[int, int]],
) -> List[Tuple[int, int, Tuple[int, int], Tuple[int, int], float]]:
    """Worker function for CPU-based comparison on a batch of pairs."""
    # Uses global variables set by _init_worker_cpu
    global _worker_units_tensor, _worker_positions_np, _worker_threshold, _worker_method, _worker_comparison_func

    if not batch_indices:
        return []

    batch_indices_np = np.array(batch_indices, dtype=np.int32)

    indices1 = batch_indices_np[:, 0].tolist()
    indices2 = batch_indices_np[:, 1].tolist()

    tensor1 = _worker_units_tensor[indices1]
    tensor2 = _worker_units_tensor[indices2]

    scores = _worker_comparison_func(tensor1, tensor2)
    diff_mask = get_diff_mask(scores, _worker_threshold, _worker_method)
    diff_indices_in_batch_tensor = torch.where(diff_mask)[0]

    different_pairs_batch = []
    if len(diff_indices_in_batch_tensor) > 0:
        scores_np = scores.numpy()
        diff_indices_np = diff_indices_in_batch_tensor.numpy()

        results_np = _collect_diff_pairs_from_batch_np(
            batch_indices_np, _worker_positions_np, scores_np, diff_indices_np
        )
        for row in results_np:
            idx1, idx2, p1r, p1c, p2r, p2c, score = row
            different_pairs_batch.append(
                (int(idx1), int(idx2), (int(p1r), int(p1c)), (int(p2r), int(p2c)), score)
            )
    return different_pairs_batch


def find_different_units_gpu(
    image_path: str,
    threshold: float = 0.9,
    unit_size: Tuple[int, int] = (512, 512),
    method: str = "ssim",
    method_params: dict = None,
    output_dir: str = None,
) -> None:
    """
    Finds and reports image units that are different based on SSIM using GPU.

    Args:
        image_path (str): Path to the image.
        threshold (float): SSIM threshold. Units with SSIM below this are
                         considered different.
        unit_size (tuple): The size of the image units to compare.
        method (str): The comparison method to use ('ssim', 'mse', 'color_histogram').
        method_params (dict): Optional dictionary of parameters for the comparison method.
    """
    image_units = divide_image_into_units(image_path, unit_size)

    if not image_units:
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("Error: No GPU (CUDA or MPS) available.")
        return

    print(f"Divided image into {len(image_units)} units.")

    num_comparisons = len(image_units) * (len(image_units) - 1) // 2

    start_time = time.time()
    positions = [pos for pos, unit in image_units]
    units = [unit for _, unit in image_units]
    positions_np = np.array(positions, dtype=np.int32)

    units_tensor = torch.from_numpy(np.array(units)).float().to(device)
    # (N, H, W, C) -> (N, C, H, W)
    units_tensor = units_tensor.permute(0, 3, 1, 2)

    indices_iter = itertools.combinations(range(len(image_units)), 2)

    different_pairs = []
    all_results_tensors = []
    batch_size = 1024  # Adjustable batch size
    positions_tensor = torch.from_numpy(positions_np).to(device)

    comparison_func = get_comparison_function(method, **(method_params or {}))
    if method == "ssim":
        ssim_module = get_ssim_module().to(device)
        compare_op = lambda t1, t2: comparison_func(t1, t2, ssim_module)
    else:
        compare_op = comparison_func

    with tqdm(
        total=num_comparisons,
        desc="Processing",
        bar_format="{desc} {n_fmt}/{total_fmt} pairs processed. ({rate_fmt})",
    ) as pbar:
        for batch_indices in iter(
            lambda: list(itertools.islice(indices_iter, batch_size)), []
        ):
            batch_indices_tensor = torch.tensor(
                batch_indices, dtype=torch.int32, device=device
            )
            indices1 = batch_indices_tensor[:, 0]
            indices2 = batch_indices_tensor[:, 1]

            tensor1 = units_tensor[indices1]
            tensor2 = units_tensor[indices2]

            scores = compare_op(tensor1, tensor2)
            diff_mask = get_diff_mask(scores, threshold, method)
            diff_indices_in_batch_tensor = torch.where(diff_mask)[0]
            if len(diff_indices_in_batch_tensor) > 0:
                results_tensor = _collect_diff_pairs_from_batch_gpu(
                    batch_indices_tensor,
                    positions_tensor,
                    scores,
                    diff_indices_in_batch_tensor,
                )
                all_results_tensors.append(results_tensor)
            pbar.update(len(batch_indices_tensor))
        pbar.set_description("✓ Comparison complete.")

    if all_results_tensors:
        final_results_tensor = torch.cat(all_results_tensors)
        results_np = final_results_tensor.cpu().numpy()
        different_pairs = [
            (int(r[0]), int(r[1]), (int(r[2]), int(r[3])), (int(r[4]), int(r[5])), r[6])
            for r in results_np
        ]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.4f}s")

    if num_comparisons > 0 and elapsed_time > 0:
        print(f"Comparisons per second: {num_comparisons/elapsed_time:.2f}")

    if not different_pairs:
        print("No significant differences found between any units.")
    else:
        print(f"\nFound {len(different_pairs)} different pairs.")
        if output_dir:
            save_different_pairs(different_pairs, units, output_dir)


def find_different_units_cpu(
    image_path: str,
    threshold: float = 0.9,
    unit_size: Tuple[int, int] = (512, 512),
    method: str = "ssim",
    method_params: dict = None,
    output_dir: str = None,
) -> None:
    """
    Finds and reports image units that are different based on SSIM using CPU
    with multiprocessing.

    Args:
        image_path (str): Path to the image.
        threshold (float): SSIM threshold. Units with SSIM below this are
                         considered different.
        unit_size (tuple): The size of the image units to compare.
        method (str): The comparison method to use ('ssim', 'mse', 'color_histogram').
        method_params (dict): Optional dictionary of parameters for the comparison method.
    """
    image_units = divide_image_into_units(image_path, unit_size)

    if not image_units:
        return

    print(f"Divided image into {len(image_units)} units.")

    num_comparisons = len(image_units) * (len(image_units) - 1) // 2

    start_time = time.time()
    positions = [pos for pos, unit in image_units]
    units = [unit for pos, unit in image_units]
    positions_np = np.array(positions, dtype=np.int32)

    units_tensor = torch.from_numpy(np.array(units)).float()
    # (N, H, W, C) -> (N, C, H, W)
    units_tensor = units_tensor.permute(0, 3, 1, 2)

    num_processes = multiprocessing.cpu_count()
    batch_size = 256

    combinations_iter = itertools.combinations(range(len(image_units)), 2)

    different_pairs = []

    initializer = partial(
        _init_worker_cpu,
        units_tensor=units_tensor,
        positions_np=positions_np,
        threshold=threshold,
        method=method,
        method_params=method_params,
    )

    with multiprocessing.Pool(
        processes=num_processes, initializer=initializer
    ) as pool:
        with tqdm(
            total=num_comparisons,
            desc="Processing",
            bar_format="{desc} {n_fmt}/{total_fmt} pairs processed. ({rate_fmt})",
        ) as pbar:
            results = []
            for batch in iter(
                lambda: list(itertools.islice(combinations_iter, batch_size)), []
            ):
                res = pool.apply_async(_compare_batch_worker_cpu, (batch,))
                results.append((res, len(batch)))

            for res, length in results:
                result_batch = res.get()
                if result_batch:
                    different_pairs.extend(result_batch)
                pbar.update(length)
            pbar.set_description("✓ Comparison complete.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.4f}s")

    if num_comparisons > 0 and elapsed_time > 0:
        print(f"Comparisons per second: {num_comparisons/elapsed_time:.2f}")

    if not different_pairs:
        print("No significant differences found between any units.")
    else:
        print(f"\nFound {len(different_pairs)} different pairs.")
        if output_dir:
            save_different_pairs(different_pairs, units, output_dir)
