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


def save_highlighted_image(
    sorted_stats: List[dict],
    image_path: str,
    unit_size: Tuple[int, int],
    output_path: str,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 5,
):
    """
    Highlights the most different unit on the original image and saves it.

    A red rectangle is drawn around the unit that is statistically most different
    from all other units.

    Args:
        sorted_stats (List[dict]): A list of dictionaries, sorted by mean score,
                                   each containing stats for a unit.
        image_path (str): Path to the original image.
        unit_size (Tuple[int, int]): The (width, height) of the units.
        output_path (str): The path to save the highlighted image.
        color (Tuple[int, int, int]): The BGR color of the highlight rectangle.
                                      Default is red (0, 0, 255).
        thickness (int): The thickness of the highlight rectangle's border.
    """
    import cv2

    if not sorted_stats:
        print("No stats available to determine the most different unit.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    most_different_unit_stat = sorted_stats[0]
    unit_pos = most_different_unit_stat["pos"]  # (row, col)
    unit_width, unit_height = unit_size
    row, col = unit_pos

    top_left = (col * unit_width, row * unit_height)
    bottom_right = ((col + 1) * unit_width, (row + 1) * unit_height)

    cv2.rectangle(image, top_left, bottom_right, color, thickness)

    try:
        cv2.imwrite(output_path, image)
        print(f"\nSaved highlighted image to '{output_path}'")
    except cv2.error as e:
        print(f"Error saving highlighted image to '{output_path}': {e}")


def _get_unit_stats(
    scores_per_unit: List[List[float]], method: str, positions: List[Tuple[int, int]]
) -> List[dict]:
    """
    Calculates statistical analysis for each unit based on its comparison scores.

    Args:
        scores_per_unit (List[List[float]]): A list where each inner list contains the
                                             comparison scores for a single unit against all others.
        method (str): The comparison method used, to determine score interpretation.
        positions (List[Tuple[int, int]]): The (row, col) positions of each unit.

    Returns:
        List[dict]: A list of dictionaries, sorted by mean score, each containing stats for a unit.
    """
    stats = []
    for i, scores in enumerate(scores_per_unit):
        if not scores:
            continue
        scores_np = np.array(scores)
        stats.append(
            {
                "unit": i,
                "pos": positions[i],
                "mean": np.mean(scores_np),
                "median": np.median(scores_np),
                "std": np.std(scores_np),
                "min": np.min(scores_np),
                "max": np.max(scores_np),
            }
        )

    higher_is_more_different = method in [
        "cielab",
        "color_clustering",
        "color_range_hsv",
    ]
    sort_reverse = higher_is_more_different
    return sorted(stats, key=lambda x: x["mean"], reverse=sort_reverse)


def _print_unit_stats(sorted_stats: List[dict], method: str):
    """
    Prints statistical analysis for each unit.

    Args:
        sorted_stats (List[dict]): A list of dictionaries, sorted by mean score,
                                   each containing stats for a unit.
        method (str): The comparison method used, to determine score interpretation.
    """
    print("\n--- Unit-wise Statistical Analysis ---")
    print(f"Analysis based on '{method}' scores.")
    higher_is_more_different = method in [
        "cielab",
        "color_clustering",
        "color_range_hsv",
    ]
    if higher_is_more_different:
        print("Note: Higher average scores indicate greater difference from other units.")
    else:
        print("Note: Lower average scores indicate greater difference from other units.")

    print(
        "\n{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "Unit", "Position", "Mean", "Median", "Std Dev", "Min Score", "Max Score"
        )
    )
    print("-" * 105)

    for stat in sorted_stats:
        pos_str = f"({stat['pos'][0]}, {stat['pos'][1]})"
        print(
            "{:<10} {:<15} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                stat["unit"],
                pos_str,
                stat["mean"],
                stat["median"],
                stat["std"],
                stat["min"],
                stat["max"],
            )
        )


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
    comparison_func = get_comparison_function(
        method, device="cpu", **(method_params or {})
    )
    if method == "ssim":
        ssim_module = get_ssim_module()
        _worker_comparison_func = lambda t1, t2: comparison_func(t1, t2, ssim_module)
    else:
        _worker_comparison_func = comparison_func


def _compare_batch_worker_cpu(
    batch_indices: List[Tuple[int, int]],
    analyze_units: bool,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Worker function for CPU-based comparison on a batch of pairs."""
    # Uses global variables set by _init_worker_cpu
    global _worker_units_tensor, _worker_positions_np, _worker_threshold, _worker_method, _worker_comparison_func

    if not batch_indices:
        return [], []

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

    all_scores_info = []
    if analyze_units:
        scores_np = scores.numpy()
        for i in range(len(batch_indices_np)):
            idx1, idx2 = batch_indices_np[i]
            score = scores_np[i]
            all_scores_info.append((int(idx1), int(idx2), score))

    return different_pairs_batch, all_scores_info


def find_different_units_gpu(
    image_path: str,
    threshold: float = 0.9,
    unit_size: Tuple[int, int] = (512, 512),
    method: str = "ssim",
    method_params: dict = None,
    output_dir: str = None,
    analyze_units: bool = False,
    highlight_output_path: str = None,
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
        output_dir (str): Directory to save different pairs.
        analyze_units (bool): If True, perform and print a statistical analysis for each unit.
        highlight_output_path (str): If provided, saves the original image with the most
                                     different unit highlighted. Requires `analyze_units`.
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

    num_units = len(image_units)
    print(f"Divided image into {num_units} units.")

    num_comparisons = len(image_units) * (len(image_units) - 1) // 2

    start_time = time.time()
    positions = [pos for pos, unit in image_units]
    units = [unit for _, unit in image_units]
    positions_np = np.array(positions, dtype=np.int32)

    units_tensor = torch.from_numpy(np.array(units)).float().to(device)
    # (N, H, W, C) -> (N, C, H, W)
    units_tensor = units_tensor.permute(0, 3, 1, 2)

    indices_iter = itertools.combinations(range(num_units), 2)

    different_pairs = []
    all_results_tensors = []
    batch_size = 1024  # Adjustable batch size
    scores_per_unit = [[] for _ in range(num_units)] if analyze_units else None
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

            if analyze_units:
                batch_indices_np = batch_indices_tensor.cpu().numpy()
                scores_np = scores.cpu().numpy()
                for k in range(len(batch_indices_np)):
                    idx1, idx2 = batch_indices_np[k]
                    score = scores_np[k]
                    scores_per_unit[idx1].append(score)
                    scores_per_unit[idx2].append(score)

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

    if analyze_units:
        sorted_stats = _get_unit_stats(scores_per_unit, method, positions)
        _print_unit_stats(sorted_stats, method)
        if highlight_output_path:
            save_highlighted_image(
                sorted_stats, image_path, unit_size, highlight_output_path
            )

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
    analyze_units: bool = False,
    highlight_output_path: str = None,
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
        output_dir (str): Directory to save different pairs.
        analyze_units (bool): If True, perform and print a statistical analysis for each unit.
        highlight_output_path (str): If provided, saves the original image with the most
                                     different unit highlighted. Requires `analyze_units`.
    """
    image_units = divide_image_into_units(image_path, unit_size)

    if not image_units:
        return

    num_units = len(image_units)
    print(f"Divided image into {num_units} units.")
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

    combinations_iter = itertools.combinations(range(num_units), 2)

    different_pairs = []
    scores_per_unit = [[] for _ in range(num_units)] if analyze_units else None
    all_scores_info = [] if analyze_units else None

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
                res = pool.apply_async(_compare_batch_worker_cpu, (batch, analyze_units))
                results.append((res, len(batch)))

            for res, length in results:
                result_batch, scores_info_batch = res.get()
                if result_batch:
                    different_pairs.extend(result_batch)
                if analyze_units and scores_info_batch:
                    all_scores_info.extend(scores_info_batch)
                pbar.update(length)
            pbar.set_description("✓ Comparison complete.")

    if analyze_units:
        for idx1, idx2, score in all_scores_info:
            scores_per_unit[idx1].append(score)
            scores_per_unit[idx2].append(score)
        sorted_stats = _get_unit_stats(scores_per_unit, method, positions)
        _print_unit_stats(sorted_stats, method)
        if highlight_output_path:
            save_highlighted_image(
                sorted_stats, image_path, unit_size, highlight_output_path
            )

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
