import numpy as np
import argparse
import os
import warnings
from functools import partial
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    save_pickle,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np
from anyBrainer.utils.utils import parallel_process


def safe_log_append(log_path, message):
    """
    Appends a single line to a log file, safely across multiprocessing workers.
    
    Parameters:
        log_path (str): Path to the log file (can include ~)
        message (str): Line to append (no newline needed)
    """
    log_path = os.path.expanduser(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        with open(log_path, "a") as f:
            f.write(message.rstrip() + "\n")
            f.flush()
    except Exception as e:
        # Optional: fallback print for debugging
        print(f"[Logging Error] Could not write to {log_path}: {e}")

def process_single_scan(scan_info, preprocess_config, target_dir):
    """
    Process a single scan for pretraining data.

    Args:
        scan_info: A tuple containing (subject_name, session_name, scan_file, scan_path)
        preprocess_config: Preprocessing configuration dictionary
        target_dir: Target directory for preprocessed data

    Returns:
        Success message or error message
    """
    subject_name, session_name, scan_file, scan_path = scan_info

    # Extract filename without extension to use as identifier
    scan_name = os.path.splitext(os.path.splitext(scan_file)[0])[0]
    filename = f"{subject_name}_{session_name}_{scan_name}"
    save_path = join(target_dir, filename)
    
    # Log files
    log_path = "~/jobs_archive/preprocess_fomo_all.log"
    warn_path = "~/jobs_archive/preprocess_fomo_warnings.log"
    
    # Intercept warnings and log them as errors with scan name
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")  # Capture all warnings

        try:
            images, image_props = preprocess_case_for_training_without_label(
                images=[read_file_to_nifti_or_np(scan_path)], **preprocess_config
            )
            image = images[0]
            
            # Fatal check: NaN / Inf or completely empty 
            if not np.isfinite(image).all() or np.isnan(image).all():
                raise ValueError("image contains NaN/Inf or is entirely NaNs")
    
            # Save the preprocessed data
            np.save(save_path + ".npy", image)
            save_pickle(image_props, save_path + ".pkl")
            
            safe_log_append(log_path, f"SUCCESS: {filename}")
            
            # If any warnings occurred, log them
            for warn in caught_warnings:
                safe_log_append(warn_path, f"{filename}: [Warning] {str(warn.message)}")
    
            return f"Processed {subject_name}/{session_name}/{scan_file}"
        except Exception as e:
            safe_log_append(log_path, f"FAILURE: {filename}: {str(e)}")
                
            return f"Error processing {subject_name}/{session_name}/{scan_file}: {str(e)}"


def preprocess_pretrain_data(in_path: str, out_path: str, num_workers: int = None, skip_existing: bool = True):
    """
    Preprocess all pretraining data in parallel.

    Args:
        in_path: Path to the source data directory
        out_path: Path to store preprocessed data
        num_workers: Number of parallel workers (default: CPU count - 1)
        skip_existing: Skip files if preprocessed .npy versions exist in target dir.
    """
    target_dir = join(out_path, "fomo-60k_preprocessed")
    ensure_dir_exists(target_dir)

    preprocess_config = {
        "normalization_operation": ["volume_wise_znorm"],
        "crop_to_nonzero": True,
        "target_orientation": "RAS",
        "target_spacing": [1.0, 1.0, 1.0],
        "keep_aspect_ratio_when_using_target_size": False,
        "transpose": [0, 1, 2],
    }

    # Collect all scan paths
    scan_infos = []
    skipped = 0
    for subject_name in sorted(os.listdir(in_path)):
        subject_dir = os.path.join(in_path, subject_name)
        if not os.path.isdir(subject_dir):
            continue

        for session_name in sorted(os.listdir(subject_dir)):
            session_dir = os.path.join(subject_dir, session_name)
            if not os.path.isdir(session_dir):
                continue

            scan_files = [f for f in os.listdir(session_dir) if f.endswith(".nii.gz")]
            for scan_file in scan_files:
                scan_path = os.path.join(session_dir, scan_file)
                
                target_file = f"{subject_name}_{session_name}_{scan_file.split('.')[0]}.npy"
                if skip_existing and os.path.exists(join(target_dir, target_file)):
                    skipped += 1
                    continue
                
                scan_infos.append((subject_name, session_name, scan_file, scan_path))

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_scan, preprocess_config=preprocess_config, target_dir=target_dir
    )

    # Process all scans in parallel using the shared utility function
    parallel_process(process_func, scan_infos, num_workers, skipped, desc="Preprocessing scans")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path", type=str, required=True, help="Path to pretrain data"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to put preprocessed pretrain data",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers to use. Default is CPU count - 1",
    )
    parser.add_argument(
        "--skip_existing",
        type=bool,
        default=True,
        help="Skip files if preprocessed .npy versions exist in target dir.",
    )
    args = parser.parse_args()
    preprocess_pretrain_data(
        in_path=args.in_path, out_path=args.out_path, num_workers=args.num_workers, 
        skip_existing=args.skip_existing
    )
