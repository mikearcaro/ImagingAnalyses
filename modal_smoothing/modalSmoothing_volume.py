#!/usr/bin/env python3
"""
roi_dilation_modal_smoothing.py

This script takes a labeled ROI NIfTI file and performs the following steps:
1. Applies binary dilation to each ROI label independently by a user-defined distance in millimeters.
2. Optionally applies modal filtering (majority filter) to smooth label edges.
3. If a binary mask is provided, both dilation and smoothing are restricted to voxels within that mask.
   The mask will be automatically resampled to match the input image dimensions if needed.

USAGE EXAMPLE:
--------------
python modalSmoothing_volume.py \
  input_rois.nii.gz \
  dilated_smoothed_rois.nii.gz \
  2.5 \
  --mask cortex_mask.nii.gz \
  --resample_to input

REQUIRED ARGUMENTS:
-------------------
input_file        Path to the input 3D NIfTI label file (e.g., atlas or ROI map).
output_file       Path to the output 3D NIfTI file with dilated and smoothed labels.
dilation_mm       Amount of dilation to apply (in millimeters).

OPTIONAL PARAMETERS:
--------------------
--mask            Path to binary mask NIfTI file. Dilation and smoothing will be limited to this region.
--resample_to     Which file to use as resampling reference: "input" (default) or "mask"

NOTES:
------
- Assumes isotropic voxel size of 1.0 mm. You can change this in the code if needed.
- Modal filter kernel size is set to 1.0 mm (also configurable in the code).
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import jit, prange
from tqdm import tqdm
import argparse
import warnings

def load_nifti(file_name):
    nifti_img = nib.load(file_name)
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    header = nifti_img.header
    return data, affine, header

def save_nifti(data, affine, file_name):
    new_img = nib.Nifti1Image(data, affine)
    nib.save(new_img, file_name)

def resample_to_target(source_data, source_affine, target_shape, target_affine, nearest=True, binary=False):
    from nibabel.processing import resample_from_to
    source_img = nib.Nifti1Image(source_data, source_affine)
    target_img = nib.Nifti1Image(np.zeros(target_shape), target_affine)
    order = 0 if nearest else 1
    resampled = resample_from_to(source_img, target_img, order=order)
    data = resampled.get_fdata()
    return (data > 0) if binary else data

def dilate_roi_label(data, label, struct, dilation_voxels, mask=None):
    mask_label = data == label
    dilated_mask = binary_dilation(mask_label, structure=struct, iterations=dilation_voxels)
    if mask is not None:
        dilated_mask &= mask
    return label, dilated_mask

def dilate_roi(data, voxel_size, dilation_mm, mask=None):
    struct = generate_binary_structure(3, 1)
    dilation_voxels = int(np.round(dilation_mm / voxel_size))
    dilated_data = np.zeros_like(data)
    labels = [label for label in np.unique(data) if label != 0]

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(dilate_roi_label, data, label, struct, dilation_voxels, mask): label
            for label in labels
        }

        with tqdm(total=len(labels), desc="Dilating ROIs") as pbar:
            for future in as_completed(futures):
                label, dilated_mask = future.result()
                dilated_data[dilated_mask] = label
                pbar.update(1)

    return dilated_data

@jit(nopython=True, parallel=True)
def apply_modal_filter_numba(data, filter_size):
    filtered_data = np.zeros_like(data)
    half_size = filter_size // 2
    for x in prange(half_size, data.shape[0] - half_size):
        for y in prange(half_size, data.shape[1] - half_size):
            for z in prange(half_size, data.shape[2] - half_size):
                window = data[x - half_size:x + half_size + 1,
                              y - half_size:y + half_size + 1,
                              z - half_size:z + half_size + 1]
                filtered_data[x, y, z] = np.bincount(window.flatten().astype(np.int32)).argmax()
    return filtered_data

@jit(nopython=True, parallel=True)
def apply_modal_filter_masked_numba(data, mask, filter_size):
    filtered_data = data.copy()
    half_size = filter_size // 2
    for x in prange(half_size, data.shape[0] - half_size):
        for y in prange(half_size, data.shape[1] - half_size):
            for z in prange(half_size, data.shape[2] - half_size):
                if not mask[x, y, z]:
                    continue
                window_data = data[x - half_size:x + half_size + 1,
                                   y - half_size:y + half_size + 1,
                                   z - half_size:z + half_size + 1]
                window_mask = mask[x - half_size:x + half_size + 1,
                                   y - half_size:y + half_size + 1,
                                   z - half_size:z + half_size + 1]
                window_flat = window_data.flatten()
                mask_flat = window_mask.flatten()
                valid_vals = window_flat[mask_flat > 0]
                if len(valid_vals) > 0:
                    filtered_data[x, y, z] = np.bincount(valid_vals.astype(np.int32)).argmax()
    return filtered_data

def apply_modal_filter(data, voxel_size, filter_mm, mask=None):
    filter_size = int(np.round(filter_mm / voxel_size))
    filter_size = max(3, filter_size // 2 * 2 + 1)
    if mask is not None:
        return apply_modal_filter_masked_numba(data, mask.astype(np.bool_), filter_size)
    else:
        return apply_modal_filter_numba(data, filter_size)

def main(input_file_name, output_file_name, dilation_mm, mask_file=None, resample_to="input", voxel_size=1.0, filter_mm=1.0):
    data, affine, header = load_nifti(input_file_name)
    mask = None

    if mask_file:
        mask_data, mask_affine, _ = load_nifti(mask_file)
        if resample_to == "input":
            if mask_data.shape != data.shape:
                warnings.warn("Mask shape does not match input. Resampling to input dimensions...")
                mask = resample_to_target(mask_data, mask_affine, data.shape, affine, nearest=True, binary=True)
            else:
                mask = mask_data > 0
        elif resample_to == "mask":
            if data.shape != mask_data.shape:
                warnings.warn("Input shape does not match mask. Resampling input to mask dimensions...")
                data = resample_to_target(data, affine, mask_data.shape, mask_affine, nearest=True, binary=False).astype(data.dtype)
                affine = mask_affine
            mask = resample_to_target(mask_data, mask_affine, data.shape, affine, nearest=True, binary=True)
        else:
            raise ValueError("--resample_to must be either 'input' or 'mask'")

    dilated_data = dilate_roi(data, voxel_size, dilation_mm, mask=mask)
    modal_smoothed_data = apply_modal_filter(dilated_data, voxel_size, filter_mm, mask=mask)

    if mask is not None:
        modal_smoothed_data[~mask] = 0

    save_nifti(modal_smoothed_data, affine, output_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply modal smoothing with dilation to a NIfTI file.")
    parser.add_argument("input_file", help="Input NIfTI file name")
    parser.add_argument("output_file", help="Output NIfTI file name")
    parser.add_argument("dilation_mm", type=float, help="Dilation parameter in mm")
    parser.add_argument("--mask", help="Optional binary mask NIfTI file to restrict dilation and smoothing to masked region")
    parser.add_argument("--resample_to", choices=["input", "mask"], default="input", help="Which file to use as resampling reference: 'input' (default) or 'mask'")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.dilation_mm, mask_file=args.mask, resample_to=args.resample_to)
