# Modal Smoothing for Neuroimaging Data

This repository provides Python scripts for applying modal smoothing to neuroimaging data, both in volumetric and cortical surface formats. These scripts are designed to smooth regions of interest (ROIs) by filling in gaps and ensuring a more contiguous labeling.

## Volume Modal Smoothing

### Description

The volume modal smoothing script applies modal smoothing to 3D volumetric neuroimaging data. This is useful for data in NIfTI format, where you have ROI labels that need smoothing.

### Usage

1. **Install dependencies**:

    ```sh
    pip install nibabel numpy scipy numba tqdm
    ```

2. **Run the script**:

    ```sh
    python modalSmoothing_parallel.py <input_file> <output_file> <dilation_mm>
    ```

    - `input_file`: Path to the input NIfTI file.
    - `output_file`: Path to the output NIfTI file.
    - `dilation_mm`: Dilation parameter in millimeters.

    **Example**:

    ```sh
    python modalSmoothing_parallel.py MMP_in_MNI_symmetrical_1.nii MMP_in_MNI_symmetrical_1_sm1.nii 1.0
    ```

### Script

```python
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import jit, prange
from tqdm import tqdm
import argparse

def load_nifti(file_name):
    nifti_img = nib.load(file_name)
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    return data, affine

def save_nifti(data, affine, file_name):
    new_img = nib.Nifti1Image(data, affine)
    nib.save(new_img, file_name)

def dilate_roi_label(data, label, struct, dilation_voxels):
    mask = data == label
    dilated_mask = binary_dilation(mask, structure=struct, iterations=dilation_voxels)
    return label, dilated_mask

def dilate_roi(data, voxel_size, dilation_mm):
    struct = generate_binary_structure(3, 1)
    dilation_voxels = int(np.round(dilation_mm / voxel_size))
    dilated_data = np.zeros_like(data)

    labels = [label for label in np.unique(data) if label != 0]
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(dilate_roi_label, data, label, struct, dilation_voxels): label for label in labels}
        
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
                window = data[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1, z - half_size:z + half_size + 1]
                filtered_data[x, y, z] = np.bincount(window.flatten().astype(np.int32)).argmax()
    return filtered_data

def apply_modal_filter(data, voxel_size, filter_mm):
    filter_size = int(np.round(filter_mm / voxel_size))
    filter_size = max(3, filter_size // 2 * 2 + 1)
    return apply_modal_filter_numba(data, filter_size)

def main(input_file_name, output_file_name, dilation_mm, voxel_size=1.0, filter_mm=1.0):
    data, affine = load_nifti(input_file_name)
    dilated_data = dilate_roi(data, voxel_size, dilation_mm)
    modal_smoothed_data = apply_modal_filter(dilated_data, voxel_size, filter_mm)
    save_nifti(modal_smoothed_data, affine, output_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply modal smoothing with dilation to a NIfTI file.")
    parser.add_argument("input_file", help="Input NIfTI file name")
    parser.add_argument("output_file", help="Output NIfTI file name")
    parser.add_argument("dilation_mm", type=float, help="Dilation parameter in mm")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.dilation_mm)
