# Modal Smoothing for Neuroimaging Data

This repository provides Python scripts for applying modal smoothing to neuroimaging data, both in volumetric and cortical surface formats. These scripts are designed to smooth regions of interest (ROIs) by filling in gaps and ensuring a more contiguous labeling.

## Volume Modal Smoothing

### Description

The volume modal smoothing script applies binary dilation followed by modal (majority) filtering to 3D volumetric neuroimaging label data. This is especially useful for smoothing ROI labels in NIfTI format while optionally restricting the smoothing operation to a binary mask.


### Features

- Applies binary dilation to each unique non-zero label independently.
- Applies modal filtering (majority vote in a voxel neighborhood) to smooth label edges.
- Allows optional restriction of smoothing to a provided binary mask.
- Supports automatic resampling of input or mask to match dimensions.

### Dependencies

Ensure the following Python packages are installed:

```sh
pip install nibabel numpy scipy numba tqdm
```

### Usage

```sh
python modalSmoothing_volume.py <input_file> <output_file> <dilation_mm> [--mask <mask_file>] [--resample_to <reference>]
```

**Arguments**:
- `input_file`: Path to the input 3D NIfTI file (typically a label map).
- `output_file`: Path to the output 3D NIfTI file.
- `dilation_mm`: Amount of dilation to apply in millimeters.

**Optional Parameters**:
- `--mask`: Path to a binary NIfTI file to restrict smoothing.
- `--resample_to`: Specify which file to use as the reference space for resampling. Options are `input` (default) or `mask`.

### Example

```sh
python modalSmoothing_volume.py input_rois.nii.gz smoothed_rois.nii.gz 2.5 --mask cortex_mask.nii.gz --resample_to input
```

This command smooths and dilates `input_rois.nii.gz` with 2.5 mm dilation and restricts modal smoothing to regions defined in `cortex_mask.nii.gz`, resampling the mask to match the input ROI volume.


### Script

```python
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

```

## Surface Modal Smoothing

### Description

The surface modal smoothing script applies modal smoothing along the cortical surface topology (mesh triangles), not in Euclidean space.
It is intended for GIFTI-format surfaces and sparse .1D.dset ROI label files.

This method ensures anatomically accurate smoothing across the cortical sheet, correctly respecting sulci and gyri, and filling holes based on surface connectivity.

### Usage

1. **Install dependencies**:

    ```sh
    pip install nibabel numpy scipy numba
    ```

2. **Run the script**:

    ```sh
    python modalSmoothing_surface.py <surface_file> <roi_file> [--mask <mask_file>] [--num_iterations <n>]
    ```

    - `surface_file`: Path to the input GIFTI surface file.
    - `roi_file`: Path to the `.1D.dset` file with ROI data.
    - `--num_iterations: (Optional) Number of smoothing iterations to apply. Default is 1.
    - `--mask: (Optional) Path to a .1D.dset mask file. Only nodes where mask==1 will be updated..   
    - `--output_file`: (Optional) Path to the output `.1D.dset` file. 

If --output_file is not specified, an output file is automatically generated based on the input ROI file and number of iterations.

    **Example**:
    Single smoothing pass (good for small holes):
    
    ```sh
    python modalSmoothing_surface.py rh.pial.gii allvisual-rh.1D.dset --mask maskforsmoothing_rh.1D.dset --num_iterations 5
    ```

    This will generate an output file named `allvisual-rh_topo_smoothed_5iter.1D.dset`.

### Script

```python
import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
from numba import jit, prange
import argparse
import os

def load_gifti_surface(surface_file):
    print(f"Loading GIFTI surface from {surface_file}...")
    gifti_img = nib.load(surface_file)
    coords = gifti_img.darrays[0].data
    print(f"Loaded GIFTI surface with {coords.shape[0]} vertices.")
    return coords

def load_1d_dset(dset_file):
    print(f"Loading ROI data from {dset_file}...")
    data = np.loadtxt(dset_file)
    nodelist = data[:, 0].astype(np.int32)
    roi_values = data[:, -1].astype(np.int32)
    print(f"Loaded ROI data with {data.shape[0]} nodes.")
    return nodelist, roi_values

def save_1d_dset(nodelist, roi_values, file_name):
    print(f"Saving smoothed ROI data to {file_name}...")
    data = np.column_stack((nodelist, roi_values))
    np.savetxt(file_name, data, fmt='%d')
    print("Data saved successfully.")

def compute_neighbors(vertices, filter_size):
    print("Computing neighbors for each vertex...")
    kdtree = KDTree(vertices)
    neighbors = []
    for i in range(vertices.shape[0]):
        _, idxs = kdtree.query(vertices[i], k=filter_size)
        neighbors.append(idxs)
    neighbors = np.array(neighbors)
    print("Neighbors computed.")
    return neighbors

@jit(nopython=True, parallel=True)
def modal_smoothing_on_surface(roi_values, neighbors):
    smoothed_data = np.zeros_like(roi_values)
    for i in prange(roi_values.shape[0]):
        window = roi_values[neighbors[i]]
        # Exclude 0 labels
        window = window[window != 0]
        if len(window) > 0:
            smoothed_data[i] = np.bincount(window).argmax()
        else:
            smoothed_data[i] = 0
    return smoothed_data

def main(surface_file, roi_file, output_file, filter_size):
    vertices = load_gifti_surface(surface_file)
    nodelist, roi_values = load_1d_dset(roi_file)
    
    neighbors = compute_neighbors(vertices, filter_size)
    print("Starting modal smoothing...")
    smoothed_data = modal_smoothing_on_surface(roi_values, neighbors)
    print("Modal smoothing completed.")
    
    save_1d_dset(nodelist, smoothed_data, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply modal smoothing to a cortical surface using GIFTI and .1D.dset files.")
    parser.add_argument("surface_file", help="Input GIFTI surface file")
    parser.add_argument("roi_file", help=".1D.dset file with ROI data")
    parser.add_argument("filter_size", type=int, help="Filter size (number of neighboring vertices to consider)")
    parser.add_argument("--output_file", help="Output .1D.dset file (optional)", default=None)
    args = parser.parse_args()

    if args.output_file is None:
        base, ext = os.path.splitext(args.roi_file)
        if ext == ".dset" and base.endswith(".1D"):
            base = base[:-3]  # Remove the trailing ".1D"
            ext = ".1D.dset"  # Reset the extension to ".1D.dset"
        args.output_file = f"{base}_ssm{args.filter_size}{ext}"

    main(args.surface_file, args.roi_file, args.output_file, args.filter_size)
```