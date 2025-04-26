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
    python modalSmoothing_volume.py <input_file> <output_file> <dilation_mm>
    ```

    - `input_file`: Path to the input NIfTI file.
    - `output_file`: Path to the output NIfTI file.
    - `dilation_mm`: Dilation parameter in millimeters.

    **Example**:

    ```sh
    python modalSmoothing_volume.py MMP_in_MNI_symmetrical_1.nii MMP_in_MNI_symmetrical_1_sm1.nii 1.0
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