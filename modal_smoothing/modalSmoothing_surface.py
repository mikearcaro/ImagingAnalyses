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
