import nibabel as nib
import numpy as np
from numba import jit, prange
import argparse
import os

def load_gifti_surface(surface_file):
    print(f"Loading GIFTI surface from", surface_file)
    gifti_img = nib.load(surface_file)
    coords = gifti_img.darrays[0].data
    faces = gifti_img.darrays[1].data.astype(np.int32)
    print(f"Loaded surface with {coords.shape[0]} vertices and {faces.shape[0]} faces.")
    return coords, faces

def load_sparse_1d_dset(dset_file, n_vertices):
    print(f"Loading sparse data from", dset_file)
    data = np.loadtxt(dset_file)
    nodelist = data[:, 0].astype(np.int32)
    values = data[:, -1].astype(np.int32)
    full_array = np.zeros(n_vertices, dtype=np.int32)
    full_array[nodelist] = values
    print(f"Loaded {len(nodelist)} nodes. Converted to full array of {n_vertices} nodes.")
    return full_array

def save_1d_dset(n_vertices, roi_values, file_name):
    print(f"Saving smoothed ROI data to", file_name)
    nodelist = np.arange(n_vertices)
    data = np.column_stack((nodelist, roi_values))
    np.savetxt(file_name, data, fmt='%d')
    print("Data saved successfully.")

def compute_topological_neighbors(faces, n_vertices):
    print("Computing topological neighbors from surface triangles...")
    neighbors = [[] for _ in range(n_vertices)]
    for tri in faces:
        a, b, c = tri
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])

    # Remove duplicates
    neighbors = [np.unique(n).astype(np.int32) for n in neighbors]
    print("Topological neighbors computed.")
    return neighbors

@jit(nopython=True, parallel=True)
def modal_smoothing_on_surface(roi_values, neighbors, mask):
    smoothed_data = np.copy(roi_values)
    for i in prange(roi_values.shape[0]):
        if mask[i] == 0:
            continue  # Only update nodes inside mask

        neighbor_labels = []
        for j in range(len(neighbors[i])):
            neighbor_labels.append(roi_values[neighbors[i][j]])

        neighbor_labels = np.array(neighbor_labels)
        neighbor_labels = neighbor_labels[neighbor_labels != 0]  # Exclude zeros

        if len(neighbor_labels) > 0:
            counts = np.bincount(neighbor_labels)
            smoothed_data[i] = np.argmax(counts)
        else:
            smoothed_data[i] = roi_values[i]  # No good neighbors: retain original
    return smoothed_data

def main(surface_file, roi_file, output_file, mask_file=None, num_iterations=1):
    coords, faces = load_gifti_surface(surface_file)
    n_vertices = coords.shape[0]

    roi_values = load_sparse_1d_dset(roi_file, n_vertices)

    if mask_file is not None:
        mask = load_sparse_1d_dset(mask_file, n_vertices)
    else:
        mask = np.ones(n_vertices, dtype=np.int32)

    neighbors = compute_topological_neighbors(faces, n_vertices)

    print("Starting modal smoothing...")
    smoothed_data = roi_values.copy()
    for it in range(num_iterations):
        print(f"Iteration {it+1}")
        smoothed_data = modal_smoothing_on_surface(smoothed_data, neighbors, mask)
    print("Modal smoothing completed.")

    save_1d_dset(n_vertices, smoothed_data, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply modal smoothing to a cortical surface using GIFTI and sparse .1D.dset files.")
    parser.add_argument("surface_file", help="Input GIFTI surface file (.gii)")
    parser.add_argument("roi_file", help="Sparse .1D.dset file with ROI data")
    parser.add_argument("--output_file", help="Output .1D.dset file", default=None)
    parser.add_argument("--mask", type=str, default=None, help="Optional sparse .1D.dset mask file")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of smoothing iterations (default: 1)")
    args = parser.parse_args()

    if args.output_file is None:
        base, ext = os.path.splitext(args.roi_file)
        if ext == ".dset" and base.endswith(".1D"):
            base = base[:-3]
            ext = ".1D.dset"
        args.output_file = f"{base}_topo_smoothed_{args.num_iterations}iter{ext}"

    main(args.surface_file, args.roi_file, args.output_file, args.mask, args.num_iterations)