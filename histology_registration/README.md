# Image Processing and Registration Scripts

This repository contains three scripts designed for processing and aligning brain slice images. Each script has its own specific purpose and functionality, ranging from brain segmentation to slice-by-slice image alignment using SIFT and diffeomorphic registration. Below are detailed explanations and instructions for using each script.

The code is still being adapted to improve flexibility and applicability to various histological images.

## 1. Brain Segmentation Script

This script is designed to segment brain slices from histological images, creating binary masks for further analysis. The segmentation process involves user interaction to refine the masks using region growing and manual editing.

### Features

- Interactive segmentation using region growing and manual editing.
- Save masks and masked images for each slice.
- Adjustable parameters for segmentation tolerance and zoom level.

### Dependencies

- `numpy`
- `cv2`
- `skimage`
- `matplotlib`

### Setup

1. **Directory Structure**:
    - Ensure your histological images are stored in a directory, e.g., `/path/to/histology_images`.
    - The script will create directories for saving masks and masked images.

2. **User-defined Variables**:
    ```python
    # Base directory containing the images to process
    BASE_DIRECTORY = "/Users/macmini_m1/Desktop/histology_alignment/orig"

    # Directory where masks will be saved
    MASKS_DIR = os.path.join(BASE_DIRECTORY, "masks")

    # Directory where masked images will be saved
    MASKED_DIR = os.path.join(BASE_DIRECTORY, "masked")

    # Regex pattern to match the image files to be processed
    FILE_PATTERN = r'Slice\d+_co\.png'

    # Enable or disable debug mode
    DEBUG_MODE = False

    # Initial tolerance value for region growing
    INITIAL_TOLERANCE = 50

    # Maximum axis size for downsampling
    MAX_AXIS_SIZE = 1000

    # Zoom level as a proportion of the short axis
    ZOOM_LEVEL = 0.1

    # List of image IDs to be skipped in the analyses
    # For example, ['0112'] will skip Slice0112_co.png
    SKIP_IMAGES = []
    ```

### How to Use

1. **Run the Script**:
    ```bash
    python segment_brain_slice_batch.py
    ```

2. **With Debug Mode**:
    ```bash
    python segment_brain_slice_batch.py debug
    ```

3. **Start from a Specific Image**:
    ```bash
    python segment_brain_slice_batch.py Slice0112_co.png
    ```

4. **Interactive Segmentation**:
    - Follow the instructions displayed in the window to refine the mask.
    - Click the save button to write the mask and masked histology files.

## 2. SIFT Registration Script

This script performs slice-by-slice alignment of brain images using SIFT (Scale-Invariant Feature Transform) and homography transformation. The goal is to align each image in a sequence to a reference slice, facilitating the construction of a coherent 3D volume from 2D slices.

### Features

- Resize images for faster processing.
- Detect key points and descriptors using SIFT.
- Match descriptors between slices.
- Estimate homography to align slices.
- Apply the transformation and save the aligned images.

### Dependencies

- `cv2`
- `numpy`
- `shutil`
- `os`
- `re`

### Setup

1. **Directory Structure**:
    - Ensure your brain slice images are stored in a directory, e.g., `/path/to/histology_alignment/orig/masked`.

2. **User-defined Variables**:
    ```python
    # Directory containing the input images
    INPUT_DIRECTORY = "/Users/macmini_m1/Desktop/histology_alignment/orig/masked"

    # Filename of the reference slice
    REF_SLICE = "Slice0111_co.png"

    # List of image IDs to be skipped in the analyses
    # For example, ['0112'] will skip Slice0112_co.png
    SKIP_IMAGES = []

    # Scale factor for resizing images to speed up processing
    SCALE_FACTOR = 0.25

    # Directory to save the output aligned images
    OUTPUT_DIRECTORY = "cv_aligned"

    # Suffix to append to the output aligned images
    OUTPUT_PREFIX = "_aligned"
    ```

### How to Use

1. **Run the Script**:
    ```bash
    python sift_registration.py
    ```

## 3. Dipy Diffeomorphic Registration Script

This script performs diffeomorphic registration on a series of brain slice images, aligning each slice to a reference slice to create a smooth, continuous volume. The primary use of this script is to preprocess and align histological slices, making them suitable for subsequent 3D volume reconstruction.

### Features

- Preprocessing steps like CLAHE and bias field correction to enhance image quality.
- Gaussian smoothing to reduce noise.
- Diffeomorphic registration using either CCMetric or SSDMetric as the similarity measure.
- Optionally save intermediate results and visualizations in debug mode.

### Dependencies

- `numpy`
- `cv2`
- `shutil`
- `os`
- `re`
- `skimage`
- `dipy`
- `SimpleITK`
- `scipy`

### Setup

1. **Directory Structure**:
    - Ensure your brain slice images are stored in a directory, e.g., `/path/to/histology_alignment/orig/masked/cv_aligned`.

2. **User-defined Variables**:
    ```python
    # Directory containing the input images
    INPUT_DIRECTORY = "/Users/macmini_m1/Desktop/histology_alignment/orig/masked/cv_aligned"

    # Filename of the reference slice
    REF_SLICE = "Slice0111_co_aligned.png"

    # List of image IDs to be skipped in the analyses
    # For example, ['0112'] will skip Slice0112_co_aligned.png
    SKIP_IMAGES = []

    # Downsample scale factor for the images to speed up registration
    SCALE_FACTOR = 0.25

    # Flags to enable CLAHE and bias field correction
    APPLY_CLAHE = True
    APPLY_BIAS_CORRECTION = True

    # Parameters for CLAHE
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)

    # Parameters for Gaussian Blur
    GAUSSIAN_BLUR_KERNEL = (3, 3)
    GAUSSIAN_BLUR_SIGMA = 1

    # Parameters for diffeomorphic registration of image pairs
    LEVEL_ITERS = [200, 150, 100, 50, 25]
    STEP_LENGTH = 0.15
    INV_ITER = 50
    METRIC_TYPE = "SSD"

    # Debug mode flag
    DEBUG_MODE = True
    ```

### How to Use

1. **Run the Script**:
    ```bash
    python dipy_diffeomorphic_registration.py
    ```

## Additional Notes

- Ensure that all dependencies are installed before running the scripts.
- Adjust the user-defined variables as per your requirements.
- Follow the instructions in the displayed windows for interactive segmentation and alignment processes.

This README provides an overview of the functionalities and usage instructions for each script. For more detailed information, refer to the inline comments within the scripts themselves.
