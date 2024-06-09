"""
This script is designed to perform diffeomorphic registration on a series of brain slice images, 
aligning each slice to a reference slice to create a smooth, continuous volume. 
The primary use of this script is to preprocess and align histological slices, 
making it suitable for subsequent 3D Volume reconstruction.

The script includes preprocessing steps such as CLAHE (Contrast Limited Adaptive Histogram Equalization) 
and bias field correction to enhance image quality and improve registration results. 
During the registration process, the script supports both the CCMetric (cross-correlation) and 
SSDMetric (sum of squared differences) as similarity measures.

The process involves:
1. Converting images to grayscale for consistent processing.
2. Resizing images for faster processing.
3. Preprocessing the images with CLAHE (Contrast Limited Adaptive Histogram Equalization) and bias field correction to enhance image quality.
4. Applying Gaussian smoothing to reduce noise.
5. Performing diffeomorphic registration using either CCMetric (cross-correlation) or SSDMetric (sum of squared differences) as the similarity measure.
6. Applying the transformation to align the moving image to the fixed image.
7. Optionally saving intermediate results and visualizations in debug mode, including downsampled images, overlays, and deformation fields.

Debug mode can be activated by setting the DEBUG_MODE variable to True. 
In debug mode, the script saves intermediate results and visualizations, 
including downsampled images, overlays, and deformation fields. 
This aids in understanding and verifying the registration process.

If you receive an error, it can be helpful to debug by adding the following before the error to see 
the current state of the image: 
cv2.imwrite('test.png',{specify current image})
  and/or 
print the image types to the console window with:
print(f"Image type: {moving.dtype}, Image shape: {moving.shape}")

NOTE: The CLAHE was originally coded to work with RGB images. 
The current version runs on greyscale but the old code remains commented out.

User-defined variables:
"""

# Directory containing the input images
INPUT_DIRECTORY = ""

# Filename of the reference slice
REF_SLICE = "Slice0111_co_aligned.png"

# List of image IDs to be skipped in the analyses
# For example, ['0112'] will skip Slice0112_co_aligned.png
SKIP_IMAGES = []

# Downsample scale factor for the images to speed up registration. 
# Resulting transformation is applied to original resolution images.
SCALE_FACTOR = 0.25

# Flags to enable CLAHE and bias field correction
APPLY_CLAHE = True  # Set to True to apply CLAHE
APPLY_BIAS_CORRECTION = True  # Set to True to apply bias field correction

# Parameters for CLAHE (Contrast Limited Adaptive Histogram Equalization)
# CLAHE enhances the contrast of images, making features more distinct and improving registration accuracy.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Parameters for Gaussian Blur
# Gaussian Blur smooths the images and reduces noise, which helps in achieving better registration.
APPLY_GAUSSIAN_SMOOTHING = True  # Set to False to skip Gaussian smoothing
GAUSSIAN_BLUR_KERNEL = (3, 3)
GAUSSIAN_BLUR_SIGMA = 1

# Parameters for diffeomorphic registration of image pairs
LEVEL_ITERS = [200, 150, 100, 50, 25]  # Number of iterations at each level of the multi-resolution pyramid
STEP_LENGTH = 0.15  # Step size for gradient descent
INV_ITER = 50  # Number of inverse consistency iterations
# Metric type for registration ("CC" for cross-correlation or "SSD" for sum of squared differences)
METRIC_TYPE = "SSD"

# Debug mode flag (set to True to enable debug mode)
DEBUG_MODE = True


import numpy as np
import cv2
import shutil
import os
import re
from skimage import exposure
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration, DiffeomorphicMap
from dipy.align.metrics import SSDMetric, CCMetric
import scipy.ndimage
from dipy.viz import regtools
import SimpleITK as sitk


def clahe_equalization(image):
    # Convert to LAB color space
    #lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # Split the LAB image into L, A, and B channels
    #lab_planes = cv2.split(lab_img) 

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    #clahe_images = [clahe.apply(plane) for plane in lab_planes]
    # Combine the CLAHE enhanced L-channel with A and B channels
    #updated_lab_img2 = cv2.merge(clahe_images)
    # Convert LAB image back to RGB color space
    #processed_rgb_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2RGB)
    
    processed_img = clahe.apply(image)
    #cv2.imwrite('test.png',processed_img)
    return processed_img

def n4_bias_field_correction(image):
    # Convert to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(sitk_image, mask_image)

    # Convert the SimpleITK images to numpy arrays for visualization
    original_array = sitk.GetArrayFromImage(sitk_image)
    mask_array = sitk.GetArrayFromImage(mask_image)
    corrected_array = sitk.GetArrayFromImage(corrected_image)

    #cv2.imwrite('test.png',corrected_array)
    return corrected_array

def preprocess_image(image):
    if APPLY_CLAHE:
        image = clahe_equalization(image)
    if APPLY_BIAS_CORRECTION:
        image = n4_bias_field_correction(image)
    return image

def register_image(fixed_img, moving_img, scale_factor=SCALE_FACTOR):
    # Convert to grayscale
    moving = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)
    fixed = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2GRAY)

    # Downsample the images
    moving = cv2.resize(moving, (0, 0), fx=scale_factor, fy=scale_factor)
    fixed = cv2.resize(fixed, (0, 0), fx=scale_factor, fy=scale_factor)

    # Preprocess the images
    fixed_preprocessed = preprocess_image(fixed)
    moving_preprocessed = preprocess_image(moving)
    
    # Optionally apply Gaussian smoothing
    if APPLY_GAUSSIAN_SMOOTHING:
        moving = cv2.GaussianBlur(moving_preprocessed, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA)
        fixed = cv2.GaussianBlur(fixed_preprocessed, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA)
    else:
        moving = moving_preprocessed
        fixed = fixed_preprocessed

    # Perform diffeomorphic registration
    dim = fixed.ndim
    if METRIC_TYPE == "CC":
        metric = CCMetric(dim)
    else:
        metric = SSDMetric(dim)
   
    # Perform diffeomorphic registration
    sdr = SymmetricDiffeomorphicRegistration(
        metric, LEVEL_ITERS, step_length=STEP_LENGTH, inv_iter=INV_ITER
    )
    mapping = sdr.optimize(fixed, moving)
    transformed_img = mapping.transform(moving, 'linear')

    return mapping, transformed_img, fixed_preprocessed

def apply_transformation(moving, mapping, scale_factor=SCALE_FACTOR):
    # Upsample the transformation to the original resolution
    forward_upsampled = scipy.ndimage.zoom(mapping.forward, zoom=[1/scale_factor, 1/scale_factor, 1], order=3)
    backward_upsampled = scipy.ndimage.zoom(mapping.backward, zoom=[1/scale_factor, 1/scale_factor, 1], order=3)

    # Create a new DiffeomorphicMap with the upsampled transformations
    upsampled_mapping = DiffeomorphicMap(mapping.dim, moving.shape[:2])
    upsampled_mapping.forward = forward_upsampled
    upsampled_mapping.backward = backward_upsampled

    # Convert the original image to floating point format
    moving_img_float = moving.astype(np.float32)

    # Apply the transformation to each channel separately for both images
    transformed_moving_channels = []
    for i in range(moving_img_float.shape[2]):
        channel = moving_img_float[:, :, i]
        transformed_channel = upsampled_mapping.transform(channel, 'linear')
        transformed_moving_channels.append(transformed_channel)
    transformed_moving_img = np.stack(transformed_moving_channels, axis=2)

    return transformed_moving_img

def save_debug_info(fixed_img, fixed_preprocessed, transformed_img_downsampled, transformed_img, mapping, current_slice, output_dir, debug_dir_downsampled, debug_dir_origres, scale_factor):
    # Save downsampled transformed image
    downsampled_output_path = os.path.join(debug_dir_downsampled, current_slice)
    cv2.imwrite(downsampled_output_path, transformed_img_downsampled.astype(np.uint8))
    
    # Upsample the transformation to the original resolution
    forward_upsampled = scipy.ndimage.zoom(mapping.forward, zoom=[1/scale_factor, 1/scale_factor, 1], order=3)
    backward_upsampled = scipy.ndimage.zoom(mapping.backward, zoom=[1/scale_factor, 1/scale_factor, 1], order=3)
    upsampled_mapping = DiffeomorphicMap(mapping.dim, fixed_img.shape[:2])
    upsampled_mapping.forward = forward_upsampled
    upsampled_mapping.backward = backward_upsampled
    
    # Process downsampled images for overlay
    fixed_preprocessed_gray = fixed_preprocessed if len(fixed_preprocessed.shape) == 2 else cv2.cvtColor(fixed_preprocessed, cv2.COLOR_BGR2GRAY)
    transformed_gray_downsampled = transformed_img_downsampled if len(transformed_img_downsampled.shape) == 2 else cv2.cvtColor(transformed_img_downsampled.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    regtools.overlay_images(fixed_preprocessed_gray, transformed_gray_downsampled, 'Fixed (Downsampled)', 'Overlay', 'Moving (After Registration - Downsampled)', os.path.join(debug_dir_downsampled, f'{current_slice}_downsampled_overlay.png'))
    regtools.plot_2d_diffeomorphic_map(mapping, 10, os.path.join(debug_dir_downsampled, f'{current_slice}_downsampled_deformation_field.png'))
    
    # Original resolution processing
    fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2GRAY)
    transformed_gray_orig = cv2.cvtColor(cv2.resize(transformed_img.astype(np.uint8), (fixed_img.shape[1], fixed_img.shape[0])), cv2.COLOR_BGR2GRAY)
    
    regtools.overlay_images(fixed_gray, transformed_gray_orig, 'Fixed (Original Resolution)', 'Overlay', 'Moving (After Registration - Original Resolution)', os.path.join(debug_dir_origres, f'{current_slice}_overlay.png'))
    regtools.plot_2d_diffeomorphic_map(upsampled_mapping, 10, os.path.join(debug_dir_origres, f'{current_slice}_deformation_field.png'))

def main(directory, refslice, debug=False):
    # Notify user if debug mode is enabled
    if debug:
        print("Debug mode is enabled.")

    # Create output directory if it does not exist
    output_dir = os.path.join(directory, "dipy_aligned")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if debug:
        debug_dir_downsampled = os.path.join(output_dir, "diffeo_downsampled")
        debug_dir_origres = os.path.join(output_dir, "diffeo_origres")
        os.makedirs(debug_dir_downsampled, exist_ok=True)
        os.makedirs(debug_dir_origres, exist_ok=True)

    # Get list of image files in the directory
    files = [f for f in os.listdir(directory) if re.match(r'Slice\d+_co_aligned\.png', f)]
    files.sort()

    # Create a mapping from file names to their indices
    file_indices = {f: int(re.search(r'\d+', f).group()) for f in files}

    # Extract index of reference slice
    if refslice not in file_indices:
        print(f"Reference slice {refslice} not found in directory.")
        return

    ref_index = file_indices[refslice]

    # Log the files being processed
    print("Files to be processed:", files)
    print("Reference index:", ref_index)

    # Copy the reference slice to the output directory with _aligned
    ref_slice_path = os.path.join(directory, refslice)
    aligned_ref_slice_path = os.path.join(output_dir, refslice)
    shutil.copyfile(ref_slice_path, aligned_ref_slice_path)
    print(f"Copied {refslice} to {refslice}")

    ref_index_in_list = files.index(refslice)

    # Filter out images to be skipped
    skipped_files = [f for f in files if any(skip_id in f for skip_id in SKIP_IMAGES)]
    files = [f for f in files if not any(skip_id in f for skip_id in SKIP_IMAGES)]
    print(f"Skipping files: {', '.join(skipped_files)}")

    # Perform descending slice-by-slice alignment
    for i in range(ref_index_in_list - 1, -1, -1):
        current_slice = files[i]
        ref_slice = files[i + 1]
        ref_slice_path = os.path.join(output_dir, ref_slice)
        
        current_slice_path = os.path.join(directory, current_slice)
        if os.path.exists(current_slice_path) and os.path.exists(ref_slice_path):
            print(f"Aligning {current_slice} to {ref_slice}")
            #fixed_img = cv2.imread(ref_slice_path).astype(np.float32)
            #moving_img = cv2.imread(current_slice_path).astype(np.float32)
            fixed_img = cv2.imread(ref_slice_path)
            moving_img = cv2.imread(current_slice_path)
            mapping, transformed_img_downsampled, fixed_preprocessed = register_image(fixed_img, moving_img)

            transformed_img = apply_transformation(moving_img, mapping)
            output_path = os.path.join(output_dir, current_slice)
            cv2.imwrite(output_path, transformed_img.astype(np.uint8))

            if debug:
                save_debug_info(fixed_img, fixed_preprocessed, transformed_img_downsampled, transformed_img, mapping, current_slice, output_dir, debug_dir_downsampled, debug_dir_origres, SCALE_FACTOR)

            if os.path.exists(output_path):
                print(f"Successfully saved: {current_slice}")
            else:
                print(f"Failed to save: {current_slice}")
        else:
            print(f"Skipping alignment for {current_slice} or {ref_slice} because one of them does not exist.")

    # Perform ascending slice-by-slice alignment
    for i in range(ref_index_in_list + 1, len(files)):
        current_slice = files[i]
        ref_slice = files[i - 1]
        ref_slice_path = os.path.join(output_dir, ref_slice)
        
        current_slice_path = os.path.join(directory, current_slice)
        if os.path.exists(current_slice_path) and os.path.exists(ref_slice_path):
            print(f"Aligning {current_slice} to {ref_slice}")
            fixed_img = cv2.imread(ref_slice_path)
            moving_img = cv2.imread(current_slice_path)
            mapping, transformed_img_downsampled, fixed_preprocessed = register_image(fixed_img, moving_img)

            transformed_img = apply_transformation(moving_img, mapping)
            output_path = os.path.join(output_dir, current_slice)
            cv2.imwrite(output_path, transformed_img.astype(np.uint8))

            if debug:
                save_debug_info(fixed_img, fixed_preprocessed, transformed_img_downsampled, transformed_img, mapping, current_slice, output_dir, debug_dir_downsampled, debug_dir_origres, SCALE_FACTOR)

            if os.path.exists(output_path):
                print(f"Successfully saved: {current_slice}")
            else:
                print(f"Failed to save: {current_slice}")
        else:
            print(f"Skipping alignment for {current_slice} or {ref_slice} because one of them does not exist.")

if __name__ == "__main__":
    # Define the input directory and reference slice
    input_directory = INPUT_DIRECTORY
    ref_slice = REF_SLICE
    debug_mode = DEBUG_MODE

    # Call the main function
    main(input_directory, ref_slice, debug=debug_mode)
