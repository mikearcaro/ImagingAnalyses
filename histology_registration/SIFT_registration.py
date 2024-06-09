"""
This script performs slice-by-slice alignment of brain images using SIFT 
(Scale-Invariant Feature Transform) and homography transformation. 
The goal is to align each image in a sequence to a reference slice, 
facilitating the construction of a coherent 3D volume from 2D slices. 
This is particularly useful in histological analyses where precise alignment of slices is crucial.

The process involves:
1. Resizing images for faster processing.
2. Detecting key points and descriptors using SIFT.
3. Matching descriptors between slices.
4. Estimating homography to align slices.
5. Applying the transformation and saving the aligned images.

Instructions to run the script:

1. Run without starting from a specific image (default to the first image):
    python segment_brain_slice_batch.py

2. Run with debug mode enabled (outputs current processes to window):
    python segment_brain_slice_batch.py debug

3. Run starting from a specific image (e.g., Slice0112_co.png):
    python segment_brain_slice_batch.py Slice0112_co.png

In the displayed window, follow instructions. Remember to click the save button 
to write the mask and histology masked files

User-defined variables:
"""

# Directory containing the input images
INPUT_DIRECTORY = "/Users/macmini_m1/Desktop/histology_alignment/orig/masked"

# Filename of the reference slice
REF_SLICE = "Slice0111_co.png"

# List of image IDs to be skipped in the analyses
# For example, ['0112'] will skip Slice0112_co_aligned.png
SKIP_IMAGES = []

# Scale factor for resizing images to speed up processing
SCALE_FACTOR = 0.25

# Directory to save the output aligned images. 
# Default is to save as subdirectory relative to input directory
OUTPUT_DIRECTORY = "sift_aligned"

# Suffix to append to the output aligned images
# Added to end of input filenames before extension
OUTPUT_PREFIX = "_aligned"

import cv2
import numpy as np
import os
import re
import shutil

def register_image(img1_path, img2_path, scale_factor):
    img1_color = cv2.imread(img1_path)  # Image to be aligned.
    img2_color = cv2.imread(img2_path)  # Reference image.

    if img1_color is None or img2_color is None:
        raise FileNotFoundError(f"Error: One of the images did not load correctly: {img1_path}, {img2_path}")

    # Resize images for faster processing
    img1_color_resized = cv2.resize(img1_color, (0, 0), fx=scale_factor, fy=scale_factor)
    img2_color_resized = cv2.resize(img2_color, (0, 0), fx=scale_factor, fy=scale_factor)

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color_resized, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color_resized, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift_detector = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift_detector.detectAndCompute(img1, None)
    kp2, des2 = sift_detector.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter out poor matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    matches = good_matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Get low-res and high-res sizes
    low_height, low_width = img1.shape
    height, width = img1_color.shape[:2]
    low_size = np.float32([[0, 0], [0, low_height], [low_width, low_height], [low_width, 0]])
    high_size = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    # Compute scaling transformations
    scale_up = cv2.getPerspectiveTransform(low_size, high_size)
    scale_down = cv2.getPerspectiveTransform(high_size, low_size)

    # Combine the transformations. Remember that the order of the transformation
    # is reversed when doing matrix multiplication
    # so this is actually scale_down -> H -> scale_up
    h_and_scale_up = np.matmul(scale_up, H)
    scale_down_h_scale_up = np.matmul(h_and_scale_up, scale_down)

    return scale_down_h_scale_up

def apply_homography(img, homography):
    height, width = img.shape[:2]
    transformed_img = cv2.warpPerspective(img, homography, (width, height))
    return transformed_img

def main(directory, refslice, scale_factor=0.5, output_dir="cv_aligned", output_prefix="_aligned"):
    # Create output directories if they do not exist
    output_directory = os.path.join(directory, output_dir)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get list of image files in the directory
    files = [f for f in os.listdir(directory) if re.match(r'Slice\d+_co\.png', f)]
    files.sort()

    # Create a mapping from file names to their indices
    file_indices = {f: int(re.search(r'\d+', f).group()) for f in files}

    # Extract index of reference slice
    ref_index = file_indices[refslice]

    # Log the files being processed
    print("Files to be processed:", files)
    print("Reference index:", ref_index)

    # Copy the reference slice to the output directory with _aligned
    ref_slice_path = os.path.join(directory, refslice)
    aligned_ref_slice_path = os.path.join(output_directory, refslice.replace(".png", output_prefix + ".png"))
    shutil.copyfile(ref_slice_path, aligned_ref_slice_path)
    print(f"Copied {refslice} to {refslice.replace('.png', output_prefix + '.png')}")

    ref_index_in_list = files.index(refslice)

    # Filter out images to be skipped
    skipped_files = [f for f in files if any(skip_id in f for skip_id in SKIP_IMAGES)]
    files = [f for f in files if not any(skip_id in f for skip_id in SKIP_IMAGES)]
    print(f"Skipping files: {', '.join(skipped_files)}")

    # Perform descending slice-by-slice alignment
    for i in range(ref_index_in_list - 1, -1, -1):
        current_slice = files[i]
        ref_slice = files[i + 1].replace(".png", output_prefix + ".png")
        ref_slice_path = os.path.join(output_directory, ref_slice)
        
        current_slice_path = os.path.join(directory, current_slice)
        if os.path.exists(current_slice_path) and os.path.exists(ref_slice_path):
            print(f"Aligning {current_slice} to {ref_slice}")
            homography = register_image(current_slice_path, ref_slice_path, scale_factor)
            original_img = cv2.imread(current_slice_path)
            aligned_img = apply_homography(original_img, homography)
            output_path = os.path.join(output_directory, current_slice.replace(".png", output_prefix + ".png"))
            cv2.imwrite(output_path, aligned_img)
            if os.path.exists(output_path):
                print(f"Successfully saved: {current_slice.replace('.png', output_prefix + '.png')}")
            else:
                print(f"Failed to save: {current_slice.replace('.png', output_prefix + '.png')}")
        else:
            print(f"Skipping alignment for {current_slice} or {ref_slice} because one of them does not exist.")

    # Perform ascending slice-by-slice alignment
    for i in range(ref_index_in_list + 1, len(files)):
        current_slice = files[i]
        ref_slice = files[i - 1].replace(".png", output_prefix + ".png")
        ref_slice_path = os.path.join(output_directory, ref_slice)
        
        current_slice_path = os.path.join(directory, current_slice)
        if os.path.exists(current_slice_path) and os.path.exists(ref_slice_path):
            print(f"Aligning {current_slice} to {ref_slice}")
            homography = register_image(current_slice_path, ref_slice_path, scale_factor)
            original_img = cv2.imread(current_slice_path)
            aligned_img = apply_homography(original_img, homography)
            output_path = os.path.join(output_directory, current_slice.replace(".png", output_prefix + ".png"))
            cv2.imwrite(output_path, aligned_img)
            if os.path.exists(output_path):
                print(f"Successfully saved: {current_slice.replace('.png', output_prefix + '.png')}")
            else:
                print(f"Failed to save: {current_slice.replace('.png', output_prefix + '.png')}")
        else:
            print(f"Skipping alignment for {current_slice} or {ref_slice} because one of them does not exist.")

if __name__ == "__main__":
    # Define the input directory and reference slice
    input_directory = INPUT_DIRECTORY
    ref_slice = REF_SLICE
    scale_factor = SCALE_FACTOR  # Define your desired scale factor here
    output_dir = OUTPUT_DIRECTORY
    output_prefix = OUTPUT_PREFIX

    # Call the main function
    main(input_directory, ref_slice, scale_factor, output_dir, output_prefix)
