import subprocess
import sys
import re
import concurrent.futures

# Ensure necessary packages are installed
required_packages = ['antspyx', 'numpy', 'nibabel', 'scipy', 'glob2']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import ants
import numpy as np
import nibabel as nib
import os
import glob
import tempfile
from scipy.ndimage import gaussian_filter

# Common variables
T1_DIRECTORY = '/Users/macmini_m1/Desktop/anat_test/T1_images'
MASK_DIRECTORY = '/Users/macmini_m1/Desktop/anat_test/masks'
OUTPUT_DIRECTORY = os.path.join('/Users/macmini_m1/Desktop/anat_test', 'output')
T1_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'orig')
MASK_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'masks')
os.makedirs(T1_OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIRECTORY, exist_ok=True)

NUMBER_OF_RANDOM_POINTS = 100  # Number of points for deformation
SD_NOISE = 2.5  # Standard deviation of noise for deformation
MAX_BIAS = 0.4  # Maximum bias intensity as a fraction of the image intensity range
BIAS_SIGMA = 15  # Sigma for Gaussian filter in bias field generation
ROTATION_ANGLES = (-3, 3)  # Range for random rotation angles in degrees
NUM_DEFORMATIONS = 2  # Number of deformations to apply
NUM_INTENSITY_BIASES = 1  # Number of intensity biases to apply
NUM_ROTATIONS = 2  # Number of rotations to apply
NUM_SPECKLE_NOISE_VERSIONS = 2  # Number of speckle noise versions to apply
NOISE_VARIANCES = [0.001, 0.01, 0.1]  # Different levels of speckle noise variance
NUM_CPUS = 4  # Number of CPUs to use for parallel processing

# Common aspects of filenames
T1_PATTERN = re.compile(r'(.+)_meanTP_(\d{6})\.nii\.gz')
MASK_PATTERN = re.compile(r'(.+)_meanTP_mask_[^_]+_(\d{6})\.nii\.gz')

def generate_bias_field(image, max_bias=MAX_BIAS):
    """
    Generate a smooth bias field using Perlin noise.
    
    Parameters:
    image (ANTsImage): The input image to match the physical space.
    max_bias (float): Maximum bias intensity as a fraction of the image intensity range.
    
    Returns:
    ANTsImage: The generated bias field.
    """
    image_shape = image.shape
    # Create a random noise field
    bias_field = np.random.rand(*image_shape)
    
    # Smooth the noise field
    bias_field = gaussian_filter(bias_field, sigma=BIAS_SIGMA)
    
    # Normalize the bias field to range [1 - max_bias, 1 + max_bias]
    bias_field = (bias_field - bias_field.min()) / (bias_field.max() - bias_field.min())
    bias_field = 1.0 + max_bias * (2.0 * bias_field - 1.0)
    
    # Convert to ANTs image and set the physical space to match the input image
    bias_field = ants.from_numpy(bias_field)
    bias_field.set_spacing(image.spacing)
    bias_field.set_origin(image.origin)
    bias_field.set_direction(image.direction)
    
    return bias_field

def apply_intensity_bias(image, bias_field):
    """
    Apply a bias field to an image.
    
    Parameters:
    image (ANTsImage): The input image.
    bias_field (ANTsImage): The bias field to apply.
    
    Returns:
    ANTsImage: The biased image.
    """
    biased_image = image * bias_field
    return biased_image

def apply_random_rotation(image, angles=None):
    """
    Apply a random rotation to an image or use provided angles.
    
    Parameters:
    image (ANTsImage): The input image.
    angles (array-like): Optional. The rotation angles in radians.
    
    Returns:
    ANTsImage: The rotated image.
    """
    if angles is None:
        angles = np.radians(np.random.uniform(*ROTATION_ANGLES, size=3))  # Random rotation angles in radians
    center = np.array(image.shape) / 2
    rotation = ants.create_ants_transform(
        transform_type='Euler3DTransform', 
        center=center, 
        parameters=angles)
    rotated_image = ants.apply_ants_transform_to_image(rotation, image, image)
    return rotated_image, angles

def add_speckle_noise(image, variance):
    """
    Add speckle (multiplicative) noise to an image.
    
    Parameters:
    image (np.ndarray): The input image data.
    variance (float): Variance of the speckle noise.
    
    Returns:
    np.ndarray: The noisy image data.
    """
    noise = np.random.randn(*image.shape) * np.sqrt(variance)
    noisy_image = image * (1 + noise)
    return noisy_image

def save_nifti(image_data, affine, filename):
    """
    Save an image as a NIFTI file.
    
    Parameters:
    image_data (np.ndarray): The image data to save.
    affine (np.ndarray): The affine transformation matrix.
    filename (str): The output file name.
    """
    nib.save(nib.Nifti1Image(image_data, affine), filename)

def retry_rotation(image, mask, affine):
    """
    Retry the rotation until the rotated image is valid (not all zeros).
    
    Parameters:
    image (ANTsImage): The image to rotate.
    mask (ANTsImage): The mask to rotate with the same transformation.
    affine (np.ndarray): The affine transformation matrix.
    
    Returns:
    tuple: The valid rotated image and mask data.
    """
    while True:
        rotated_img, rotation_angles = apply_random_rotation(image)
        rotated_img_data = rotated_img.numpy()
        if np.any(rotated_img_data):  # Check if the rotated image is not all zeros
            rotated_mask, _ = apply_random_rotation(mask, angles=rotation_angles)
            rotated_mask_data = (rotated_mask.numpy() > 0.5).astype(np.int32)
            return rotated_img_data, rotated_mask_data

# Function to check if T1 and mask files match
def check_matching_files(t1_file, mask_file):
    t1_match = re.match(T1_PATTERN, os.path.basename(t1_file))
    mask_match = re.match(MASK_PATTERN, os.path.basename(mask_file))
    if not t1_match or not mask_match:
        return False
    return t1_match.group(1) == mask_match.group(1) and t1_match.group(2) == mask_match.group(2)

# Processing function for parallel execution
def process_file_pair(t1_file, mask_file):
    base_name = os.path.basename(t1_file).replace('.nii.gz', '')
    mask_base_name = os.path.basename(mask_file).replace('.nii.gz', '')
    
    for i in range(NUM_DEFORMATIONS):
        print(f'Processing {t1_file} and {mask_file}, deformation {i + 1}...')

        # Load the NIfTI image and mask
        img = ants.image_read(t1_file)
        mask = ants.image_read(mask_file)

        # Generate a synthetic displacement field using simulate_displacement_field
        displacement_field = ants.simulate_displacement_field(
            domain_image=img,
            field_type="bspline",
            number_of_random_points=NUMBER_OF_RANDOM_POINTS,
            sd_noise=SD_NOISE
        )

        # Save the displacement field to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
            displacement_field_path = temp_file.name
            ants.image_write(displacement_field, displacement_field_path)

        # Apply the displacement field to the T1 image and mask
        transformed_img = ants.apply_transforms(fixed=img, moving=img, transformlist=[displacement_field_path], interpolator='linear')
        transformed_mask = ants.apply_transforms(fixed=mask, moving=mask, transformlist=[displacement_field_path], interpolator='nearestNeighbor')

        # Convert transformed mask to binary
        transformed_mask_data = (transformed_mask.numpy() > 0.5).astype(np.int32)

        # Use the affine from the original NIfTI image
        original_img_nib = nib.load(t1_file)
        affine = original_img_nib.affine

        # Save the deformed T1 image and mask without rotation
        save_nifti(transformed_img.numpy(), affine, os.path.join(T1_OUTPUT_DIRECTORY, f'{base_name}_deformed_{i + 1}_bias_0_rot_0.nii.gz'))
        save_nifti(transformed_mask_data, affine, os.path.join(MASK_OUTPUT_DIRECTORY, f'{mask_base_name}_deformed_{i + 1}_bias_0_rot_0.nii.gz'))

        # Apply random intensity biases
        for j in range(NUM_INTENSITY_BIASES):
            print(f'Applying intensity bias {j + 1} for deformation {i + 1}...')

            # Generate a bias field for the transformed image
            bias_field = generate_bias_field(transformed_img)

            biased_img = apply_intensity_bias(transformed_img, bias_field)

            # Save the deformed and biased T1 image without rotation
            save_nifti(biased_img.numpy(), affine, os.path.join(T1_OUTPUT_DIRECTORY, f'{base_name}_deformed_{i + 1}_bias_{j + 1}_rot_0.nii.gz'))

            # Save the same mask without rotation for all intensity biases
            save_nifti(transformed_mask_data, affine, os.path.join(MASK_OUTPUT_DIRECTORY, f'{mask_base_name}_deformed_{i + 1}_bias_{j + 1}_rot_0.nii.gz'))

            # Apply random rotations
            for k in range(NUM_ROTATIONS):
                print(f'Applying random rotation {k + 1} for deformation {i + 1}, intensity bias {j + 1}...')

                rotated_img_data, rotated_mask_data = retry_rotation(biased_img, transformed_mask, affine)

                # Save the rotated T1 image
                rotated_img_nib = nib.Nifti1Image(rotated_img_data, affine)
                rotated_image_filename = os.path.join(T1_OUTPUT_DIRECTORY, f'{base_name}_deformed_{i + 1}_bias_{j + 1}_rot_{k + 1}.nii.gz')
                nib.save(rotated_img_nib, rotated_image_filename)

                # Save the rotated mask
                rotated_mask_nib = nib.Nifti1Image(rotated_mask_data, affine)
                rotated_mask_filename = os.path.join(MASK_OUTPUT_DIRECTORY, f'{mask_base_name}_deformed_{i + 1}_bias_{j + 1}_rot_{k + 1}.nii.gz')
                nib.save(rotated_mask_nib, rotated_mask_filename)

                # Apply different levels of speckle noise to rotated images
                for noise_variance in NOISE_VARIANCES:
                    for l in range(NUM_SPECKLE_NOISE_VERSIONS):
                        print(f'Applying speckle noise (variance {noise_variance}) version {l + 1} for deformation {i + 1}, intensity bias {j + 1}, rotation {k + 1}...')
                        noisy_img_data = add_speckle_noise(rotated_img_data, variance=noise_variance)

                        # Save the noisy T1 image
                        noisy_img_nib = nib.Nifti1Image(noisy_img_data, affine)
                        noisy_image_filename = os.path.join(T1_OUTPUT_DIRECTORY, f'{base_name}_deformed_{i + 1}_bias_{j + 1}_rot_{k + 1}_noise_{l + 1}_var_{noise_variance}.nii.gz')
                        nib.save(noisy_img_nib, noisy_image_filename)

                        # Save the mask again, no need to add noise to the mask
                        save_nifti(rotated_mask_data, affine, os.path.join(MASK_OUTPUT_DIRECTORY, f'{mask_base_name}_deformed_{i + 1}_bias_{j + 1}_rot_{k + 1}_noise_{l + 1}_var_{noise_variance}.nii.gz'))

        # Clean up the temporary file
        os.remove(displacement_field_path)

# Find all T1 and mask files that match the patterns
t1_files = glob.glob(os.path.join(T1_DIRECTORY, '*_meanTP_*.nii.gz'))
mask_files = glob.glob(os.path.join(MASK_DIRECTORY, '*_meanTP_mask_*.nii.gz'))

# Pair T1 and mask files
file_pairs = []
for t1_file in t1_files:
    for mask_file in mask_files:
        if check_matching_files(t1_file, mask_file):
            file_pairs.append((t1_file, mask_file))
            break

if not file_pairs:
    print("No matching T1 and mask files found.")
    sys.exit()

# Process file pairs in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
    futures = [executor.submit(process_file_pair, t1_file, mask_file) for t1_file, mask_file in file_pairs]
    for future in concurrent.futures.as_completed(futures):
        future.result()  # Raise any exceptions encountered during processing

print('All MRI files have been processed.')

