# MRI Augmentation and Deformation Script

This script performs a variety of augmentation and deformation operations on MRI images and their corresponding masks. It generates multiple versions of the images with different deformations, intensity biases, rotations, and noise levels. This script is designed to create synthetic imaging data with realistic brain properties, which can be used to train image segmentation algorithms when only limited samples are available.

The script leverages tools from the antspyx package and uses parallel processing to handle multiple files efficiently, ensuring faster processing times.

## Features

- Generates synthetic displacement fields to deform MRI images.
- Applies random intensity biases to the images.
- Applies random rotations to the images.
- Adds speckle noise to the images.
- Ensures that the corresponding masks undergo the same transformations as their MRI images.
- Processes multiple files in parallel using `concurrent.futures`.

## Dependencies

The script requires the following Python packages:

- `antspyx`
- `numpy`
- `nibabel`
- `scipy`
- `glob2`

The script will automatically check for these dependencies and install them if they are not already installed.

## Setup

1. **Directory Structure**: 
    - Ensure your MRI images (T1 images) are stored in a directory, e.g., `/path/to/T1_images`.
    - Ensure your corresponding masks are stored in a separate directory, e.g., `/path/to/masks`.
    - Modify the script to set the correct paths for these directories.

2. **Output Directories**:
    - The script will create an output directory and subdirectories for the augmented T1 images and masks.

## How to Use

1. **Set Input and Output Paths**:
    - Update the variables `T1_DIRECTORY` and `MASK_DIRECTORY` in the script to point to your directories containing T1 images and masks.
    - The output directories are set up within the script and created automatically.

2. **Run the Script**:
    - Execute the script using Python:
      ```bash
      python deformT1.py
      ```

3. **Control Parameters**:
    - You can control various parameters such as the number of deformations, intensity biases, rotations, speckle noise versions, and the number of CPUs used for parallel processing by modifying the corresponding variables at the top of the script.

## Example

### Directory Structure

```text
/path/to/
    ├── T1_images/
    │   ├── subject1_meanTP_123456.nii.gz
    │   └── subject2_meanTP_654321.nii.gz
    ├── masks/
    │   ├── subject1_meanTP_mask_cortex_123456.nii.gz
    │   └── subject2_meanTP_mask_cortex_654321.nii.gz
    └── output/
        ├── orig/
        │   ├── subject1_meanTP_123456_deformed_1_bias_0_rot_0.nii.gz
        │   └── ...
        ├── masks/
        │   ├── subject1_meanTP_mask_cortex_123456_deformed_1_bias_0_rot_0.nii.gz
        │   └── ...
```

##  Script Configuration
In the script, ensure the following variables are correctly set:


``` python
Copy code
T1_DIRECTORY = '/path/to/T1_images'
MASK_DIRECTORY = '/path/to/masks'
OUTPUT_DIRECTORY = '/path/to/output'
Running the Script
Execute the script:
``` 
``` bash
python deformT1.py
```

This will process all paired T1 images and masks, apply the specified transformations, and save the results in the output directory.

##  Notes
The script ensures that the T1 and mask files are correctly paired based on their filenames.
The transformations are applied in parallel to utilize multiple CPU cores for faster processing.
