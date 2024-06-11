### README

This script processes converts 1D.dset files (AFNI/SUMA) to annot (FreeSurfer). It generates ROI annotations for surface-based analysis. It assumes a specific folder / data organization structure that is unlieky to generalize.

#### Requirements
- Python 3.x
- Nibabel
- Pandas
- NumPy

#### Usage
1. Place the script in the directory containing your data folders.
2. Ensure your data folders follow a four-letter naming convention.
3. Each data folder should contain:
   - `rois`: ROI data files.
   - `surf`: Surface geometry files.

#### Script Functionality
1. **Iterating Over Data Folders**:
   - Iterates over all four-letter named folders in the current directory.

2. **Loading ROI Values**:
   - Loads ROI values from `allparietal-rh.1D.dset` in the `rois` subdirectory.
   - Stores data in a Pandas DataFrame.

3. **Loading Surface Geometry**:
   - Loads surface geometry data from `rh.inflated` in the `surf` subdirectory.
   - Determines surface size.

4. **Generating ROI Values**:
   - Generates ROI values for the entire surface based on loaded data.
   - Fills vertices without an ROI label with zeros.

5. **Color Table Creation**:
   - Generates a color table for ROIs.

6. **Saving New Annotation File**:
   - Saves an annotation file (`allparietal-rh.annot`) in the `rois` subdirectory.
   - Includes ROI values and color information.

#### Important Notes
- Ensure correct paths to ROI and surface geometry files.
- This script generates random colors for ROIs, which can be modified.



