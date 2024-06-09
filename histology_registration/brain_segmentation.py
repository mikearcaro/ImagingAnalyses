"""
This script is used for segmenting brain slices from histological images. 
The primary objective is to generate masks for each brain slice, 
which can be used for further analysis or visualization. 
The script offers interactive editing capabilities, allowing users to refine the generated masks manually.

Instructions to run the script:

1. Run without starting from a specific image (default to the first image):
    python segment_brain_slice_batch.py

2. Run with debug mode enabled (outputs current processes to window):
    python segment_brain_slice_batch.py debug

3. Run starting from a specific image (e.g., Slice0112_co.png):
    python segment_brain_slice_batch.py Slice0112_co.png

In the displayed window, follow instructions. Remember to click the save button 
to write the mask and histology masked files

"""

# The base directory containing the images to process
BASE_DIRECTORY = "/Users/macmini_m1/Desktop/histology_alignment/orig"

# The directory where masks will be saved
MASKS_DIR = os.path.join(BASE_DIRECTORY, "masks")

# The directory where masked images will be saved
MASKED_DIR = os.path.join(BASE_DIRECTORY, "masked")

# The regex pattern to match the image files to be processed
FILE_PATTERN = r'Slice\d+_co\.png'

# Maximum axis size for downsampling
# This parameter defines the maximum size of the image's largest dimension (height or width) after downsampling.
# Downsampling helps in reducing the image size, which can speed up the processing time.
MAX_AXIS_SIZE = 1000

# Initial tolerance value for region growing
# This parameter controls the sensitivity of the region growing algorithm used in the segmentation process.
# A higher value will allow more variation in pixel intensity, resulting in larger regions being grown.
INITIAL_TOLERANCE = 50

# Zoom level as a proportion of the short axis
# This parameter sets the zoom level for the magnified view in the editor as a proportion of the shortest axis of the image.
# It helps in providing a detailed view of a specific region when editing the mask.
ZOOM_LEVEL = 0.1

# Enable or disable debug mode
DEBUG_MODE = False


import os
import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import flood
from matplotlib.widgets import LassoSelector, Slider, Button
from matplotlib.path import Path
import matplotlib.pyplot as plt
import sys
import re

# Initialize directories
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(MASKED_DIR, exist_ok=True)

def determine_best_overlay_color(image_rgb):
    # Calculate the mean color values
    mean_color = image_rgb.mean(axis=(0, 1))

    # Choose the channel with the lowest mean intensity for the overlay
    # This will ensure the overlay has good contrast
    if mean_color[0] < mean_color[1] and mean_color[0] < mean_color[2]:
        return (255, 0, 0)  # Red
    elif mean_color[1] < mean_color[0] and mean_color[1] < mean_color[2]:
        return (0, 255, 0)  # Green
    else:
        return (0, 0, 255)  # Blue

class MaskEditor:
    def __init__(self, images, masks, image_files, current_index=0, initial_tolerance=50, debug=False):
        self.images = images
        self.masks = masks
        self.image_files = image_files
        self.current_index = current_index
        self.image_rgb = images[current_index]
        self.original_mask = masks[current_index].copy()
        self.mask = masks[current_index]
        self.short_axis = min(self.image_rgb.shape[:2])
        self.zoom_level = int(self.short_axis * ZOOM_LEVEL)
        self.history = []  # To store the history of mask changes for undo functionality
        self.tolerance = initial_tolerance
        self.debug = debug
        self.overlay_color = determine_best_overlay_color(self.image_rgb)
        self.fig, self.ax = plt.subplots(2, 3, figsize=(20, 11), gridspec_kw={'height_ratios': [2, 1]})

        self.fig.subplots_adjust(hspace=0.05, wspace=0.05)  # Adjust the spacing between subplots
        # Remove x and y axes labels
        for ax in self.ax.flatten():
            ax.set_xticks([])
            ax.set_yticks([])   

        self.fig.suptitle("Close window to save current mask and move to next image.\n"
                          "Press 'q' to quit the program.\n"
                          "Press 'r' to restart edits for the current image.\n"
                          "Press 'u' to undo the previous change.\n")
        
        # Navigation buttons and current slice filename at the top right
        self.ax_filename = self.fig.add_axes([0.75, 0.94, 0.2, 0.04])
        self.ax_filename.axis('off')
        self.filename_text = self.ax_filename.text(0.5, 0.5, self.image_files[self.current_index], 
                                                   horizontalalignment='center', verticalalignment='center',
                                                   fontsize=12, color='blue')

        self.ax_prev = plt.axes([0.70, 0.90, 0.04, 0.04])
        self.ax_next = plt.axes([0.75, 0.90, 0.04, 0.04])
        self.ax_save = plt.axes([0.80, 0.90, 0.06, 0.04])

        self.btn_prev = Button(self.ax_prev, '<')
        self.btn_next = Button(self.ax_next, '>')
        self.btn_save = Button(self.ax_save, 'Save')

        self.btn_prev.on_clicked(self.prev_slice)
        self.btn_next.on_clicked(self.next_slice)
        self.btn_save.on_clicked(self.save_current_mask)

        self.ax[0, 0].imshow(self.image_rgb)
        self.ax[0, 0].set_title("Original Image (Left click to add, right click to remove)")
        self.mask_display = self.ax[0, 1].imshow(self.mask, cmap='gray')
        self.ax[0, 1].set_title("Brain Slice Mask (Left click to add, right click to remove)")
        self.overlay_display = self.ax[0, 2].imshow(self.overlay_image())
        self.ax[0, 2].set_title("Overlay of Mask on Histology")

        self.lasso = LassoSelector(self.ax[0, 1], onselect=self.onselect)
        self.coords = []
        self.add_mode = True  # True for adding regions, False for removing regions
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.update_magnifier)

        # Add vertical slider for tolerance
        self.ax_tol = plt.axes([0.01, 0.25, 0.03, 0.65])
        self.slider_tol = Slider(self.ax_tol, 'Tolerance', 1, 100, valinit=initial_tolerance, orientation='vertical')
        self.slider_tol.on_changed(self.update_tolerance)

        # Magnified views
        self.magnifier_img = self.ax[1, 0].imshow(np.zeros_like(self.image_rgb), aspect='auto')
        self.ax[1, 0].axis('off')
        self.ax[1, 0].set_title('Magnified View (Original Image)')

        self.magnifier_mask = self.ax[1, 1].imshow(np.zeros_like(self.mask), cmap='gray', aspect='auto')
        self.ax[1, 1].axis('off')
        self.ax[1, 1].set_title('Magnified View (Mask)')

        self.magnifier_overlay = self.ax[1, 2].imshow(np.zeros_like(self.image_rgb), aspect='auto')
        self.ax[1, 2].axis('off')
        self.ax[1, 2].set_title('Magnified View (Overlay)')

        self.update_display()

    def on_button_press(self, event):
        if event.inaxes == self.ax[0, 0]:
            x, y = int(event.xdata), int(event.ydata)
            if self.debug:
                print(f"Clicked on Original Image at ({x}, {y})")
            if event.button == 1:  # Left click
                self.add_mode = True
                self.region_grow(x, y)
            elif event.button == 3:  # Right click
                self.add_mode = False
                self.region_grow(x, y)
        elif event.inaxes == self.ax[0, 1]:
            if event.button == 1:  # Left click
                self.add_mode = True
                if self.debug:
                    print("Left click detected on Mask")
            elif event.button == 3:  # Right click
                self.add_mode = False
                if self.debug:
                    print("Right click detected on Mask")

    def region_grow(self, x, y):
        if self.debug:
            print(f"Performing region grow at ({x}, {y}), add_mode: {self.add_mode}")
        grayscale_img = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
        new_region = flood(grayscale_img, (y, x), tolerance=self.tolerance, connectivity=1)
        
        self.history.append(self.mask.copy())  # Save the current mask to history before modification

        if self.add_mode:
            self.mask[new_region] = 255
            if self.debug:
                print("Added region to mask")
        else:
            self.mask[new_region] = 0
            if self.debug:
                print("Removed region from mask")
        self.update_display()

    def onselect(self, verts):
        path = Path(verts)
        self.coords.append((path, self.add_mode))
        if self.debug:
            print("Lasso selection detected")
        self.update_mask()

    def update_mask(self):
        y, x = np.indices(self.mask.shape)
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        for path, mode in self.coords:
            mask = path.contains_points(points)
            mask = mask.reshape(self.mask.shape)
            self.history.append(self.mask.copy())  # Save the current mask to history before modification
            if mode:  # Add mode
                self.mask[mask] = 255
                if self.debug:
                    print("Adding region to mask")
            else:  # Remove mode
                self.mask[mask] = 0
                if self.debug:
                    print("Removing region from mask")
        self.update_display()

    def update_display(self):
        self.mask_display.set_data(self.mask)
        self.overlay_display.set_data(self.overlay_image())
        self.fig.canvas.draw_idle()

    def update_tolerance(self, val):
        if self.debug:
            print(f"Updated tolerance to {val}")
        self.tolerance = val

    def on_key_press(self, event):
        if event.key == 'q':
            if self.debug:
                print("Quitting the editor")
            plt.close('all')
            sys.exit()
        elif event.key == 'r':
            if self.debug:
                print("Restarting the mask for the current image")
            self.mask = self.original_mask.copy()
            self.update_display()
        elif event.key == 'u':
            if self.history:
                self.mask = self.history.pop()  # Restore the last saved mask from history
                if self.debug:
                    print("Undo the previous change")
                self.update_display()

    def overlay_image(self):
        # Create an overlay of the mask on the histology image
        overlay = self.image_rgb.copy()
        alpha = 0.5  # Transparency factor
        mask_3d = np.zeros_like(overlay)
        mask_3d[self.mask == 255] = self.overlay_color
        overlay = cv2.addWeighted(overlay, alpha, mask_3d, 1 - alpha, 0)
        return overlay

    def update_magnifier(self, event):
        if event.inaxes in [self.ax[0, 0], self.ax[0, 1], self.ax[0, 2]]:
            x, y = int(event.xdata), int(event.ydata)
            zoom = self.zoom_level  # Use calculated zoom level
            x0, x1 = max(0, x - zoom), min(self.image_rgb.shape[1], x + zoom)
            y0, y1 = max(0, y - zoom), min(self.image_rgb.shape[0], y + zoom)

            if event.inaxes == self.ax[0, 0]:
                self.magnifier_img.set_data(self.image_rgb[y0:y1, x0:x1])
                self.ax[1, 0].cla()
                self.ax[1, 0].imshow(self.image_rgb[y0:y1, x0:x1])
                self.ax[1, 0].axvline(x - x0, color='red')
                self.ax[1, 0].axhline(y - y0, color='red')
            elif event.inaxes == self.ax[0, 1]:
                self.magnifier_mask.set_data(self.mask[y0:y1, x0:x1])
                self.ax[1, 1].cla()
                self.ax[1, 1].imshow(self.mask[y0:y1, x0:x1], cmap='gray')
                self.ax[1, 1].axvline(x - x0, color='red')
                self.ax[1, 1].axhline(y - y0, color='red')
            elif event.inaxes == self.ax[0, 2]:
                self.magnifier_overlay.set_data(self.overlay_image()[y0:y1, x0:x1])
                self.ax[1, 2].cla()
                self.ax[1, 2].imshow(self.overlay_image()[y0:y1, x0:x1])
                self.ax[1, 2].axvline(x - x0, color='red')
                self.ax[1, 2].axhline(y - y0, color='red')

            self.fig.canvas.draw_idle()
            # Remove x and y axes labels
            for ax in self.ax.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

    def prev_slice(self, event):
        self.current_index = (self.current_index - 1) % len(self.images)
        self.load_slice()

    def next_slice(self, event):
        self.current_index = (self.current_index + 1) % len(self.images)
        self.load_slice()

    def save_current_mask(self, event):
        mask_path = os.path.join(MASKS_DIR, f'{self.image_files[self.current_index]}')
        cv2.imwrite(mask_path, self.mask)

        masked_image = cv2.bitwise_and(self.image_rgb, self.image_rgb, mask=self.mask)
        masked_image_path = os.path.join(MASKED_DIR, f'{self.image_files[self.current_index]}')
        cv2.imwrite(masked_image_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

        if self.debug:
            print(f"Saved mask and masked image for {self.image_files[self.current_index]}")

    def load_slice(self):
        self.image_rgb = self.images[self.current_index]
        self.original_mask = self.masks[self.current_index].copy()
        self.mask = self.masks[self.current_index]
        self.history = []
        self.overlay_color = determine_best_overlay_color(self.image_rgb)

        self.ax[0, 0].imshow(self.image_rgb)
        self.mask_display.set_data(self.mask)
        self.overlay_display.set_data(self.overlay_image())
        self.filename_text.set_text(self.image_files[self.current_index])
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

def segment_brain_slice(image_rgb, debug=False):
    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding to get a binary image
    adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # Find the largest connected component (the brain slice)
    labels = measure.label(cleaned_mask)
    properties = measure.regionprops(labels)
    if not properties:
        raise RuntimeError("No regions found")
    largest_region = max(properties, key=lambda x: x.area)
    
    # Create a mask for the largest connected component
    brain_slice_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
    brain_slice_mask[labels == largest_region.label] = 255
    
    # Use flood fill to fill internal structures
    flood_fill_mask = brain_slice_mask.copy()
    h, w = flood_fill_mask.shape
    mask_flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_fill_mask, mask_flood, (0, 0), 255)
    
    # Invert flood-filled mask
    flood_fill_mask = cv2.bitwise_not(flood_fill_mask)
    
    # Combine the original mask with the flood-filled mask
    combined_mask = cv2.bitwise_or(brain_slice_mask, flood_fill_mask)
    
    # Select only the largest connected component from the combined mask
    final_labels = measure.label(combined_mask)
    final_properties = measure.regionprops(final_labels)
    if not final_properties:
        raise RuntimeError("No regions found")
    largest_final_region = max(final_properties, key=lambda x: x.area)
    final_mask = np.zeros_like(combined_mask, dtype=np.uint8)
    final_mask[final_labels == largest_final_region.label] = 255
    
    return final_mask

def load_images_and_masks(input_dir):
    image_files = [f for f in os.listdir(input_dir) if re.match(FILE_PATTERN, f)]
    image_files.sort()
    images = []
    masks = []
    for file in image_files:
        image_path = os.path.join(input_dir, file)
        image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        images.append(image_rgb)
        mask = segment_brain_slice(image_rgb)
        masks.append(mask)
    return images, masks, image_files

def process_directory(input_dir, start_image=None, debug=False):
    images, masks, image_files = load_images_and_masks(input_dir)
    if start_image and start_image in image_files:
        start_index = image_files.index(start_image)
    else:
        start_index = 0
    editor = MaskEditor(images, masks, image_files, current_index=start_index, initial_tolerance=INITIAL_TOLERANCE, debug=debug)
    editor.show()

if __name__ == "__main__":
    start_image = None
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'debug':
            DEBUG_MODE = True
        elif sys.argv[1].lower().startswith('slice'):
            start_image = sys.argv[1]
    process_directory(BASE_DIRECTORY, start_image=start_image, debug=DEBUG_MODE)
