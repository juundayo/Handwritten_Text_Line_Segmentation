import logging
import os
import time
import cv2
import argparse
import numpy as np
import random

import src.line_segmentation.preprocessing.energy_map
from src.line_segmentation.bin_algorithm import majority_voting, draw_bins
from src.line_segmentation.polygon_manager import polygon_to_string, get_polygons_from_lines, draw_polygons
from src.line_segmentation.preprocessing.load_image import prepare_image, load_image
from src.line_segmentation.preprocessing.preprocess import preprocess
from src.line_segmentation.seamcarving_algorithm import draw_seams, get_seams, post_process_seams, draw_seams_red, draw_colored_seams
from src.line_segmentation.utils.XMLhandler import writePAGEfile
from src.line_segmentation.utils.graph_logger import GraphLogger
from src.line_segmentation.utils.util import create_folder_structure, save_img

# ----------------------------------------------------------------------------#

IMG_PATH = "/home/ml3/Desktop/Thesis/.venv/NewImplementation/Images/035_4.tif"

# ----------------------------------------------------------------------------#

def extract_textline(input_path: str, output_path: str, penalty_reduction: int, seam_every_x_pxl: int,
                    vertical: bool, console_log: bool, small_component_ratio: float, expected_lines: int = None):
    """
    Function to compute the text lines from a segmented image. 
    """

    # -------------------------------
    start_whole = time.time()
    # -------------------------------

    # Image loading
    img = load_image(input_path)
    original_img4 = img

    # Creating the folders and getting the new root folder.
    root_output_path = create_folder_structure(input_path, output_path, (penalty_reduction, seam_every_x_pxl, small_component_ratio))

    # Init the logger with the logging path.
    init_logger(root_output_path, console_log)

    # Init the graph logger.
    GraphLogger.IMG_SHAPE = img.shape
    GraphLogger.ROOT_OUTPUT_PATH = root_output_path

    # Image preparation.
    img = prepare_image(img, cropping=False, vertical=vertical)
    save_img(img, path=os.path.join(root_output_path, 'preprocess', 'original.png'))

    # Image pre-processing.
    img = preprocess(img, small_component_ratio)
    save_img(img, path=os.path.join(root_output_path, 'preprocess', 'after_preprocessing.png'))
    
    original_img1 = img
    original_img2 = img
    original_img3 = img

    # Energy map creation.
    energy_map, connected_components = src.line_segmentation.preprocessing.energy_map.create_energy_map(img,
                                                                                      blurring=False,
                                                                                      projection=False,
                                                                                      asymmetric=True)
    # Visualizing the energy map as heatmap.
    heatmap = src.line_segmentation.preprocessing.energy_map.create_heat_map_visualization(energy_map)
    save_img(heatmap, path=os.path.join(root_output_path, 'energy_map', 'energy_map_without_seams.png'))

    # Getting the seams and drawing them on the heatmap.
    seams = get_seams(energy_map, penalty_reduction, seam_every_x_pxl, expected_lines)
    draw_seams(heatmap, seams)
    save_img(heatmap, path=os.path.join(root_output_path, 'energy_map', 'energy_map_with_seams.png'))

    # Seams post-processing and drawing them on the heatmap.
    seams = post_process_seams(energy_map, seams, expected_lines)
    draw_seams_red(heatmap, seams)
    save_img(heatmap, path=os.path.join(root_output_path, 'energy_map', 'energy_map_postprocessed_seams.png'))

    colored_img = draw_colored_seams(original_img4.copy(), seams)
    cv2.imwrite(os.path.join(root_output_path, "colored_lines.png"), colored_img)

    # Bin extraction.
    lines, centroids, values = majority_voting(connected_components, seams)

    # Draw the bins on a white gray image of the text with red seams.
    draw_bins(original_img3, centroids, root_output_path, seams, values)

    # Getting polygons from the lines and drawing them on the original image.
    polygons = get_polygons_from_lines(original_img3, lines, connected_components, vertical)
    save_img(draw_polygons(img.copy(), polygons, vertical), path=os.path.join(root_output_path, 'polygons_on_text.png'))

    # Writing the results on the XML file.
    writePAGEfile(os.path.join(root_output_path, 'polygons.xml'), polygon_to_string(polygons))

    # -------------------------------
    stop_whole = time.time()
    logging.info("finished after: {diff} s".format(diff=stop_whole - start_whole))
    # -------------------------------

    return

# ----------------------------------------------------------------------------#

def colorize_lines_from_components(original_img, lines, connected_components, save_path):
    """
    Colorize each detected text line in the image with a distinct random color.

    Args:
        original_img (np.ndarray): The original grayscale or binary image.
        lines (list): List of lines (from majority_voting).
        connected_components (list): Connected components belonging to each line.
        save_path (str): Output path to save the colorized image.
    """
    # Convert grayscale to BGR (so we can add colors)
    label_image, regionprops_list = connected_components  

    # Convert grayscale to BGR
    if len(original_img.shape) == 2:
        color_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        color_img = original_img.copy()

    # Start with a white background
    color_img[:] = (255, 255, 255)

    # Assign a random color per line
    line_colors = {
        line_id: (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        for line_id in range(len(lines))
    }

    # Paint components belonging to each line
    for line_id, comps in enumerate(lines):
        for comp in comps:
            # Ensure comp_id is an integer (sometimes it may be a tuple or list)
            if isinstance(comp, (tuple, list, np.ndarray)):
                comp_id = int(comp[0])
            else:
                comp_id = int(comp)

            mask = (label_image == comp_id).astype(np.uint8) * 255
            color = line_colors[line_id]
            color_layer = np.zeros_like(color_img, dtype=np.uint8)
            color_layer[:] = color
            color_img = np.where(mask[..., None] == 255, color_layer, color_img)

    # Save
    cv2.imwrite(save_path, color_img)
    print(f"Saved colorized textlines to {save_path}")

# ----------------------------------------------------------------------------#

def save_line_mask_with_overlay(img, lines, mask_path, overlay_path, alpha=0.6, dtype=np.uint16):
    """
    img:         original image (H, W, 3 or 1)
    lines:       list of lines, each line is a list of np.array([x, y])
    mask_path:   path to save raw label mask
    overlay_path:path to save color-coded overlay image
    alpha:       blending factor for text pixels
    dtype:       np.uint8 or np.uint16
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=dtype)
    overlay = img.copy()

    random.seed(42)  # Optional: consistent colors across runs
    label_colors = {}

    for label, line in enumerate(lines, start=1):
        color = tuple(random.randint(50, 255) for _ in range(3))
        label_colors[label] = color

        for coord in line:
            if isinstance(coord, np.ndarray) and coord.shape == (2,):
                x, y = map(int, coord)
                if 0 <= y < h and 0 <= x < w:
                    mask[y, x] = label
                    orig_pixel = overlay[y, x]
                    blended_pixel = (1 - alpha) * orig_pixel + alpha * np.array(color)
                    overlay[y, x] = blended_pixel.astype(np.uint8)

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay)

# ----------------------------------------------------------------------------#

def init_logger(root_output_path, console_log):
    # Creating a logging format.
    formatter = logging.Formatter(fmt='%(asctime)s %(filename)s:%(funcName)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    # Getting the logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Creating and adding a file handler.
    handler = logging.FileHandler(os.path.join(root_output_path, 'logs', 'extract_textline.log'))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if console_log:
        # Creating and adding a stderr handler.
        stderr_handler = logging.StreamHandler()
        stderr_handler.formatter = formatter
        logger.addHandler(stderr_handler)

# ----------------------------------------------------------------------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default=IMG_PATH, help='Path to the input file')
    parser.add_argument('--output-path', type=str, default='./output', help='Path to the output folder')
    parser.add_argument('--seam-every-x-pxl', type=int, default=100, help='After how many pixel a new seam should be'
                                                                          ' casted')
    parser.add_argument('--penalty_reduction', type=int, default=6000, help='Punishment reduction for the seam'
                                                                            ' leaving the y axis')
    parser.add_argument('--small_component_ratio', type=float, default=0.1, help='Ratio of the small components')
    parser.add_argument('--console_log', action='store_false', help='Deactivate console logging')
    parser.add_argument('--vertical', action='store_true', help='Is the text orientation vertical?')
    parser.add_argument('--expected_lines', type=int, default=None, nargs='?', help='Expected number of text lines (optional)')

    args = parser.parse_args()

    extract_textline(**args.__dict__)

    logging.info('Terminated')
