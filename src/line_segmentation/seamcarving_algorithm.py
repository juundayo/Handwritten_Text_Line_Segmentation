import itertools
import sys

import cv2
import numba
import numpy as np
import random

import src.line_segmentation

# ----------------------------------------------------------------------------#

"""
    Code from: https://github.com/danasilver/seam-carving/blob/master/seamcarve.py
    homepage: http://www.faculty.idc.ac.il/arik/SCWeb/imret/
"""

# ----------------------------------------------------------------------------#

@numba.jit()
def horizontal_seam(energies, penalty_reduction, bidirectional=False):
    """
    Spawns seams from the left to the right or from both directions. It returns the list of seams as point list.

    :param energies: the energy map
    :param penalty_reduction: if the penalty_reduction is smaller or equal to 0 we wont apply a penalty reduction
    :param bidirectional: if True there will be seams from left to right and right to left, else just from left to right
    :return: seams as point list
    """
    height, width = energies.shape[:2]
    ori_y = 0     # The y position we started (needed for the penalty).
    previous = 0  # The last point we visit.
    
    # The points of the seam:
    seam_forward = [] # From left to right
    seam_backward = [] # From right to left.

    # Spawns seams from left to right.
    for i in range(0, width, 1):
        col = energies[:, i]
        if i == 0:
            ori_y = previous = np.argmin(col)
        else:
            top = col[previous - 1] if previous - 1 >= 0 else sys.maxsize
            middle = col[previous]
            bottom = col[previous + 1] if previous + 1 < height else sys.maxsize

            if penalty_reduction > 0:
                top += ((ori_y - (previous - 1)) ** 2) / penalty_reduction
                middle += ((ori_y - previous) ** 2) / penalty_reduction
                bottom += + ((ori_y - (previous + 1)) ** 2) / penalty_reduction

            previous = previous + np.argmin(np.array([top, middle, bottom])) - 1

        seam_forward.append([i, previous])

    # Spawns seams from right to left.
    if bidirectional:
        for i in range(width-1, -1, -1):
            col = energies[:, i]
            if i == width-1:
                ori_y = previous = np.argmin(col)
            else:
                top = col[previous - 1] if previous - 1 >= 0 else sys.maxsize
                middle = col[previous]
                bottom = col[previous + 1] if previous + 1 < height else sys.maxsize

                if penalty_reduction > 0:
                    top += ((ori_y - (previous - 1)) ** 2) / penalty_reduction
                    middle += ((ori_y - previous) ** 2) / penalty_reduction
                    bottom += + ((ori_y - (previous + 1)) ** 2) / penalty_reduction

                previous = previous + np.argmin(np.array([top, middle, bottom])) - 1

            seam_backward.append([i, previous])

    return [seam_forward, seam_backward[::-1]]

# ----------------------------------------------------------------------------#

def draw_seams(img, seams, bidirectional=True):

    x_axis = np.expand_dims(np.array(range(0, len(seams[0]))), -1)
    seams = [np.concatenate((x, np.expand_dims(seam, -1)), 
                            axis=1) for seam, x in zip(seams, itertools.repeat(x_axis))]

    for i, seam in enumerate(seams):
        # Get the seam from the left [0] and the seam from the right[1].
        if bidirectional and i % 2 == 0:
            cv2.polylines(img, np.int32([seam]), False, (0, 0, 0), 3) # Black.
        else:
            cv2.polylines(img, np.int32([seam]), False, (255, 255, 255), 3) # White.

# ----------------------------------------------------------------------------#

def draw_colored_seams(original_img, seams):
    """
    Colorizes entire text content line by line using the seam boundaries.

    Args:
        original_img (np.ndarray): Grayscale or BGR input image.
        seams (list): List of seam polylines.
        thickness (int): Used to smooth seam masks if needed.
    Returns:
        np.ndarray: Colorized image with different colors per text line.
    """
    # Ensuring the image is grayscale.
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_img.copy()

    # Inverting so text is 1, background 0.
    binary = (gray < 128).astype(np.uint8)

    # Converting base to white background.
    h, w = gray.shape
    color_img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Converting seams into usable polylines.
    x_axis = np.expand_dims(np.arange(len(seams[0])), -1)
    seams = [
        np.concatenate((x, np.expand_dims(seam, -1)), axis=1)
        for seam, x in zip(seams, itertools.repeat(x_axis))
    ]

    # Sorting seams from top to bottom.
    seams_sorted = sorted(seams, key=lambda s: np.mean(s[:, 1]))

    # Adding top and bottom boundaries for masking.
    seam_boundaries = [np.vstack([[0, 0], [w-1, 0]])] + seams_sorted + [np.vstack([[0, h-1], [w-1, h-1]])]

    # Looping over line regions between seams.
    for line_id in range(len(seam_boundaries)-1):
        upper = seam_boundaries[line_id]
        lower = seam_boundaries[line_id+1]

        # Building polygon between upper and lower seams.
        poly = np.vstack([upper, np.flipud(lower)])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

        # Random color for this line.
        colours = [
            (255, 182, 193),  # Light Pink
            (173, 216, 230),  # Light Blue
            (152, 251, 152),  # Pale Green
            (255, 239, 213),  # Papaya Whip
            (221, 160, 221),  # Plum
            (240, 230, 140),  # Khaki (Soft Yellow)
            (176, 224, 230),  # Powder Blue
            (255, 218, 185),  # Peach Puff
            (230, 230, 250),  # Lavender
            (245, 222, 179),  # Wheat
            (175, 238, 238),  # Pale Turquoise
            (255, 228, 225),  # Misty Rose
            (240, 248, 255),  # Alice Blue
            (245, 245, 220),  # Beige
        ]
    
        # Use predefined pastel colors, cycle through if more lines
        colour = colours[line_id % len(colours)]

        # Applying color only where there is text.
        line_mask = cv2.bitwise_and(binary, binary, mask=mask)
        color_layer = np.zeros_like(color_img, dtype=np.uint8)
        color_layer[:] = colour
        color_img = np.where(line_mask[..., None] == 1, color_layer, color_img)

    return color_img

# ----------------------------------------------------------------------------#

def encode_lines_bitmap(original_img, seams):
    """
    Encodes line contents using bit encoding where:
    0 = background
    1-n = line value for each part of the text (n lines)
    
    Args:
        original_img (np.ndarray): Grayscale or BGR input image.
        seams (list): List of seam polylines.
    
    Returns:
        np.ndarray: Bit-encoded image with values 0-n for background and lines.
    """
    # Ensuring the image is grayscale.
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_img.copy()

    # Inverting so text is 1, background 0.
    binary = (gray < 128).astype(np.uint8)

    # Converting seams into usable polylines.
    h, w = gray.shape
    x_axis = np.expand_dims(np.arange(len(seams[0])), -1)
    seams = [
        np.concatenate((x, np.expand_dims(seam, -1)), axis=1)
        for seam, x in zip(seams, itertools.repeat(x_axis))
    ]

    # Sorting seams from top to bottom.
    seams_sorted = sorted(seams, key=lambda s: np.mean(s[:, 1]))

    # Adding top and bottom boundaries for masking.
    seam_boundaries = [np.vstack([[0, 0], [w-1, 0]])] + seams_sorted + [np.vstack([[0, h-1], [w-1, h-1]])]

    # Creatting the output bitmap initialized with zeros (background).
    encoded_img = np.zeros((h, w), dtype=np.uint8)

    # Processing each line region.
    for line_id in range(len(seam_boundaries)-1):
        upper = seam_boundaries[line_id]
        lower = seam_boundaries[line_id+1]

        # Building polygon between upper and lower seams.
        poly = np.vstack([upper, np.flipud(lower)])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        # Getting text pixels within this line region.
        line_text_pixels = cv2.bitwise_and(binary, binary, mask=mask)
        
        # Assigning line value (1-indexed) to text pixels in this region.
        encoded_img[line_text_pixels == 1] = line_id + 1

    return encoded_img

# ----------------------------------------------------------------------------#

def draw_seams_red(img, seams, bidirectional=True):
    x_axis = np.expand_dims(np.array(range(0, len(seams[0]))), -1)
    seams = [np.concatenate((x, np.expand_dims(seam, -1)), 
                            axis=1) for seam, x in zip(seams, itertools.repeat(x_axis))]

    for i, seam in enumerate(seams):
        # Getting the seam from the left [0] and the seam from the right[1].
            cv2.polylines(img, np.int32([seam]), False, (0, 0, 255), 3)  # Red!

# ----------------------------------------------------------------------------#

def get_seams(ori_energy_map, penalty_reduction, seam_every_x_pxl, expected_lines=None):
    seams = [] # List with all seams
    left_column_energy_map = np.copy(ori_energy_map[:, 0])   # Left most column of the energy map.
    right_column_energy_map = np.copy(ori_energy_map[:, -1]) # Right most column of the energy map.
    # show_img(ori_enegery_map)

    for seam_at in range(0, ori_energy_map.shape[0], seam_every_x_pxl):
        energy_map = src.line_segmentation.preprocessing.energy_map.prepare_energy(ori_energy_map,
                                                                                   left_column_energy_map,
                                                                                   right_column_energy_map, seam_at)

        seams.extend(horizontal_seam(energy_map, penalty_reduction=penalty_reduction, bidirectional=True))

    # Strip seams of x coordinate, which is totally useless as  
    # the x coordinate is basically the index in the array.
    seams = np.array([np.array(s)[:, 1] for s in seams])

    if expected_lines is not None:
        seams = adjust_seam_count(seams, expected_lines, ori_energy_map.shape[0])

    return seams

# ----------------------------------------------------------------------------#

def adjust_seam_count(seams, expected_lines, image_height):
    """
    Adjust the number of seams to match the expected_lines parameter.
    
    Args:
        seams: Array of seams
        expected_lines: Desired number of lines/seams
        image_height: Height of the image
    
    Returns:
        Adjusted array of seams
    """
    current_count = len(seams)

    if current_count == expected_lines:
        return seams
    
    # Sorting seams by their average y-position.
    sorted_indices = np.argsort([np.mean(seam) for seam in seams])
    sorted_seams = seams[sorted_indices]

    if current_count > expected_lines:
        # Removing excess seams with highest energy or most irregular paths.
        # This is the most common case.
        return remove_excess_seams(sorted_seams, expected_lines)
    else:
        # Adding missing seams (interpolating between existing seams).
        # This is a very rare case, which might never happen.
        return add_missing_seams(sorted_seams, expected_lines, image_height)

# ----------------------------------------------------------------------------#

def remove_excess_seams(seams, target_count):
    """
    Remove excess seams to reach the target count.
    Prioritize removing seams that are closest 
    to each other or have highest energy.
    """
    if len(seams) <= target_count:
        return seams
    
    # Calculates distances between adjacent seams.
    distances = []
    for i in range(len(seams) - 1):
        avg_distance = np.mean(np.abs(seams[i] - seams[i + 1]))
        distances.append(avg_distance)
    
    # Finding pairs with smallest distances (most likely to be redundant).
    smallest_dist_indices = np.argsort(distances)[:len(seams) - target_count]

    # Creating a mask to keep seams.
    keep_mask = np.ones(len(seams), dtype=bool)
    for idx in smallest_dist_indices:
        keep_mask[idx + 1] = False  # Remove the second seam in close pairs.
    
    return seams[keep_mask]  

# ----------------------------------------------------------------------------#

def add_missing_seams(seams, target_count, image_height):
    """
    Adds missing seams by interpolating between existing seams.
    """
    if len(seams) >= target_count:
        return seams
    
    additional_seams_needed = target_count - len(seams)
    new_seams = []
    
    # Adding top and bottom boundaries if not already present.
    all_seams = list(seams)
    
    # Interpolating between existing seams.
    for i in range(len(all_seams) - 1):
        if len(new_seams) >= additional_seams_needed:
            break
            
        # Creating interpolated seam between current and next seam
        interpolated_seam = (all_seams[i] + all_seams[i + 1]) // 2
        new_seams.append(interpolated_seam)
    
    # If we still need more seams, create them at the boundaries.
    while len(new_seams) < additional_seams_needed:
        # Adding a seam at the top (average of top seam and top boundary).
        top_seam = (all_seams[0] + np.zeros_like(all_seams[0])) // 2
        new_seams.insert(0, top_seam)
        
        if len(new_seams) < additional_seams_needed:
            # Adding a seam at the bottom (average of bottom seam and bottom boundary).
            bottom_seam = (all_seams[-1] + np.full_like(all_seams[-1], image_height - 1)) // 2
            new_seams.append(bottom_seam)
    
    # Combining and sorting all seams.
    all_seams.extend(new_seams[:additional_seams_needed])
    sorted_indices = np.argsort([np.mean(seam) for seam in all_seams])
    
    return np.array(all_seams)[sorted_indices]

# ----------------------------------------------------------------------------#

def post_process_seams(energy_map, seams, expected_lines=None):
    # Check that the seams are as wide as the image.
    assert energy_map.shape[1] == len(seams[0])

    # Adjust seam count if expected_lines is specified.
    if expected_lines is not None and len(seams) != expected_lines:
        seams = adjust_seam_count(seams, expected_lines, energy_map.shape[0])

    # TODO implement a tabu-list to prevent two seams to repeatedly swap a third seam between them
    SAFETY_STOP = 100
    iteration = 0
    repeat = True
    while repeat:

        # Safety exit in case of endless loop meeting condition. See above.
        iteration += 1
        if iteration >= SAFETY_STOP:
            break

        repeat = False
        for index, seam_A in enumerate(seams):
            for seam_B in seams[index:]:
                # Compute seams overlap.
                overlap = seam_A - seam_B

                # Smooth the overlap.
                overlap[abs(overlap) < 10] = 0

                # Make the two seams really overlap.
                seam_A[overlap == 0] = seam_B[overlap == 0]

                # Find non-zero sequences.
                sequences = non_zero_runs(overlap)

                if len(sequences) > 0:
                    for i, sequence in enumerate(sequences):

                        target = sequence[1] - sequence[0]

                        left = sequence[0] - sequences[i - 1, 1] if i > 0 else sequence[0]
                        right = sequences[i + 1, 0] - sequence[1] if i < len(sequences)-1 else energy_map.shape[1] - sequence[1]

                        if target > left and target > right:
                            continue

                        repeat = True

                        # Expand the sequence into a range.
                        sequence = range(*sequence)
                        # Compute the seam.
                        energy_A = measure_energy(energy_map, seam_A, sequence)
                        energy_B = measure_energy(energy_map, seam_B, sequence)

                        # Remove the weaker seam sequence.
                        if energy_A > energy_B:
                            seam_A[sequence] = seam_B[sequence]
                        else:
                            seam_B[sequence] = seam_A[sequence]

    return seams

# ----------------------------------------------------------------------------#

def non_zero_runs(a):
    """
    Finding the consecutive non-zeros in a numpy array. Modified from:
    https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    """
    # Creating an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([1], np.equal(a, 0).view(np.int8), [1]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

# ----------------------------------------------------------------------------#

def measure_energy(energy_map, seam, sequence):
    """
    Compute the energy of that seams for the specified range
    """
    return energy_map[seam[sequence], sequence].sum()
