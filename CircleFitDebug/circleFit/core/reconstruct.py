"""Orchestrate the circle reconstruction."""
import cv2
import numpy as np
import traceback
from pathlib import Path

# Relative imports 
from .extract_points import extract_arc_points
from .fit_circle import fit_circle_to_points
from .draw_dashed import draw_dashed_circle

def reconstruct_circle_from_image(image_path, base_folder):
    """
    Processes a single image, with logic to handle transparency and expand the canvas,
    without saving intermediate synthetic files.
    """
    # --- Step 1: Load Image and Prepare ---
    original_image_with_alpha = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if original_image_with_alpha is None:
        print("  [DEBUG] FAILED: Could not load image file.")
        return None

    original_image_bgr = cv2.imread(str(image_path))
    image_to_process = None
    base_for_drawing = None
    comparison_image = None

    # --- Step 2: Handle Transparency and Prepare for Processing ---
    if len(original_image_with_alpha.shape) == 3 and original_image_with_alpha.shape[2] == 4:
        print("  [INFO] Alpha channel detected. Creating a standardized image in memory.")
        
        h, w = original_image_with_alpha.shape[:2]
        canvas = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
        alpha_channel = original_image_with_alpha[:, :, 3]
        bgr_channels = original_image_with_alpha[:, :, :3]
        canvas[alpha_channel > 0] = bgr_channels[alpha_channel > 0]
        
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        # Create the standard black-on-white image for processing
        _, binary_image = cv2.threshold(gray_canvas, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        image_to_process = binary_image
        # The base for drawing is the black-on-white version
        base_for_drawing = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        # The comparison image is the INVERSE: a white arc on a black background
        comparison_image = cv2.bitwise_not(base_for_drawing)
    else:
        print("  [INFO] No alpha channel. Standardizing image in memory.")
        gray_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Create the standard black-on-white image for processing
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        image_to_process = binary_image
        # The base for drawing is the black-on-white version
        base_for_drawing = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        # The comparison image is the INVERSE: a white arc on a black background
        comparison_image = cv2.bitwise_not(base_for_drawing)

    # --- Step 3: Point Extraction ---
    arc_points = extract_arc_points(image_to_process)
    if arc_points is None:
        print("  [DEBUG] FAILED: Could not extract points from the image.")
        return None

    # --- Step 4: Circle Fitting ---
    try:
        center_x, center_y, radius = fit_circle_to_points(arc_points)
    except Exception as e:
        print(f"  [DEBUG] FAILED: Circle fitting algorithm failed. Reason: {e}")
        return None

    # --- Step 5: Canvas Expansion Logic ---
    h, w = base_for_drawing.shape[:2]
    min_x, max_x = center_x - radius, center_x + radius
    min_y, max_y = center_y - radius, center_y + radius
    
    pad_left = int(max(0, -min_x))
    pad_top = int(max(0, -min_y))
    pad_right = int(max(0, max_x - w))
    pad_bottom = int(max(0, max_y - h))
    
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        print("  [INFO] Circle extends beyond viewport. Expanding canvas...")
        new_h = h + pad_top + pad_bottom
        new_w = w + pad_left + pad_right
        
        expanded_canvas = np.full((new_h, new_w, 3), (0, 0, 0), dtype=np.uint8)
        expanded_canvas[pad_top:pad_top+h, pad_left:pad_left+w] = base_for_drawing
        
        base_for_drawing = expanded_canvas
        center_x += pad_left
        center_y += pad_top
        arc_points += np.array([pad_left, pad_top])

    # --- Step 6: Final Drawing ---
    output_image = base_for_drawing.copy()
    draw_dashed_circle(output_image, center_x, center_y, radius)
    cv2.circle(output_image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
    for point in arc_points[::5]:
        cv2.circle(output_image, tuple(point), 2, (255, 255, 255), -1)

    # --- Step 7: Create Side-by-Side Image with NO WHITE SPACE ---
    reconstructed_side = output_image
    h1, w1 = reconstructed_side.shape[:2]
    
    h2, w2 = comparison_image.shape[:2]

    mega_h = max(h1, h2)
    mega_w = w1 + w2
    mega_image = np.full((mega_h, mega_w, 3), (0, 0, 0), dtype=np.uint8)

    mega_image[0:h1, 0:w1] = reconstructed_side
    
    # Pad the comparison image if it's shorter
    if h1 > h2:
        padding = np.zeros((h1 - h2, w2, 3), dtype=np.uint8)
        comparison_image = cv2.vconcat([comparison_image, padding])

    mega_image[0:mega_h, w1:w1+w2] = comparison_image
    
    return {
        'image': mega_image,
        'center': (center_x, center_y),
        'radius': radius
    }