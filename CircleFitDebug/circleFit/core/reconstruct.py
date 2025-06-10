"""Orchestrate the circle reconstruction."""
import cv2
import traceback

# Relative imports 
from .extract_points import extract_arc_points
from .fit_circle import fit_circle_to_points
from .draw_dashed import draw_dashed_circle

def reconstruct_circle_from_image(image_path):
    """Process a single image and reconstruct the complete circle."""
    # --- Step 1: Image Loading ---
    image = cv2.imread(str(image_path))
    if image is None:
        # DEBUG: This is the first potential point of failure.
        print("  [DEBUG] FAILED at Step 1: Image Loading. File might be missing, corrupted, or not a valid image format.")
        return None
    
    # --- Step 2: Point Extraction ---
    arc_points = extract_arc_points(image)
    if arc_points is None:
        # DEBUG: The specific reason for failure is printed from within extract_arc_points.
        print("  [DEBUG] FAILED at Step 2: Point Extraction.")
        return None
    
    # --- Step 3: Circle Fitting ---
    try:
        center_x, center_y, radius = fit_circle_to_points(arc_points)
    except Exception as e:
        # DEBUG: This is the third potential point of failure.
        print(f"  [DEBUG] FAILED at Step 3: Circle Fitting. The algorithm could not proceed.")
        print(f"  [DEBUG]   Reason: {e}")
        # To see the full error stack for deep debugging, uncomment the next line:
        # traceback.print_exc()
        return None
    
    # --- Success: If we reach here, all steps passed ---
    output_image = image.copy()
    draw_dashed_circle(output_image, center_x, center_y, radius)
    cv2.circle(output_image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
    
    for point in arc_points[::5]:  # Sample every 5th point
        cv2.circle(output_image, tuple(point), 2, (0, 0, 0), -1)
    
    return {
        'image': output_image,
        'center': (center_x, center_y),
        'radius': radius
    }