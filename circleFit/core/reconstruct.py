"""Orchestrate the circle reconstruction."""
import cv2

# Relative imports 
from .extract_points import extract_arc_points
from .fit_circle import fit_circle_to_points
from .draw_dashed import draw_dashed_circle

def reconstruct_circle_from_image(image_path):
    """Process a single image and reconstruct the complete circle."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    arc_points = extract_arc_points(image)
    if arc_points is None:
        return None
    
    try:
        center_x, center_y, radius = fit_circle_to_points(arc_points)
    except Exception:
        return None
    
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
