# FILE: ./circleFit/core/extract_points.py (with temporary modifications)
import cv2

def extract_arc_points(image):
    """Extract arc points from image using edge detection."""
    # ... (existing code)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --- ADD THIS FOR DEBUGGING ---
    cv2.imwrite("DEBUG_01_gray.png", gray)
    # -----------------------------
    
    # Detect edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # --- ADD THIS FOR DEBUGGING ---
    cv2.imwrite("DEBUG_02_blurred.png", blurred)
    # -----------------------------

    edges = cv2.Canny(blurred, 50, 150)

    # --- ADD THIS FOR DEBUGGING ---
    cv2.imwrite("DEBUG_03_edges.png", edges)
    # -----------------------------
    
    # Find contours and select the largest one
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # ... (rest of the code)
