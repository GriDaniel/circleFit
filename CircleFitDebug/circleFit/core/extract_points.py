"""Extract points from an image."""
import cv2

def extract_arc_points(image):
    """Extract arc points from image using edge detection."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Detect edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours and select the largest one
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        # DEBUG: Explain why no contours were found.
        print("  [DEBUG]   Reason: No contours were found. This can happen if the image is blank, has no contrast, or Canny edge detection settings are unsuitable.")
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.reshape(-1, 2)
    
    # Filter out noise
    if len(points) < 10:
        # DEBUG: Explain why the found contour was rejected.
        print(f"  [DEBUG]   Reason: The largest contour found only has {len(points)} points, which is below the threshold of 10. This is likely noise.")
        return None
    
    return points