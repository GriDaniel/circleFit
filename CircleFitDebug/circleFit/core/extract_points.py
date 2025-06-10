"""Extract points from an image."""
import cv2

def extract_arc_points(image):
    """Extract arc points from image using edge detection."""
    # Step 1: Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # --- VISUAL DEBUG STEP 1: Save the grayscale image ---
    # This shows you the starting point after removing color.
    # Look for low contrast between the arc and the background.
    cv2.imwrite("DEBUG_01_gray.png", gray)
    # ----------------------------------------------------

    # Step 2: Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # --- VISUAL DEBUG STEP 2: Save the blurred image ---
    # This shows you what the Canny detector will work on.
    # Check if a very thin arc has been "washed out" or erased by the blur.
    cv2.imwrite("DEBUG_02_blurred.png", blurred)
    # -------------------------------------------------

    # Step 3: Detect edges with the Canny algorithm
    edges = cv2.Canny(blurred, 50, 150)
    
    # --- VISUAL DEBUG STEP 3: Save the edge map ---
    # THIS IS THE MOST IMPORTANT DEBUG IMAGE.
    # If this image is all black, the next step will fail.
    # It tells you that the Canny parameters (50, 150) are not suitable for this image.
    cv2.imwrite("DEBUG_03_edges.png", edges)
    # --------------------------------------------

    # Step 4: Find contours (continuous shapes) in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        # DEBUG: Explain why no contours were found.
        print("  [DEBUG]   Reason: No contours were found. This is because the Canny edge map (DEBUG_03_edges.png) was likely empty. Try adjusting contrast or Canny thresholds.")
        return None
    
    # Step 5: Select the largest contour found
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.reshape(-1, 2)
    
    # Step 6: Filter out contours that are too small to be a valid arc
    if len(points) < 10:
        # DEBUG: Explain why the found contour was rejected.
        print(f"  [DEBUG]   Reason: The largest contour found only has {len(points)} points, which is below the threshold of 10. This is likely just noise.")
        return None
    
    return points
