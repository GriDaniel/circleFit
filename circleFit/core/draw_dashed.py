"""Utility functions for drawing the complete circle(s)."""
import cv2
import numpy as np

def draw_dashed_circle(image, center_x, center_y, radius, color=(6, 6, 149), thickness=2):
    """Draw a dashed circle on the image."""
    circumference = 2 * np.pi * radius
    num_segments = int(circumference / 20)  # 20-pixel dashes
    
    for i in range(0, num_segments, 2):  # Skip every other segment for dash effect
        angle1 = (i / num_segments) * 2 * np.pi
        angle2 = ((i + 1) / num_segments) * 2 * np.pi
        
        point1 = (int(center_x + radius * np.cos(angle1)), 
                 int(center_y + radius * np.sin(angle1)))
        point2 = (int(center_x + radius * np.cos(angle2)), 
                 int(center_y + radius * np.sin(angle2)))
        
        cv2.line(image, point1, point2, color, thickness)
