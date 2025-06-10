"""Circle fitting algorithm."""
import numpy as np
from scipy.optimize import leastsq

def fit_circle_to_points(points):
    """Fit a circle to points using least squares optimization."""
    # DEBUG: Add pre-condition checks for valid input data.
    # A circle requires at least 3 points to be uniquely defined.
    if points is None or len(points) < 3:
        raise ValueError("Input for circle fitting must be an array with at least 3 points.")

    def distance_from_center(center):
        distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
        return distances - distances.mean()
    
    initial_center = points.mean(axis=0)
    
    # DEBUG: The leastsq function can fail if points are collinear. Capture the success flag.
    optimized_center, ier = leastsq(distance_from_center, initial_center)
    
    # DEBUG: Check if the optimization algorithm converged successfully.
    if ier not in [1, 2, 3, 4]:
        raise RuntimeError("Circle fitting optimization failed to converge. The points may be collinear or otherwise problematic.")
    
    distances = np.sqrt((points[:, 0] - optimized_center[0])**2 + 
                       (points[:, 1] - optimized_center[1])**2)
    radius = distances.mean()
    
    return optimized_center[0], optimized_center[1], radius