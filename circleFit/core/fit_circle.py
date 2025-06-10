"""Circle fitting algorithm."""
import numpy as np
from scipy.optimize import leastsq

def fit_circle_to_points(points):
    """Fit a circle to points using least squares optimization."""
    def distance_from_center(center):
        distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
        return distances - distances.mean()
    
    initial_center = points.mean(axis=0)
    optimized_center, _ = leastsq(distance_from_center, initial_center)
    
    distances = np.sqrt((points[:, 0] - optimized_center[0])**2 + 
                       (points[:, 1] - optimized_center[1])**2)
    radius = distances.mean()
    
    return optimized_center[0], optimized_center[1], radius
