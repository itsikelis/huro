import numpy as np

def rotate(quat, grav):
    # store shape
    # reshape to (N, 3) for multiplication
    # extract components from quaternions
    xyz = quat[1:3]
    t = np.cross(xyz, grav)*2.0
    return grav - quat[0:1] * t + np.cross(xyz,t)