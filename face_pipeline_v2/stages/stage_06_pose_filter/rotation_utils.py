import numpy as np


def compute_euler_angles_from_rotation_matrix(R):
    """
    Convert rotation matrix to 6DRepNet style Euler angles

    Return:
        pitch, yaw, roll
        degree
    """

    sy = np.sqrt(
        R[0, 0] ** 2 +
        R[1, 0] ** 2
    )

    singular = sy < 1e-6


    if not singular:

        x = np.arctan2(
            R[2,1],
            R[2,2]
        )

        y = np.arctan2(
            -R[2,0],
            sy
        )

        z = np.arctan2(
            R[1,0],
            R[0,0]
        )

    else:

        x = np.arctan2(
            -R[1,2],
            R[1,1]
        )

        y = np.arctan2(
            -R[2,0],
            sy
        )

        z = 0


    return np.degrees(
        np.array([x, y, z])
    )