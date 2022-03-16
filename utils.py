import numpy as np

EARTH_AXIS_TILT = 23.5
EARTH_RADIUS = 6_371_000  # [m]
SUN_DISTANCE = 14_960_000_000_000  # [m]


class TimeSec:
    MIN = 60
    HOUR = 60 ** 2
    DAY = 24 * 60 ** 2
    YEAR = 365 * 24 * (60 ** 2) + 5 * (60 ** 2) + 48 * 60 + 45


def tilt_to_equator(pos: np.ndarray) -> np.ndarray:
    dy = np.cos(np.radians(EARTH_AXIS_TILT))
    dz = np.sin(np.radians(EARTH_AXIS_TILT))

    x = pos[0]
    y = dy * pos[1]
    z = dz * pos[1] + pos[2]

    return np.array([x, y, z])


def angle_between_vectors(v, u):
    theta = np.arccos(v @ u / (np.linalg.norm(v) * np.linalg.norm(u)))
    return theta * 180 / np.pi
