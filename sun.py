import numpy as np

from utils import TimeSec




class Sun:
    RADIUS = 696_340_000  # m
    EARTH_DISTANCE = 149_600_000_000  # m

    def possition_at_time(self, t):
        t = (t / TimeSec.YEAR) % 1
        theta = t * 2 * np.pi

        dx = np.sin(theta)
        dy = np.cos(theta)

        x = dx * self.EARTH_DISTANCE
        y = dy * self.EARTH_DISTANCE

        return np.array([x, y, 0])