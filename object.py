import numpy as np

from utils import TimeSec, tilt_to_equator


class Object:

    def __init__(self, orbit, rotation_period, axis_rotation) -> None:
        self.period = orbit.T
        self.t = 0
        self.rotation_period = rotation_period
        self.orbit = orbit
        self.axis_rotation = axis_rotation

    def possition_at_time(self, t):
        p = self.orbit.a * (1 - self.orbit.e ** 2)
        self.orbit.t = t
        x, y, z = np.array([np.cos(self.orbit.f), np.sin(self.orbit.f), 0 * self.orbit.f]) \
                  * p / (1 + self.orbit.e * np.cos(self.orbit.f))

        pos = tilt_to_equator(np.array([x, y, z]))

        return pos



