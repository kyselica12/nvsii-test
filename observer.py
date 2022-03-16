import numpy as np

from utils import TimeSec, tilt_to_equator, EARTH_RADIUS



class Observer:
    def __init__(self, lat, lon) -> None:
        self.lat, self.lon = lat, lon

    def possition_at_time(self, t):
        t = (t / TimeSec.DAY) % 1

        lat, lon = np.deg2rad(self.lat), np.deg2rad((self.lon + t * 360) % 360)
        x = EARTH_RADIUS * np.cos(lat) * np.cos(lon)
        y = EARTH_RADIUS * np.cos(lat) * np.sin(lon)
        z = EARTH_RADIUS * np.sin(lat)

        pos = tilt_to_equator(np.array(x, y, z))

        return pos
