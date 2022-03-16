import numpy as np

from utils import angle_between_vectors, EARTH_RADIUS, TimeSec
from object import Object
from sun import Sun
from observer import Observer


class Simulator:

    def __init__(self, start_time, object: Object, observer: Observer, sun: Sun) -> None:

        self.t = start_time
        self.object = object
        self.observer = observer
        self.sun = sun

    def _is_night(self):
        v1 = self.sun.possition_at_time(self.t)
        v2 = self.observer.possition_at_time(self.t)
        # angle between Sun and observer has to be more than 90 degrees
        # so observer is in the shadow
        theta = np.arccos((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi

        return theta > 90 + 18

    def _sattelite_in_field_of_view(self):
        v1 = self.sun.possition_at_time(self.t)
        v2 = self.object.possition_at_time(self.t)
        v3 = self.observer.possition_at_time(self.t)

        p = np.array([0, 0, 0])

        # distance to Earth center
        # if its greater than radius sattelite can see Sun

        distance = (np.linalg.norm(np.cross((p - v1), (p - v2)))) / np.linalg.norm(v2 - v1)

        # angle between telescope and satellite < 90°
        theta = angle_between_vectors(v3, v2 - v3)

        return distance > EARTH_RADIUS and theta < 90

    def switch_to_night(self):
        while not self._is_night():
            self.t += TimeSec.HOUR // 2

    def forward_to_satellite_observation(self):
        self.switch_to_night()
        while not self._sattelite_in_field_of_view():
            self.t += self.object.period // 100

    def simulate_period(self, n_examples):

        self.forward_to_satellite_observation()

        step = self.object.rotation_period / n_examples
        end_t = self.t + self.object.rotation_period
        angle = 0
        angle_step = 2 * np.pi / n_examples

        while self.t <= end_t and self._is_night() and self._sattelite_in_field_of_view():
            sun_pos = self.sun.possition_at_time(self.t)
            observer_pos = self.observer.possition_at_time(self.t)
            object_pos = self.object.possition_at_time(self.t)

            yield sun_pos, observer_pos, (object_pos, angle, self.object.axis_rotation)

            self.t += step
            angle += angle_step


