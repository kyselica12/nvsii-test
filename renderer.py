import nvisii
import numpy as np

from utils import SUN_DISTANCE
from abc import ABC, abstractmethod


class Renderer(ABC):

    @abstractmethod
    def move(self, sun_pos, observer_pos, object_pos, angle, object_axis_rotation):
        pass

    @abstractmethod
    def render(self, name):
        pass


class NVISIIRenderer(Renderer):

    def __init__(self, render_size=20, file_type="HDR", output_path="./", object=None, object_size=0, spp=64,
                 camera_distance=200) -> None:
        self.render_size = render_size
        self.file_type = file_type
        self.output_path = output_path
        self.object = object
        self.object_size = object_size
        self.spp = spp
        self.camera_distance = camera_distance
        self.sensor_size = 100
        self.size_multiplier = 1.5

        self.sun_distance_ratio = 1 / 1000

        self._initialize()
        self._create_camera()
        self._create_sun()

    def _initialize(self):
        nvisii.initialize()
        nvisii.resize_window(self.render_size, self.render_size)
        nvisii.enable_denoiser()

    def _create_sun(self):
        nvisii.set_dome_light_intensity(0)
        nvisii.disable_dome_light_sampling()
        self.sun = nvisii.entity.create(
            name="sun",
            light=nvisii.light.create('sun'),
            # mesh = nvisii.mesh.create_sphere('sun', radius=696_340),
            transform=nvisii.transform.create(
                name='sun',
                position=(SUN_DISTANCE * self.sun_distance_ratio, 0, 0)
            )
        )
        self.sun.get_light().set_intensity(10 ** 5)
        self.sun.get_light().set_exposure(50)
        self.sun.get_light().set_temperature(8000)

    def _create_camera(self):
        self.camera = nvisii.entity.create(
            name="camera",
            transform=nvisii.transform.create("camera"),
            camera=nvisii.camera.create(
                name="camera",
                aspect=1.
            )
        )
        nvisii.set_camera_entity(self.camera)

        magnification = self.sensor_size / (self.object_size * self.size_multiplier * 1000)
        focal_length = (self.camera_distance / ((1 / magnification) + 1)) * 1000
        self.camera.get_camera().set_focal_length(focal_length, self.sensor_size, self.sensor_size)

    def move(self, sun_pos, observer_pos, object_pos, angle, object_axis_rotation):
        if not isinstance(observer_pos, np.ndarray):
            camera_pos = np.array(observer_pos)
        if not isinstance(object_pos, np.ndarray):
            object_pos = np.array(object_pos)

        offset = (object_pos - camera_pos)
        direction = offset / np.linalg.norm(offset)
        direction *= self.camera_distance

        self.object.get_transform().set_position(direction)
        self.object.get_transform().set_angle_axis(angle, object_axis_rotation)

        self.camera.get_transform().look_at(
            at=direction,
            up=(0, 0, 1),
            eye=(0, 0, 0)
        )

        sun_position_offset = object_pos - direction
        self.sun.get_transform().set_position(
            (sun_pos - sun_position_offset) * self.sun_distance_ratio
        )

    def render(self, name):
        nvisii.render_to_file(
            width=self.render_size,
            height=self.render_size,
            samples_per_pixel=self.spp,
            file_path=f"{self.output_path}\{name}.{self.file_type}"
        )

    def close(self):
        nvisii.deinitialize()

def import_calipso():
    sdb = nvisii.import_scene(
        file_path='../objects/calipso.obj',
        position=(0, 0, 0),
        args=["verbose"]
    )

    # # # # # # # # # # # # # # # # # # # # # # # # #
    size = 0
    S = None
    import numpy as np
    S = sdb.entities[0]
    S.get_transform().set_angle_axis(
        90 / 180 * np.pi, (1, 0, 0)
    )

    max_values = [0, 0, 0]
    min_values = [np.inf, np.inf, np.inf]
    for i_s, s in enumerate(sdb.entities):
        mc = s.get_max_aabb_corner()
        nc = s.get_min_aabb_corner()

        max_values = [max(v, u) for v, u in zip(max_values, mc)]
        min_values = [min(v, u) for v, u in zip(min_values, nc)]

    size = max(np.array(max_values) - np.array(min_values))

    return S, size