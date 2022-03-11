from distutils.log import debug
import glob

import bpy
from math import radians
#from scipy.constants import kilo
from orbital import KeplerianElements, earth
from copy import copy
import numpy as np
from numpy import cos, save, sin, pi
import time
import traceback
import os
import logging
import shutil

import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

from collections import namedtuple

kilo = 1000
# CONSTANTS -------------------------------------------
RATIO = 1_000_000

logging.basicConfig(filename="C:\\Users\\danok\\work\\dizertacka\\artificial_data\\LOG.log", 
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)

# YEAR = 365 days, 5 hours, 48 minutes and 45 seconds
# start_date = Phelion - June solstice 21/22
class TimeSec:
    MIN = 60
    HOUR = 60 ** 2
    DAY = 24 * 60 ** 2
    YEAR = 365 * 24 * (60 ** 2) + 5 * (60 ** 2) + 48 * 60 + 45


SatelliteArguments = namedtuple("SatelliteArguments", "filename orbit axis_rotation rotation_period initial_scale local_rotation_axis")
GPS_Coordinates = namedtuple("GPS_Coordinates", "lat lon")

def tilt_to_equator(x, y, z):
    dy = np.cos(np.radians(Observer.EARTH_AXIS_TILT))
    dz = np.sin(np.radians(Observer.EARTH_AXIS_TILT))

    y = dy * y
    z = dz * y + z

    return x, y, z


def angle_between_vectors(v, u):
    theta = np.arccos(v @ u / (np.linalg.norm(v) * np.linalg.norm(u)))
    return theta * 180 / np.pi


class Observer:
    LEN_FOCUS = 20_000_000 / RATIO # [mm]
    EARTH_RADIUS = 6_371_000  # [m]
    EARTH_AXIS_TILT = 23.5  # [degrees]

    def __init__(self, lat, lon, move_to_center=True) -> None:
        self.lat, self.lon = lat, lon
        self.move_to_center = move_to_center

        if not move_to_center:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=self.EARTH_RADIUS / RATIO, enter_editmode=False, align='WORLD',
                                                location=(0, 0, 0), scale=(1, 1, 1))
            self.earth = bpy.context.object

        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', rotation=(0, 0, 0),
                                  location=(0, 0, 0), scale=(1, 1, 1))
        self.camera = bpy.data.objects["Camera"]

        self.camera = bpy.data.objects["Camera"]
        self.camera.data.lens = 20_000_000  # for example 2 or 3 meters for 1000 km distance is enaught

        self.camera.data.clip_start = 0.001
        self.camera.data.clip_end = 100_000
        self.camera.data.sensor_width = 100
        self.satellite = None
        self.magnification = 1

        if self.move_to_center:
            self.camera.location[0] = 0
            self.camera.location[1] = 0
            self.camera.location[2] = 0

    def focus_on_satellite(self, satellite):
        self.satellite = satellite

        size2 = satellite.size
        size = max(self.satellite.obj.dimensions.x,self.satellite.obj.dimensions.y, self.satellite.obj.dimensions.z)
        self.magnification = self.camera.data.sensor_width / (size * 1.5 * RATIO * 1000) 
        logging.debug(f"Object size {size}, {satellite.size} ,{size * RATIO} {max(self.satellite.obj.dimensions.x,self.satellite.obj.dimensions.y, self.satellite.obj.dimensions.z)}")
        self.LEN_FOCUS /= size        
        bpy.ops.object.select_pattern(pattern="Camera")
        bpy.ops.object.constraint_add(type='TRACK_TO')
        self.camera.constraints["Track To"].target = satellite.obj

    def possition_at_time(self, t):
        t = (t / TimeSec.DAY) % 1

        lat, lon = np.deg2rad(self.lat), np.deg2rad((self.lon + t * 360) % 360)
        x = self.EARTH_RADIUS * np.cos(lat) * np.cos(lon)
        y = self.EARTH_RADIUS * np.cos(lat) * np.sin(lon)
        z = self.EARTH_RADIUS * np.sin(lat)

        x, y, z = tilt_to_equator(x, y, z)

        x /= RATIO
        y /= RATIO
        z /= RATIO

        return x, y, z

    def move(self, t):
        x, y, z = self.possition_at_time(t)
        if not self.move_to_center:
            # time in sec
            self.camera.location[0] = x
            self.camera.location[1] = y
            self.camera.location[2] = z

            logging.debug(f"camera location: {self.camera.location[0], self.camera.location[1], self.camera.location[2]}")
        
        v = np.array([x,y,z])
        u = np.array(self.satellite.possition_at_time(t))
        distance = np.linalg.norm(u - v) * RATIO / kilo
        logging.info(f"distance {distance}")
 
        self.camera.data.lens = (distance * kilo * kilo / ((1 / self.magnification) + 1))
        
class Sun:
    RADIUS = 696_340_000  # m
    EARTH_DISTANCE = 149_600_000_000  # m

    def __init__(self) -> None:
        bpy.ops.object.light_add(type='SUN', radius=self.RADIUS / RATIO, align='WORLD',
                                 location=(0, 0, 0), scale=(1, 1, 1))
        self.light = bpy.context.object
        self.light.rotation_euler[0] = -radians(90)
        self.move(0)

    def move(self, t, compensate_camera_location=(0,0,0)):
        t = (t / TimeSec.YEAR) % 1
        theta = t * 2 * np.pi

        dx = np.sin(theta)
        dy = np.cos(theta)

        x = dx * self.EARTH_DISTANCE / RATIO
        y = dy * self.EARTH_DISTANCE / RATIO

        self.light.location[0] = x
        self.light.location[1] = y
        self.light.rotation_euler[2] = 2 * pi - theta

        self.light.location[0] -= compensate_camera_location[0]
        self.light.location[1] -= compensate_camera_location[1]
        self.light.location[2] -= compensate_camera_location[2]

    def possition_at_time(self, t):
        t = (t / TimeSec.YEAR) % 1
        theta = t * 2 * np.pi

        dx = np.sin(theta)
        dy = np.cos(theta)

        x = dx * self.EARTH_DISTANCE / RATIO
        y = dy * self.EARTH_DISTANCE / RATIO

        return x, y, 0

class Satellite:

    def __init__(self, path, orbit, rotation_period, axis_rotation, initial_scale, local_rotation_axis="Y") -> None:
        self.name, type = os.path.split(path)[1].split('.')
        self.period = orbit.T
        self.t = 0
        self.rotation_period = rotation_period
        
        self.orbit = orbit

        self.axis_rotation = axis_rotation
        self.local_rotation_axis = local_rotation_axis
        
        # bpy.ops.mesh.primitive_cube_add(size=5)
                
        if type == "obj":
            logging.info(f"load {self.name} {path}")
            bpy.ops.import_scene.obj(filepath=path)
        elif type == "glb":
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            logging.error("Cannot import object file")

        self.obj = None
        for obj in bpy.context.scene.objects:
            logging.info(f"{obj.name}, {self.name}")
            name = obj.name.split('.')[0]
            #TODO opravit meno Atlas 5
            if name.lower() in self.name.lower() or self.name.lower() in name.lower() or name == "_root":
                self.obj = obj
                break

        logging.info(f"RATIO: {RATIO} {1 * initial_scale / RATIO}")

        self.obj.rotation_mode = "XYZ"

        self.size = max(self.obj.dimensions.x,self.obj.dimensions.y, self.obj.dimensions.z) * initial_scale / RATIO


        self.obj.scale[0] = 1 * initial_scale / RATIO  # convert to meters
        self.obj.scale[1] = 1 * initial_scale / RATIO
        self.obj.scale[2] = 1 * initial_scale / RATIO

        self.obj.rotation_euler[0] = np.radians(axis_rotation[0])
        self.obj.rotation_euler[1] = np.radians(axis_rotation[1])
        self.obj.rotation_euler[2] = np.radians(axis_rotation[2])

    
    def move(self, t, compensate_camera_location=(0,0,0)):

        x, y, z = self.possition_at_time(t)

        logging.debug(f"SATELITE MOVE: {x, y, z}")
        self.obj.location[0] = x
        self.obj.location[1] = y
        self.obj.location[2] = z

        self.obj.location[0] -= compensate_camera_location[0]
        self.obj.location[1] -= compensate_camera_location[1]
        self.obj.location[2] -= compensate_camera_location[2]

        # ---- ROTATION along axis --------
        t2 = (t / self.rotation_period) % 1
        dt = t2 - self.t if t2 >= self.t else t2 + 1 - self.t
        theta = dt * 2 * np.pi

        logging.debug(f"Rotation {theta * 180 / np.pi:.2f}, {self.t * 360:.2f}")
        logging.debug(f"position satellite: {self.obj.location}, {compensate_camera_location}")

        self.obj.rotation_euler.rotate_axis(self.local_rotation_axis, theta)

        logging.debug(f"ROTATION {self.obj.rotation_euler[2] * 180 / np.pi:.2f}")
        self.t = t2

    def possition_at_time(self, t):
        p = self.orbit.a * (1 - self.orbit.e ** 2)
        self.orbit.t = t
        x, y, z = np.array([cos(self.orbit.f), sin(self.orbit.f), 0 * self.orbit.f]) \
                  * p / (1 + self.orbit.e * cos(self.orbit.f))

        x, y, z = tilt_to_equator(x, y, z)

        x /= RATIO
        y /= RATIO
        z /= RATIO

        return x, y, z   

class Scene:

    def __init__(self, satellite_arguments: SatelliteArguments, telescope_coordinates: GPS_Coordinates,
                 render_image_size=20,
                 light_curve_size=300,
                 start_time=0,
                 output_folder=".",
                 image_folder=".",
                 save_images=False,
                 name="telescope",
                 image_type="HRD") -> None:

        self.t = start_time
        self.image_index = 0
        self.series_index = 0
        self.output_folder = output_folder
        self.image_folder = image_folder
        self.save_images = save_images
        self.name = name
        self.light_curve_size = light_curve_size
        self.render_image_size = render_image_size
        self.IMAGE_TYPE = image_type

        self._setup_scene()
        self._create_objects(satellite_arguments, telescope_coordinates)

    def _create_objects(self, satellite_arguments, telescope_coordinates):
        self.sun = Sun()
        self.satellite = Satellite(path=satellite_arguments.filename,
                                   orbit=satellite_arguments.orbit,
                                   rotation_period=satellite_arguments.rotation_period,
                                   axis_rotation=satellite_arguments.axis_rotation,
                                   initial_scale=satellite_args.initial_scale,
                                   local_rotation_axis=satellite_args.local_rotation_axis)
        self.telescope = Observer(telescope_coordinates.lat,
                                  telescope_coordinates.lon,
                                  move_to_center=True)
        self.telescope.focus_on_satellite(self.satellite)

    def _setup_scene(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        bpy.context.scene.use_gravity = False
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
        bpy.data.scenes["Scene"].render.resolution_x = self.render_image_size
        bpy.data.scenes["Scene"].render.resolution_y = self.render_image_size

    def _is_night(self):

        x1, y1, z1 = self.sun.possition_at_time(self.t)
        x2, y2, z2 = self.telescope.possition_at_time(self.t)

        v1 = np.array([x1, y1])
        
        v2 = np.array([x2, y2])

        # angle between Sun and observer has to be more than 90 degrees
        # so observer is in the shadow

        theta = np.arccos((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi

        logging.debug(f"{v1} {v2, z2} {theta:0.4f}")

        return theta > 90 + 18

    def _sattelite_in_field_of_view(self):
        v1 = np.array(self.sun.possition_at_time(self.t))
        v2 = np.array(self.satellite.possition_at_time(self.t))
        v3 = np.array(self.telescope.possition_at_time(self.t))

        p = np.array([0, 0, 0])

        # distance to Earth center
        # if its greater than radius sattelite can see Sun
        distance = (np.linalg.norm(np.cross((p - v1), (p - v2)))) / np.linalg.norm(v2 - v1)

        # angle between telescope and satellite < 90°
        theta = angle_between_vectors(v3, v2 - v3)
        
        logging.debug(f"{theta:.4f}")

        return distance > Observer.EARTH_RADIUS / RATIO and theta < 90

    def switch_to_night(self):
        logging.debug("switch to night loop")
        while not self._is_night():
            self.t += TimeSec.HOUR // 2

        logging.debug("NIGHT")

    def capture_loop(self):
        step = self.satellite.rotation_period / self.light_curve_size
        logging.info(f"CAPTURE LOOP {step, self.satellite.rotation_period}")

        folder = self._get_image_folder()
        Path(folder).mkdir(parents=True, exist_ok=True)

        while self._is_night() and self._sattelite_in_field_of_view() and self.image_index < self.light_curve_size:
            logging.info(f"Moving {self.t}")
            dx, dy, dz = self.telescope.possition_at_time(self.t)
            self.sun.move(self.t, compensate_camera_location=(dx, dy, dz))
            self.satellite.move(self.t, compensate_camera_location=(dx, dy, dz))
            self.telescope.move(self.t)

            logging.info(f"capture: {self.image_index}, {self.telescope.possition_at_time(self.t)} {self.t} {self.satellite.obj.rotation_euler}")
            self.render_image()
            self.image_index += 1

            self.t += step

        if self.image_index > 0:
            self.create_light_curve()

    def loop(self, num=5):
        logging.info("loop")
        for i in range(num):
            logging.info("forwarding in time")
            self.forward_to_satellite_observation()
            logging.info("capture loop")
            self.capture_loop()
            self.series_index += 1
            self.image_index = 0

    def forward_to_satellite_observation(self):
        self.switch_to_night()
        
        logging.debug("satellite loop")
        while not self._sattelite_in_field_of_view():
            self.t += self.satellite.period // 100

    def render_image(self):
        bpy.context.scene.camera = self.telescope.camera
        bpy.context.scene.render.image_settings.file_format = self.IMAGE_TYPE
        image_name = self._get_image_name() + f".{self.IMAGE_TYPE}"
        folder = self._get_image_folder()
        bpy.context.scene.render.filepath = os.path.join(folder, image_name)
        logging.info(f"{bpy.context.scene.render.filepath}")
        bpy.ops.render.render(write_still=True)
        bpy.ops.render.render(use_viewport=True)

    def _get_image_folder(self):
        folder = f"{self.image_folder}/lc_{self.name}_{self.series_index}"
        return folder

    def _get_image_name(self):
        image_name = f"{self.name}_{self.series_index:04d}_{self.image_index:04d}"
        return image_name

    def create_light_curve(self):

        folder = self._get_image_folder()
        values = np.zeros((self.light_curve_size,))
        for file in glob.iglob(f"{folder}/{self.name}_*_*.{self.IMAGE_TYPE}"):
            name = os.path.split(file)[-1]
            idx = int(name.split(".")[0].split("_")[-1])
            img = cv.imread(file)
            img_gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            values[idx] = np.sum(img_gr)

        if not self.save_images and os.path.exists(folder):
            shutil.rmtree(folder)

        values = values.reshape(1, self.light_curve_size)
        file_name = f"{self.output_folder}/{self.name}_multi_array.npy"

        logging.info(f"Shape {values.shape} {np.sum(values[0] != 0)} {self.image_index}")
        ok = values[0] != 0
        d = values[0][ok]

        d = (d - np.min(d)) / (np.max(d) - np.min(d))

        fig, ax = plt.subplots()
        ax.scatter(np.arange(len(values[0]))[ok], values[0][ok])
        plt.savefig(f"{self.output_folder}/{self.name}_{self.series_index}_plot.png")
        
        with open(file_name, "ab") as f:
            np.save(f, values)


if __name__ == "__main__":

    try:
        logging.info("------------- START ---------------")
        
        OUTPUT_FOLDER = "C:\\Users\\danok\\work\\dizertacka\\artificial_data\\data"
        IMAGE_FOLDER = "R:\\Temp"
        SEQUENCE_NAME = "Calipso" #"Calipso"#"Ares_1"#"Calipso_v016_trip"
        BOX_WING_SATELLITE = "C:\\Users\\danok\\work\\dizertacka\\artificial_data\\objects\\calipso.obj"
        ROCKET_BODY = "C:\\Users\\danok\\work\\dizertacka\\artificial_data\\objects\\ares1_scaled.obj"
        FALCON9 = "C:\\Users\\danok\\work\\dizertacka\\artificial_data\\objects\\spacex-falcon-9-v11\\falcon9.obj"
        ALTAS_5 = "C:\\Users\\danok\\work\\dizertacka\\artificial_data\\objects\\Atlas_V_stage2.glb"

        orbit = KeplerianElements.with_altitude(altitude=200_000, body=earth)
        SATELLITE_PERIOD = 20 #TimeSec.MIN * 2 #  orbit.T / 6 / 10
        
        satellite_args = SatelliteArguments(filename=ALTAS_5,
                                            orbit=orbit,
                                            rotation_period=SATELLITE_PERIOD,
                                            axis_rotation=(0, 0, 0), # 90,0,0
                                            initial_scale=10,
                                            local_rotation_axis="Z")  # Y
        telescope_pos = GPS_Coordinates(lat=0, lon=0)

        scene = Scene(satellite_arguments=satellite_args,
                      telescope_coordinates=telescope_pos,
                      render_image_size=300,
                      light_curve_size=10,
                      start_time=0, #TimeSec.DAY * 3 + TimeSec.HOUR * 4,
                      output_folder=OUTPUT_FOLDER,
                      image_folder=IMAGE_FOLDER,
                      save_images=True,
                      name=SEQUENCE_NAME,
                      image_type="PNG")

        logging.info("START")
        
        #scene.forward_to_satellite_observation()
        t = 0#scene.t
        
        bpy.context.scene.frame_set(0)

        dx, dy, dz = scene.telescope.possition_at_time(t)
        scene.sun.move(t, compensate_camera_location=(dx, dy, dz))
        scene.satellite.move(t,compensate_camera_location=(dx, dy, dz))
        scene.telescope.move(t)

        #scene.satellite.obj.rotation_euler.rotate_axis("Z", radians(30))

        scene.satellite.obj.keyframe_insert(data_path="location", index=-1)
        scene.telescope.camera.keyframe_insert(data_path="location", index=-1)
        scene.sun.light.keyframe_insert(data_path="location", index=-1)

        scene.satellite.obj.keyframe_insert(data_path="rotation_euler", index=-1)
        scene.telescope.camera.keyframe_insert(data_path="rotation_euler", index=-1)
        scene.sun.light.keyframe_insert(data_path="rotation_euler", index=-1)

        for i in range(10):
            t += 2
            bpy.context.scene.frame_set(i*25)
            dx, dy, dz = scene.telescope.possition_at_time(t)
            scene.sun.move(t, compensate_camera_location=(dx, dy, dz))
            scene.satellite.move(t,compensate_camera_location=(dx, dy, dz))
            scene.telescope.move(t)

            # scene.render_image()
            scene.image_index += 1

            scene.satellite.obj.keyframe_insert(data_path="location", index=-1)
            scene.telescope.camera.keyframe_insert(data_path="location", index=-1)
            scene.sun.light.keyframe_insert(data_path="location", index=-1)

            scene.satellite.obj.keyframe_insert(data_path="rotation_euler", index=-1)
            scene.telescope.camera.keyframe_insert(data_path="rotation_euler", index=-1)
            scene.sun.light.keyframe_insert(data_path="rotation_euler", index=-1)


        # scene.loop(1)
        logging.info("END")
        # TODO -> nautical twighlight + 12

    except Exception as e:
        logging.error("".join(traceback.TracebackException.from_exception(e).format()))

    # .\blender.exe -b -t 0 --python C:\Users\danok\work\dizertacka\artificial_data\basic_scene.py
 