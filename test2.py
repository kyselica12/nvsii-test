from dis import dis
from turtle import position
from unicodedata import name
import nvisii
import time
import numpy as np

opt = lambda: None
opt.nb_objects = 2
opt.spp = 128
opt.width = 400
opt.height = 400
opt.out = "05_load_obj_file.png"

# # # # # # # # # # # # # # # # # # # # # # # # #
nvisii.initialize()
nvisii.resize_window(int(opt.width), int(opt.height))
nvisii.enable_denoiser()

camera = nvisii.entity.create(
    name="camera",
    transform=nvisii.transform.create("camera"),
    camera=nvisii.camera.create(
        name="camera",
        aspect=float(opt.width) / float(opt.height)
    )
)

nvisii.set_camera_entity(camera)
nvisii.set_dome_light_intensity(0)
nvisii.disable_dome_light_sampling()

sun = nvisii.entity.create(
    name="sun",
    light=nvisii.light.create('sun'),
    # mesh = nvisii.mesh.create_sphere('sun', radius=696_340),
    transform=nvisii.transform.create(
        name='sun',
        position=(14_960_000_000, 0, 0)
    )
)

sun.get_light().set_intensity(10 ** 5)
sun.get_light().set_exposure(50)
sun.get_light().set_temperature(8000)


# # # # # # # # # # # # # # # # # # # # # # # # #
# This function loads a signle obj mesh. It ignores
# the associated .mtl file
def import_calipso():
    sdb = nvisii.import_scene(
        file_path='./objects/calipso.obj',
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


def import_atlas():
    mesh = nvisii.mesh.create_from_file("glb", "./objects/Atlas_V_stage2.glb")

    obj_entity = nvisii.entity.create(
        name="obj_entity",
        mesh=mesh,
        transform=nvisii.transform.create("obj_entity"),
        material=nvisii.material.create("obj_entity")
    )

    S = obj_entity
    mc = S.get_max_aabb_corner()
    nc = S.get_min_aabb_corner()
    size = max(mc - nc)

    return S, size


S, size = import_calipso()

camera.get_transform().look_at(
    at=(0, 0, 0),
    up=(0, 0, 1),
    eye=(0, 1, 0),
)

# 0.01
S.get_transform().set_angle_axis(0 / 180 * np.pi, (1, 0, 0))

print(size)

x = 0
y = 1

S.get_transform().add_angle_axis(10 / 180 * np.pi, (0, 0, 1))

w = 100
for i in range(10):
    x -= 2
    y += 0.5

    # S.get_transform().add_angle_axis(10 / 180 * np.pi, (0,0,1))
    S.get_transform().set_position((0, 200, 0))

    p1 = S.get_transform().get_position()
    p2 = camera.get_transform().get_position()
    p1 = np.array(list(p1))
    p2 = np.array(list(p2))
    dist = np.sum((p1 - p2) ** 2) ** 0.5

    print(p1, p2)

    magnification = w / (size * 2 * 1000)

    focus = (dist / ((1 / magnification) + 1)) * 1000

    print(dist, size, focus, magnification)

    camera.get_transform().look_at(
        at=S.get_transform().get_position(),
        up=(0, 0, 1),
        eye=(0, 0, 0), previous=False
    )
    camera.get_camera().set_focal_length(focus, w, w)
    time.sleep(0.5)

nvisii.render_to_file(
    width=opt.width,
    height=opt.height,
    samples_per_pixel=opt.spp,
    file_path="calipso_dist_200_000.png"
)

# let's clean up GPU resources
nvisii.deinitialize()