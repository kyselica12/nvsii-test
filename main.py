from orbital import KeplerianElements, earth

from sun import Sun
from observer import Observer
from object import Object
from simulator import Simulator
from renderer import NVISIIRenderer, import_calipso

sun = Sun()
observer = Observer(lat=0, lon=0)
orbit = KeplerianElements.with_altitude(altitude=200_000, body=earth)
object = Object(orbit=orbit, rotation_period=20, axis_rotation=(0, 0, 1))

simulator = Simulator(start_time=0, object=object, observer=observer, sun=sun)

calipso, size = import_calipso()
renderer = NVISIIRenderer(
    render_size=20,
    file_type="PNG",
    output_path="R:/Temp",
    object=calipso,
    object_size=size,
    camera_distance=200,
    spp=64
)

n_series = 1
n_examples = 5

idx = 0
for i in range(n_series):
    for sun_pos, observer_pos, (object_pos, angle, object_axis_rotation) \
            in simulator.simulate_period(n_examples):

        renderer.move(sun_pos, observer_pos, object_pos, angle, object_axis_rotation)
        renderer.render(f"calipso_{i:03d}_{idx:03d}")
        idx += 1

renderer.close()
