import nvisii

opt = lambda : None
opt.nb_objects = 50
opt.spp = 256 
opt.width = 500
opt.height = 500 
opt.out = "05_load_obj_file.png" 

# # # # # # # # # # # # # # # # # # # # # # # # #
nvisii.initialize(headless=True, verbose=True)

nvisii.enable_denoiser()

camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create("camera"),
    camera = nvisii.camera.create(
        name = "camera",  
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
    eye = (1,0.7,0.2),
)
nvisii.set_camera_entity(camera)

nvisii.set_dome_light_sky(sun_position = (10, 10, 1), saturation = 2)
nvisii.set_dome_light_exposure(4)

# # # # # # # # # # # # # # # # # # # # # # # # #

# This function loads a signle obj mesh. It ignores 
# the associated .mtl file
sdb = nvisii.import_scene(
    file_path = './calipso.obj',
    position = (0,0,0),
    scale = (0.1, 0.1, 0.1),
    rotation = nvisii.angleAxis(3.14 * .5, (1,0,0)),
    args = ["verbose"] # list assets as they are loaded
)


mesh = nvisii.mesh.create_from_file("obj", "./calipso.obj")

# obj_entity = nvisii.entity.create(
#     name="obj_entity",
#     mesh = mesh,
#     transform = nvisii.transform.create("obj_entity"),
#     material = nvisii.material.create("obj_entity")
# )

# # lets set the obj_entity up
# obj_entity.get_transform().set_rotation( 
#     (0.7071, 0, 0, 0.7071)
# )
print()

# obj_entity.get_transform().set_scale((0.2, 0.2, 0.2))

# obj_entity.get_material().set_base_color(
#     (0.9,0.12,0.08)
# )  
# obj_entity.get_material().set_roughness(0.7)   
# obj_entity.get_material().set_specular(1)   
# obj_entity.get_material().set_sheen(1)


# # # # # # # # # # # # # # # # # # # # # # # # #
for i_s, s in enumerate(sdb.entities):
    print(s.get_name().lower())

    # if s.get_name().lower() == "calipso_v016_trip.009_calipso_v016_trip.009":
        # s.get_transform().set_rotation( 
        #     (0.7071, 0, 0, 0.7071)
        # )



nvisii.render_to_file(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp,
    file_path=opt.out 
)

# let's clean up GPU resources
nvisii.deinitialize()