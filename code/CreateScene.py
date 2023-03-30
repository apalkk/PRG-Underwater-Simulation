# Create Custom Mesh for Ground surface

# Author: Chahat Deep Singh
# Modified By: Nitesh Jha, Mayank Joshi
# University of Maryland. College Park
# MIT License (c) 2021

# Import Libraries
import bpy
import bmesh
import ant_landscape
import math
import numpy as np
import random
import os
from math import pi

# Force Enter the Object Mode
try:
    bpy.ops.object.mode_set(mode = 'OBJECT')
except:
    pass


# List of Flags:

START_FRAME = 1
END_FRAME = 200

SURFACE_SIZE = 50
def change_pass_index(ob, levels=10):
    def recurse(ob, parent, depth):
        if depth > levels:
            return
        if ob.name.split('_')[0]=='Untitled':
            ob.pass_index=1
            print("changed pass_index")
        print("  " * depth, ob.name)

        for child in ob.children:
            recurse(child, ob,  depth + 1)
    recurse(ob, ob.parent, 0)


def add_bluerov(model_path,bluerov_location=(0,0,0), front_cam_orientation=(-20, 180, 0), bottom_cam_orientation=(90, 0, 180)):
    bpy.ops.wm.collada_import(filepath=model_path)
    #bpy.context.object.pass_index = 1
    model_name = "BlueROV"
    #scene = bpy.context.scene
    #root_obs = (o for o in scene.objects if not o.parent)

    #for o in root_obs:
    #    change_pass_index(o)

    obj=bpy.context.scene.objects["Untitled_282"]
    obj.name=model_name
    # Initial position at origin
    obj.location.x=0
    obj.location.y=0
    obj.location.z=0
    obj.rotation_euler.x=1.57 # Orientation of DAE

    # camera facing downwards
    roll=bottom_cam_orientation[0] * np.pi / 180.0               # same as obj.rotation_euler.z
    pitch=bottom_cam_orientation[1] * np.pi / 180.0
    yaw=bottom_cam_orientation[2] * np.pi / 180.0

    bottom_cam, cam_obj = set_camera(0,0,0,roll, pitch, yaw)
    cam_obj.parent = bpy.data.objects[model_name]

    # camera (lidar/sonar) facing front
    roll=front_cam_orientation[0] * np.pi / 180.0
    pitch=front_cam_orientation[1] * np.pi / 180.0
    yaw=front_cam_orientation[2] * np.pi / 180.0

    front_cam, cam_obj2 = set_camera(0, 0, 0.03, roll, pitch, yaw)
    cam_obj2.parent = bpy.data.objects[model_name]

    # Move to x,y,z
    obj=bpy.context.scene.objects[model_name]
    obj.location.x=bluerov_location[0]
    obj.location.y=bluerov_location[1]
    obj.location.z=bluerov_location[2]

    obj=bpy.context.scene.objects["BlueROV"]
    obj.location=bluerov_location

    return front_cam, bottom_cam

def set_light(x=0,y=0,z=60,energy=50000):
    # Select all the lights:
    bpy.ops.object.select_by_type(type='LIGHT')

    # Delete all the lights:
    bpy.ops.object.delete()

    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(x,y,z), scale=(1, 1, 1))
    # bpy.context.space_data.context = 'DATA'
    bpy.context.object.data.energy = energy



def set_camera(x=0, y=0, z=2, roll=0, pitch=0, yaw=0, track=False, focal_length=36):

    # creates a new camera object at x,y,z, roll, pitch, yaw
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(x, y, z),\
     rotation=(roll, pitch, yaw), scale=(1, 1, 1))

    if track:
        bpy.ops.object.constraint_add(type='TRACK_TO')
        bpy.context.object.constraints["Track To"].target = bpy.data.objects["BlueROV"]
    bpy.context.object.data.lens = focal_length

    return bpy.context.object.name, bpy.context.object



# Delete all the current meshes
def delete_objs():
    # Select all the lights:
    bpy.ops.object.select_by_type(type='LIGHT')

    # Delete all the lights:
    bpy.ops.object.delete()

    # Select all the Meshes:
    bpy.ops.object.select_by_type(type='MESH')

    # Delete all the objects
    bpy.ops.object.delete()

     # selects previously generated camera
    bpy.ops.object.select_by_type(type='CAMERA')

    # # deletes previously generated camera
    bpy.ops.object.delete()



    # Deselect all (if required):
    bpy.ops.object.select_all(action='DESELECT')


def apply_texture(PassiveObject, mat):
    if PassiveObject.data.materials:
        PassiveObject.data.materials[0] = mat
    else:
        PassiveObject.data.materials.append(mat)



def create_landscape(FloorNoise=1.2, texture_dir_path=None, surface_size=None):
    # Create a plane of Size X, Y, Z
    if surface_size is None:
        surface_size = SURFACE_SIZE
    [mesh_size_x, mesh_size_y, mesh_size_z] = [surface_size, surface_size, 0] # in m

    # list of acceptable noise type for terrain generation
    noise=['multi_fractal', 'hybrid_multi_fractal',\
     'hetero_terrain', 'fractal', 'shattered_hterrain',\
      'strata_hterrain', 'vl_hTerrain', 'distorted_heteroTerrain', \
      'double_multiFractal', 'rocks_noise', 'slick_rock', 'planet_noise']

    noise_type=random.choice(noise)
    noise_val=random.randint(0,100)

    bpy.ops.mesh.landscape_add(ant_terrain_name="Landscape", land_material="", water_material="", texture_block="", at_cursor=False, smooth_mesh=True, \
        tri_face=False, sphere_mesh=False, subdivision_x=512, subdivision_y=512, mesh_size=2, mesh_size_x=mesh_size_x, mesh_size_y=mesh_size_y, random_seed=noise_val, \
        noise_offset_x=0, noise_offset_y=0, noise_offset_z=1, noise_size_x=1, noise_size_y=1, noise_size_z=2, noise_size=3, noise_type=noise_type, \
        basis_type='BLENDER', vl_basis_type='VORONOI_F1', distortion=1.5, hard_noise='1', noise_depth=8, amplitude=1.47, frequency=1.71, dimension=FloorNoise,\
        lacunarity=2, offset=1, gain=1, marble_bias='1', marble_sharp='5', marble_shape='3', height=1, height_invert=False, height_offset=0, fx_mixfactor=0, \
        fx_mix_mode='0', fx_type='0', fx_bias='0', fx_turb=0, fx_depth=0, fx_amplitude=0.5, fx_frequency=1.5, fx_size=1, fx_loc_x=0, fx_loc_y=0, fx_height=0.5,\
        fx_invert=False, fx_offset=0, edge_falloff='0', falloff_x=4, falloff_y=4, edge_level=0, maximum=5, minimum=-0.5, vert_group="", strata=5, strata_type='0',\
        water_plane=False, water_level=0.01, remove_double=False, show_main_settings=True, show_noise_settings=True, show_displace_settings=True, refresh=True, auto_refresh=True)
    bpy.context.active_object.name = 'Landscape'

    # scale up the created landscape and then apply texture
    # bpy.context.object.scale[0] = 10
    # bpy.context.object.scale[1] = 10
    # bpy.context.object.scale[2] = 10


    PassiveObject = bpy.context.view_layer.objects.active
    mat = bpy.data.materials.new(name='Texture')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')

    if texture_dir_path is not None \
    and os.path.exists(texture_dir_path) \
    and len(os.listdir(texture_dir_path)):
        # randomly select a texture from texture dir
        texture_path = texture_dir_path + "//" + random.choice(os.listdir(texture_dir_path))
        if not os.path.exists(texture_path):
            print("CHECK RELATIVE PATH AGAIN")
            return True
        print(texture_path)
        texImage.image = bpy.data.images.load(filepath=texture_path)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        apply_texture(PassiveObject, mat)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.smart_project()

    bpy.ops.object.mode_set(mode='OBJECT')

    # # set terrain rigidbody to Passive

    bpy.ops.rigidbody.object_add(type='PASSIVE')

    # Low Poly
    # bpy.ops.object.modifier_add(type='DECIMATE')
    # bpy.ops.object.modifier_set_active(modifier="Decimate")
    # bpy.context.object.modifiers["Decimate"].ratio =1
    bpy.context.object.rigid_body.collision_shape = 'MESH'




def add_oyster(model_dir_path=None,texture_dir_path=None, n_clusters=5, min_oyster=5, max_oyster=None, x_range=5,y_range=5):

    # if surface_size is None:
    #     surface_size = SURFACE_SIZE

    if model_dir_path is None or not os.path.exists(model_dir_path):
        print("MODELS NOT FOUND")
        return


    cal_n_oysters = True
    if max_oyster is None:
        n_oyster = min_oyster
        cal_n_oysters = False

    # calculate cluster offset values
    cluster_offset_x=x_range*0.05
    cluster_offset_y=y_range*0.05

    # list of -1 and 1 to choose sign for cluster offset
    signs=[-1,1,1,-1,-1,1,-1]

    # list of mesh names in model_dir_path
    mesh_names=os.listdir(model_dir_path)


    # list of textures in texture_dir_path
    if texture_dir_path is None:
        texture_names = []
    else:
        texture_names=os.listdir(texture_dir_path)
    pass_idx=1
    for i in range(n_clusters):

        if cal_n_oysters:
            n_oyster = random.choice(range(min_oyster, max_oyster))

        cluster_mesh_names = [random.choice(mesh_names) for i in range(n_oyster)]

        # Set center of cluster around which oysters will be dispersed
        cluster_center=[(random.random()*2-1)*x_range*.50 + random.choice(signs)*cluster_offset_x,(random.random()*2-1)*y_range*0.50+random.choice(signs)*cluster_offset_y]

        # Boundary condition in x axis
        if cluster_center[0] > x_range*0.45:

            cluster_center[0]  = x_range*0.45
        elif cluster_center[0] < -x_range*0.45:
            cluster_center[0]  = -x_range*0.45

        # Boundary condition in y axis
        if cluster_center[1] > y_range*0.45:
            cluster_center[1]  = y_range*0.45
        elif cluster_center[1] < -y_range*0.45:
            cluster_center[1]  = -y_range*0.45
        print("cluster_center:",cluster_center)
        # Variation in coordinates within a cluster
        var_x=x_range*0.7
        var_y=y_range*0.7

        # Z is sequentially incremented for oyster within a cluster
        z_val=0.2



        for mesh_name in cluster_mesh_names:
            # z_val+=.05
            oyster_file_path=model_dir_path + "\\" + mesh_name
            bpy.ops.import_mesh.stl(filepath=oyster_file_path)
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')

            # Low Poly
            # bpy.ops.object.modifier_add(type='DECIMATE')
            # bpy.ops.object.modifier_set_active(modifier="Decimate")
            # bpy.context.object.modifiers["Decimate"].ratio = 0.005


            # Set oyster scales
            bpy.context.object.scale[0] = random.uniform(0.15, 0.20)
            bpy.context.object.scale[1] = random.uniform(0.15, 0.20)
            bpy.context.object.scale[2] = random.uniform(0.5, 0.6)


            # Set oyster location in x and y randomly
            rn=random.random()
            bpy.context.object.location.x=rn*var_x+cluster_center[0]
            rn=random.random()
            bpy.context.object.location.y=rn*var_y+cluster_center[1]
            bpy.context.object.location.z=z_val

            [Roll, Pitch, Yaw] = [(random.randint(-180, 180)) * pi / 180 for x in range(3)]
            bpy.ops.transform.rotate(value=Roll, orient_axis='X')
            bpy.ops.transform.rotate(value=Pitch, orient_axis='Y')
            bpy.ops.transform.rotate(value=Yaw, orient_axis='Z')

            # Set pass index of object for creating masks - compositing
            bpy.context.object.pass_index = pass_idx
            pass_idx+=1

            # Applying rigit body dynamics
            bpy.ops.rigidbody.object_add(type='ACTIVE')
            # Set mass
            bpy.context.object.rigid_body.mass = 20
            # bpy.context.object.rigid_body.collision_shape = 'MESH'


            # Apply texture and Smart UV project
            current_Object = bpy.context.view_layer.objects.active
            mat = bpy.data.materials.new(name='Texture')
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]
            texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')

            if len(texture_names):
                # randomly select one texture file and apply it
                texPath=texture_dir_path+'\\'+random.choice(texture_names)
                texImage.image = bpy.data.images.load(filepath=texPath)
                mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
                apply_texture(current_Object, mat)
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.uv.smart_project()


            bpy.ops.object.mode_set(mode='OBJECT')

            #Deselect all
            bpy.ops.object.select_all(action='DESELECT')

def delete_oysters():
#    objs = [obj for obj in bpy.context.scene.objects if obj.name.startswith("oyster")]
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("oyster"):
            obj.select_set(True)
            bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')

def add_water(surface_size,depth):
     # Make world color blue
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.383, 0.262, 1)

    # Add water cube, scale it to surface size and depth
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0),\
     scale=(5, 5, 1))

    water_object=bpy.context.view_layer.objects.active
    mat = bpy.data.materials.new(name='Volume')
    mat.use_nodes = True
    volume = mat.node_tree.nodes.get('Principled Volume')

#    water_object.materials.
    water_object.data.materials[0] = mat


if __name__ == '__main__':

    pass

    # # delete all previously created objects in the scene
    # delete_objs()


    # # set arguments for landscape
    # floor_noise=3.5
    # landscape_texture_dir = r"..//data//blender_data//landscape//textures//"
    # create_landscape(floor_noise,landscape_texture_dir)

    # # set arguments for oysters
    # n_clusters=1
    # min_oyster=5
    # max_oyster=None
    # oyster_model_dir = r"..//data//blender_data//oysters//model//"
    # oyster_texture_dir = r"..//data//blender_data//oysters//textures//"
    # add_oyster(oyster_model_dir,oyster_texture_dir,n_clusters,min_oyster, max_oyster)

    # # Create camera
    # set_camera()
