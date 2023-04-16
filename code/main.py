import bpy
import numpy as np
import os
import time
import sys
import requests
import json
import re
import math
# Local Imports:
from CreateScene import delete_objs, create_landscape, add_bluerov, add_oyster, set_camera, set_light
from Utils import get_position, render_img, save_values
from ImuUtils import  cal_linear_acc, cal_angular_vel, acc_gen, gyro_gen, accel_high_accuracy, gyro_high_accuracy, vib_from_env, cal_imu_step
from RangeScanner import run_scanner, tupleToArray
#import range_scanner
from simulate import set_motion

key = "<API-KEY>"
url = "https://api.openai.com/v1/chat/completions"

INPUT_MODE = True
FRAME_INTERVAL = 8
END_FRAME = 1000 #9160
TIME_TO_WAIT =  0 #[EXPERIMENTAL VALUE] wait for this many frames for oysters to settle down properly

object_dict = {
    'BlueROV' : 'The main robot in our simulation'
}

prompt = """Output code in python that achieves the desired goal.

Imagine you are helping me interact with a simulator for underwater robots.

The simulator consists a robot underwater, along with several objects. Apart from the bot, none of the objects are movable.
Within the code, we have the following commands available to us. You are not to use any other hypothetical functions. Assume the bot is at (0,0,0) cartesian coordinates.

When I ask you to do something, please give me Python code that is needed to achieve that task.
You should only use the following functions that I have defined for you. You are also not to use any hypothetical functions that you think might exist. You should only use the functions that I have defined for you.
You can use simple Python functions from libraries such as math and numpy.

set_bot_position(points): Takes a tuple as input indicating the X,Y and Z coordinates you want the bot to move to.

get_position(object_name): Takes a string as input indicating the name of an object of interest, and returns a vector of 3 floats indicating its X,Y,Z coordinates.

get_bot_position(): Get the current XYZ coordinates of the drone.

set_yaw(angle): Set the yaw angle of the drone (in degrees).

set_pitch(angle): Set the pitch angle of the drone (in degrees).

set_roll(angle): Set the roll angle of the drone (in degrees).

put_oyster(): Adds oysters to the simulation.

put_bot(): Adds a new bot to the simulation.

In terms of axis conventions, forward means positive X axis. Right means positive Y axis. Up means positive Z axis.

Are you ready?"""

instructions = ["Move the bot up bu 10 points"]

chat_history = [
    {
        "role": "system",
        "content": prompt
    }
]

set_counter = 0

C = bpy.context

# Enable Post Processing and Use Nodes
C.scene.render.use_compositing = True
C.scene.use_nodes = True

path = r"/Users/aadipalnitkar/Underwater-share/code"

if path not in sys.path:
    sys.path.append(path)
                                

START_FRAME = 0
CURR_FRAME = 0
DEG_2_RAD = np.pi/180.0
SURFACE_SIZE = 80

IMU_RATE = 120
FRAME_RATE = 30
IMU_STEP_SIZE = cal_imu_step(IMU_RATE, FRAME_RATE)
FRAME_SKIP = 3 # range scanner will run only on every 3rd frame

ACC_ERROR = accel_high_accuracy
GYRO_ERROR = gyro_high_accuracy

# sets random vibration to accel with RMS for x/y/z axis - 1/2/3 m/s^2, can be zero or changed to other values
ACC_ENV = '[0.003 0.001 0.01]-random'
ACC_VIB = vib_from_env(ACC_ENV, IMU_RATE)


# sets sinusoidal vibration to gyro with frequency being 0.5 Hz and amp for x/y/z axis being 6/5/4 deg/s
GYRO_ENV = '[6 5 4]d-0.5Hz-sinusoidal'
GYRO_VIB = vib_from_env(GYRO_ENV, IMU_RATE)


# imu output file name
IMU_FILENAME = "imu_values.txt"

# range scanner output file name
SCANNER_FILENAME = "scanner_values.txt"

# range scanner output file name
POSITIONS_FILENAME = "position_values.txt"

def start_pipeline(floor_noise,landscape_texture_dir,bluerov_path,bluerov_location,oysters_model_dir,oysters_texture_dir,\
             n_clusters, min_oyster, max_oyster,out_dir, motion_path, save_imu=False, save_scanner=False):

    global instruct
    instruct = 0
    global set_counter
    set_counter = 0
    
    # if output dir not present, make one
    if not os.path.exists(out_dir):                                                                                                                                                                         
        os.makedirs(out_dir)

    # if render output dir not present, make one
    render_out_dir = os.path.join(out_dir, "render_output")
    if not os.path.exists(render_out_dir):
        os.makedirs(render_out_dir)
    
    # if front camera render output dir not present, make one
    front_cam_dir = os.path.join(render_out_dir, "front_cam")
    if not os.path.exists(front_cam_dir):
        os.makedirs(front_cam_dir)
        
    # if third person camera render output dir not present, make one
    third_cam_dir = os.path.join(render_out_dir, "third_cam")
    if not os.path.exists(third_cam_dir):
        os.makedirs(third_cam_dir)
    
    # if bottom facing camera render output dir not present, make one
    bottom_cam_dir = os.path.join(render_out_dir, "bottom_cam")
    if not os.path.exists(bottom_cam_dir):
        os.makedirs(bottom_cam_dir)

    # if imu output dir not present, make one
    imu_dir = os.path.join(out_dir, "imu_dir")
    if not os.path.exists(imu_dir):
        os.makedirs(imu_dir)

    # if scanner output dir not present, make one
    scanner_dir = os.path.join(out_dir, "scanner_dir")
    if not os.path.exists(scanner_dir):
        os.makedirs(scanner_dir)
    
    # if positions output dir not present, make one
    position_dir = os.path.join(out_dir, "position_dir")
    if not os.path.exists(position_dir):
        os.makedirs(position_dir)


    # set point source light
#    set_light(0, 0, 10, 10000)
    
    # create a random landscape everytime
#    create_landscape(floor_noise, landscape_texture_dir, SURFACE_SIZE)
    
    # import blueROV 3d model
    
    print("Adding bluerove")
    front_cam, bottom_cam = add_bluerov(bluerov_path, bluerov_location, front_cam_orientation=(-20, 180, 0))
    print("bluerov addedd")
    
    # bpy context object
    context = bpy.context

    # scanner object for the rotating LiDAR
    scanner_object = context.scene.objects[front_cam]
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    
    #bpy.context.scene.world.mist_settings.depth = 45
    bpy.context.scene.world.mist_settings.falloff = 'QUADRATIC'

#    
    # import oysters at some random location according to cluster size
#    add_oyster(oysters_model_dir,oysters_texture_dir, n_clusters, min_oyster, max_oyster, 5)
    
    
    # set motion path in keyframes
    set_motion('BlueROV', motion_path)

    # add third person view camera to the scene
    third_cam_x = 20
    third_cam_y = 6
    third_cam_z = 3
    third_cam_r = 0.34
    third_cam_p = 0
    third_cam_yaw = 2.35 
    third_cam,_ = set_camera(third_cam_x, third_cam_y, third_cam_z, third_cam_r, third_cam_p, third_cam_yaw, True, focal_length=100)
    
    
    # create variables to store values
    x_array     = []
    y_array     = []
    z_array     = []
    roll_array  = []
    pitch_array = []
    yaw_array   = []
    simulated_accel = None
    simulated_gyro = None

    x_coordinates = None
    y_coordinates = None
    z_coordinates = None
    distances     = None
    intensities   = None
    scene = bpy.context.scene
    
    ###BFAB80 HPL color
    ####0e92b8  NBRF color
    #### -------------------main loop----------------------
    for frame_count in range(START_FRAME, END_FRAME+1, FRAME_INTERVAL):
        CURR_FRAME = frame_count
        if set_counter > 0 : set_counter -= 1
#        TIME_TO_WAIT=3
#        for wait_count in range(TIME_TO_WAIT):
#            bpy.context.scene.frame_set(wait_count)
        if(frame_count > FRAME_INTERVAL and set_counter == 0):
            if(INPUT_MODE):
             i = input("Give a command")
             try:
                 print(i)
             except:
                 print("WARNING - Blank Input, Not Sending To GPT")
             instructions.append(i)
            if(len(instructions) > instruct):
                try:
                 string = ask(chat_history,instructions[instruct])
                except:
                 raise Exception("Error with API key")
                print(string)
                try:
                 exec(extract_python_code(string))
                except Exception as e:
                 print("WARNING : Possible GPT code code block error")
                 print(e)
                 try:
                     exec(string)
                 except Exception as e:
                     print("WARNING : GPT - Code could not be executed")
                     print(e)
                
                instruct += 1
        print("frame: ",frame_count)
        print(get_bot_position())
        bpy.context.scene.frame_set(frame_count)
        for scene in bpy.data.scenes:
            for node in scene.node_tree.nodes:
                if node.type == 'OUTPUT_FILE':
                    node.base_path = os.path.join(third_cam_dir,str(frame_count)+"_masks")
#                    if os.path.exists(node.base_path):
#                        print('path exists')
#                    else:
#                        print('path created')
#                        os.mkdir(node.base_path)
        

#        x, y, z, rot_x, rot_y, rot_z = get_position('BlueROV')
#        
#        x_array.append(x)
#        y_array.append(y)
#        z_array.append(z)
#        roll_array.append(rot_x)
#        pitch_array.append(rot_y)
#        yaw_array.append(rot_z)
#        save_values(position_dir, POSITIONS_FILENAME, [[x], [y], [z], [rot_x], [rot_y], [rot_z]])
#        
#        
#        if frame_count >= START_FRAME + 2:
#            # calculate true accelerometer values
#            true_accel = cal_linear_acc(x_array, y_array, z_array, 30)
#           

#            # calculate true gyroscope values
#            true_gyro = cal_angular_vel(roll_array, pitch_array, yaw_array, 30)
#        

#            # calculate simulated accelerometer values from true values
#            simulated_accel = acc_gen(IMU_RATE, true_accel, ACC_ERROR, ACC_VIB)
#            
#            # calculate simulated gyroscope values from true values
#            simulated_gyro = gyro_gen(IMU_RATE, true_gyro, GYRO_ERROR, GYRO_VIB)
#         

#            # array has 3 elements in it, remove the first element
#            x_array.pop(0)
#            y_array.pop(0)
#            z_array.pop(0)
#            roll_array.pop(0)
#            pitch_array.pop(0)
#            yaw_array.pop(0)

#            # save the simulated values in a text file
#            if save_imu:
#                data_2_write = [simulated_accel, simulated_gyro]
#                save_values(imu_dir, IMU_FILENAME, data_2_write)

                # plot simulated accelerometer and simulated gyro for each point of time

    #            if frame_count % FRAME_SKIP == 0:
    #                print("Started Range Scanner")
    #                start = time.time()
    #                scan_values = run_scanner(context, scanner_object, fov=120)
    #                print("scan complete")
    #                mapped_data = np.array(list(map(lambda hit: tupleToArray(hit), scan_values))).transpose()
    #                if len(mapped_data):
    #                    x_coordinates = [x_ - x for x_ in mapped_data[2]]
    #                    y_coordinates = [y_ - y for y_ in mapped_data[3]]
    #                    z_coordinates = mapped_data[4]
    #                    distances     = mapped_data[5]
    #                    # intensities   = mappedData[6]  # we do not need intensities values currently

    #                        # save the values in a text file
    #                    if save_scanner:
    #                        data_2_write = [x_coordinates, y_coordinates, z_coordinates, distances]
    #                        save_values(scanner_dir, SCANNER_FILENAME, data_2_write)
    #                        save_plots(scanner_dir, str(frame_count)+".png", data_2_write, [x, y, z])
    #                print("Completed Range Scanner in: ", time.time() - start)
                
            # save only some frames depending on the imu rate and frame rate
          

            # save RGB and DEPTH images
        #render_img(bottom_cam_dir, frame_count, bottom_cam, save_RGB=True)
        #render_img(front_cam_dir, frame_count, front_cam, save_RGB=True)
        #render_img(front_cam_dir, frame_count, front_cam, save_DEPTH=True)
        #render_img(third_cam_dir, frame_count, third_cam, save_RGB=True)
        render_img(third_cam_dir, frame_count, third_cam, save_regular=True,save_RGB=False)


def generate_circular_points(radius, center, init_angle=0.0, num_points=20):

    center_x = center[0]
    center_y = center[1]

    points = []
    
    for theta in np.linspace(np.pi, 3*np.pi, num_points):
        x = radius * np.cos(theta) + center_x
        y = radius * np.sin(theta) + center_y

        theta += init_angle
        points.append([x, y, theta])

    return points

def get_bot_position():
    """
    Get the position of BlueROV in the simulation

    Returns:
        tuple: The x, y, and z coordinates of the object's position
    """
    return get_position('BlueROV')

def get_position(object_name):
    """
    Get the current position of an object in the simulation

    Args:
        object_name (str): The name of the object

    Returns:
        tuple: The x, y, and z coordinates of the object's position
    """
    obj = bpy.data.objects[object_name]
    position = obj.location
    return (position.x, position.y, position.z)

def set_bot_position(points):
    """
    Sets the motion of an object in the future by making the object move to a certian
    set of points.

    Args:
        object_name (str): The name of the object
        points (tuple): The x, y, and z components of the object's motion

    Returns:
        None
    """
    global set_counter
    obj = bpy.data.objects['BlueROV']
    set_motion('BlueROV',{(CURR_FRAME + (FRAME_INTERVAL * set_counter)):[points,obj.rotation_euler]})
    set_counter += 1

def set_yaw(angle):
    """
    Sets the motion of an object in the future by making the object move to a certian
    set of points.

    Args:
        object_name (str): The name of the object
        points (tuple): The x, y, and z components of the object's motion

    Returns:
        None
    """
    obj = bpy.data.objects['BlueROV']
    set_motion('BlueROV',{CURR_FRAME:[obj.location,(obj.rotation_euler[0],obj.rotation_euler[1],angle)]})

def set_pitch(angle):
    """
    Sets the motion of an object in the future by making the object move to a certian
    set of points.

    Args:
        object_name (str): The name of the object
        points (tuple): The x, y, and z components of the object's motion

    Returns:
        None
    """
    obj = bpy.data.objects['BlueROV']
    set_motion('BlueROV',{CURR_FRAME:[obj.location,(obj.rotation_euler[0],angle,obj.rotation_euler[2])]})

def set_roll(angle):
    """
    Sets the motion of an object in the future by making the object move to a certian
    set of points.

    Args:
        object_name (str): The name of the object
        points (tuple): The x, y, and z components of the object's motion

    Returns:
        None
    """
    obj = bpy.data.objects['BlueROV']
    set_motion('BlueROV',{CURR_FRAME:[obj.location,(angle,obj.rotation_euler[1],obj.rotation_euler[2])]})

def put_oyster():
    """
    Sets the motion of an object in the future by making the object move to a certian
    set of points.

    Args:
        object_name (str): The name of the object
        points (tuple): The x, y, and z components of the object's motion

    Returns:
        None
    """
    add_oyster(oysters_model_dir,oysters_texture_dir, n_clusters, min_oyster, max_oyster, 5)

def put_bot():
    """
    Sets the motion of an object in the future by making the object move to a certian
    set of points.

    Args:
        object_name (str): The name of the object
        points (tuple): The x, y, and z components of the object's motion

    Returns:
        None
    """
    add_bluerov(base_dir_path + "//data//blender_data//blueROV//BlueRov2.dae", bluerov_location, front_cam_orientation=(-20, 180, 0))

def ask(chat_history,prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    headers = {"Authorization": f"Bearer {key}"}
    data = {'model': 'gpt-3.5-turbo', 'messages': chat_history}
    poster = requests.post(url, headers=headers, json=data).json()['choices'][0]['message']
    #print(poster['content'])

    chat_history.append(
        {
            "role": "assistant",
            "content": poster['content'],
        }
    )
    return poster['content']

def extract_python_code(content):
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None



def generate_motion_path(points, end_frame, object_z = 3.5, object_roll=0.0, object_pitch=0.0):
    num_key_frames = len(points)
    motion_path = {}
    for idx, key_frame in enumerate(np.linspace(0, end_frame, num_key_frames)):
        motion_path[key_frame] = [(points[idx][0],points[idx][1], object_z), (object_roll, object_pitch, points[idx][2])]
    
    return motion_path
        
    


# ---------------------------------------------------------

if __name__=="__main__":

    # register range_scanner module
    #range_scanner.register()

    # absolute path of the script
    script_path = os.path.dirname(os.path.abspath(__file__))
    print("script_path:",script_path)

    # remove the last dir from path so that we are in base directory and can navigate further
    base_dir_path = script_path.split('code')[0]
    print("base_dir_path:",base_dir_path)
    base_dir_path  = r"/Users/aadipalnitkar/Underwater-share"

    try:
        # delete all previously created objects from the scene
#        delete_objs()
        
        # landscape parameters
        floor_noise =1  # seabed smoothens out as the floor_noise is increased
        landscape_texture_dir = base_dir_path + "//data//blender_data//landscape//textures//"
        
        # blueRov parameters, initial position and orientation
        bluerov_model_path = base_dir_path + "//data//blender_data//blueROV//BlueRov2.dae"
        bluerov_location = (0, 0, 0)
        bluerov_orientation = (0, 0, 0)
        
        # oysters paramteres
        oysters_model_dir = base_dir_path + "//data//blender_data//oysters//model//"
        oysters_texture_dir = base_dir_path + "//data//blender_data//oysters//textures//"
        n_clusters = 1
        min_oyster = 5
        max_oyster = None
    
        # dir where all the results will be saved
        out_dir = base_dir_path + "//data//final_output_3_10_23_test//"
        #all densities are 5 by default. z=10 as default
        
#        points = generate_circular_points(radius=17.5, center=[30, 16], init_angle=0.22-3.14, num_points=20)
        
#        motion_path = generate_motion_path(points, 17000, object_roll=1.57)
        
        # bluerov motion path
        motion_path = {
            0+TIME_TO_WAIT: [bluerov_location, bluerov_orientation],

            2000+TIME_TO_WAIT: [(0, 0, 0),
            (90*DEG_2_RAD, 0, 0*DEG_2_RAD)],
            
            3000+TIME_TO_WAIT: [(16.75, 4, 1.5),
            (90*DEG_2_RAD, 0, 60*DEG_2_RAD)],
            
            4000+TIME_TO_WAIT: [(20,3,1.5),
            (90*DEG_2_RAD, 0, 90*DEG_2_RAD)],
            
            5000+TIME_TO_WAIT: [(23.784,7,1.5),
            (90*DEG_2_RAD, 0, 170*DEG_2_RAD)],
            
            7000+TIME_TO_WAIT: [(23.13,21,1.5),
            (90*DEG_2_RAD, 0, 180*DEG_2_RAD)],
            
            8000+TIME_TO_WAIT: [(27.287,24.265,1.5),
            (90*DEG_2_RAD, 0, 90*DEG_2_RAD)],
            
            9000+TIME_TO_WAIT: [(32.12, 21, 1.5),
            (90*DEG_2_RAD, 0, 0*DEG_2_RAD)],
                    
            11000+TIME_TO_WAIT: [(35.43,7,1.5),
            (90*DEG_2_RAD, 0, 20*DEG_2_RAD)],
                    
            12000+TIME_TO_WAIT: [(38,4,1.5),
            (90*DEG_2_RAD, 0, 90*DEG_2_RAD)],
            
            13000+TIME_TO_WAIT: [(42,7,1.5),
            (90*DEG_2_RAD, 0, 180*DEG_2_RAD)],
            
            15000+TIME_TO_WAIT: [(40,21.54,1.5),
            (90*DEG_2_RAD, 0, 200*DEG_2_RAD)],
            
            16000+TIME_TO_WAIT: [(36.22,23.72,1.5),
            (90*DEG_2_RAD, 0, 270*DEG_2_RAD)],
            
            18500+TIME_TO_WAIT: [(19,23.184,1.5),
            (90*DEG_2_RAD, 0, 270*DEG_2_RAD)],
            
            20000+TIME_TO_WAIT: [(13.5,18.5,1.5),
            (90*DEG_2_RAD, 0, 360*DEG_2_RAD)],
                        }
        print("starting pipeline")
        # start everything
        
        start = time.time()
        start_pipeline(floor_noise, landscape_texture_dir,
                       bluerov_model_path, bluerov_location,
                       oysters_model_dir, oysters_texture_dir, n_clusters, min_oyster, max_oyster,
                       out_dir, motion_path,save_imu=True, save_scanner=True)
        print("total time: ", time.time() - start)
        print("Done - complete")
    except Exception as e:
        print(e)

    # unregister range_scanner module
    #range_scanner.unregister()
