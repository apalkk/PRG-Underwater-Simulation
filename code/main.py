import bpy
import numpy as np
import os
import sys

path = r"/Users/aadipalnitkar/PRG-Underwater-Simulation/code"

if path not in sys.path:
    sys.path.append(path)

from CreateScene import delete_objs, create_landscape, add_bluerov, add_oyster, set_camera, set_light
from Utils import get_position, render_img, save_values#, save_plots
from ImuUtils import  cal_linear_acc, cal_angular_vel, acc_gen, gyro_gen, accel_high_accuracy, gyro_high_accuracy, vib_from_env, cal_imu_step
from RangeScanner import run_scanner, tupleToArray
#import range_scanner
from Simulate import set_motion # !!! Change back to simulate if it does not work !!!


START_FRAME = 1
END_FRAME =15

SURFACE_SIZE = 20

TIME_TO_WAIT = 1  # [EXPERIMENTAL VALUE] wait for this many frames for oysters to settle down properly

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
    set_light(0, 0, 30, 1000)
    
    # create a random landscape everytime
    create_landscape(floor_noise, landscape_texture_dir)
    
    # import blueROV 3d model 
    front_cam, bottom_cam = add_bluerov(bluerov_path, bluerov_location)

    # bpy context object
    context = bpy.context

    # scanner object for the rotating LiDAR
    scanner_object = context.scene.objects[front_cam]
    
    # import oysters at some random location according to cluster size
    add_oyster(oysters_model_dir,oysters_texture_dir, n_clusters, min_oyster, max_oyster)
    
    # set motion path in keyframes
    set_motion('BlueROV', motion_path)

    # add third person view camera to the scene
    third_cam_x = 5.72
    third_cam_y = 5.25
    third_cam_z = 22.2
    third_cam_r = 0.34
    third_cam_p = 0
    third_cam_yaw = 2.35  
    third_cam,_ = set_camera(third_cam_x, third_cam_y, third_cam_z, third_cam_r, third_cam_p, third_cam_yaw)
    
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

    
    for frame_count in range(END_FRAME):
        context.scene.frame_set(frame_count)

        if frame_count >= TIME_TO_WAIT:  # assuming all objects settle down

            x, y, z, rot_x, rot_y, rot_z = get_position('BlueROV')
            
            x_array.append(x)
            y_array.append(y)
            z_array.append(z)
            roll_array.append(rot_x)
            pitch_array.append(rot_y)
            yaw_array.append(rot_z)
            save_values(position_dir, POSITIONS_FILENAME,[x,y,z,rot_x,rot_y,rot_z])

            if frame_count >= TIME_TO_WAIT+3:
                print(len(x_array))
                # calculate true accelerometer values
                true_accel = cal_linear_acc(x_array, y_array, z_array, 30)

                # calculate true gyroscope values
                true_gyro = cal_angular_vel(roll_array, pitch_array, yaw_array, 30)

                # calculate simulated accelerometer values from true values
                simulated_accel = acc_gen(IMU_RATE, true_accel, ACC_ERROR, ACC_VIB)

                # calculate simulated gyroscope values from true values
                simulated_gyro = gyro_gen(IMU_RATE, true_gyro, GYRO_ERROR, GYRO_VIB)

                # array has 3 elements in it, remove the first element
                x_array.pop(0)
                y_array.pop(0)
                z_array.pop(0)
                roll_array.pop(0)
                pitch_array.pop(0)
                yaw_array.pop(0)

                # save the simulated values in a text file
                if save_imu:
                    data_2_write = [simulated_accel, simulated_gyro]
                    save_values(imu_dir, IMU_FILENAME, data_2_write)

                # plot simulated accelerometer and simulated gyro for each point of time

                if frame_count % FRAME_SKIP == 0:
                    scan_values = run_scanner(context, scanner_object)
                    mapped_data = np.array(list(map(lambda hit: tupleToArray(hit), scan_values))).transpose()

                    x_coordinates = mapped_data[2]
                    y_coordinates = mapped_data[3]
                    z_coordinates = mapped_data[4]
                    distances     = mapped_data[5]
                    # intensities   = mappedData[6]  # we do not need intensities values currently

                    # save the values in a text file
                    if save_scanner:
                        data_2_write = [x_coordinates, y_coordinates, z_coordinates, distances]
                        save_values(scanner_dir, SCANNER_FILENAME, data_2_write)
                        save_plots(scanner_dir, str(frame_count)+".png", data_2_write, [x, y, z])
                #     # plot the coordinates, distances and intensities for each point of time

            # save only some frames depending on the imu rate and frame rate
            if frame_count % IMU_STEP_SIZE == 0:
                # save RGB and DEPTH images
                render_img(bottom_cam_dir, frame_count, bottom_cam, save_both=True)
                render_img(front_cam_dir, frame_count, front_cam, save_both=True)
                render_img(third_cam_dir, frame_count, third_cam, save_both=True)

            # [OPTIONAL] - display the image continously with opencv
            # [OPTIONAL] - run yolo-oyster detection on the rendered img


if __name__=="__main__":
    
    # register range_scanner module
    # range_scanner.register()

    # absolute path of the script
    script_path = os.path.dirname(os.path.abspath(__file__))
    print("script_path:",script_path)

    # remove the last dir from path so that we are in base directory and can navigate further
    base_dir_path = script_path.split('code')[0]
    print("base_dir_path:",base_dir_path)

    try:
        # delete all previously created objects from the scene
        delete_objs()
        
        # landscape parameters
        floor_noise = 3.5  # seabed smoothens out as the floor_noise is increased
        landscape_texture_dir = base_dir_path + "//data//blender_data//landscape//textures//"
        
        # blueRov parameters, initial position and orientation
        bluerov_model_path = base_dir_path + "//data//blender_data//blueROV//BlueRov2.dae"
        bluerov_location = (-0.85, -0.65, 3.45)
        bluerov_orientation = (1.57, 0, 1.57)
        
        # oysters paramteres
        oysters_model_dir = base_dir_path + "//data//blender_data//oysters//model//"
        oysters_texture_dir = base_dir_path + "//data//blender_data//oysters//textures//"
        n_clusters = 1
        min_oyster = 1
        max_oyster = None
    
        # dir where all the results will be saved
        out_dir = base_dir_path + "//data//output//"
        
        # bluerov motion path
        motion_path = {
            0+TIME_TO_WAIT: [bluerov_location, bluerov_orientation],
            80+TIME_TO_WAIT: [(bluerov_location[0]+4.5, bluerov_location[1], bluerov_location[2]),
            (bluerov_orientation[0], bluerov_orientation[1], bluerov_orientation[2])],
            100+TIME_TO_WAIT: [(bluerov_location[0]+5, bluerov_location[1], bluerov_location[2]),
            (bluerov_orientation[0], bluerov_orientation[1]+0.2, bluerov_orientation[2]+1.57)],
            180+TIME_TO_WAIT: [(bluerov_location[0]+4.5, bluerov_location[1]+2.3, bluerov_location[2]+0.7),
            (bluerov_orientation[0], bluerov_orientation[1]+0.1, bluerov_orientation[2]+1.57)],
            200+TIME_TO_WAIT: [(bluerov_location[0]+4, bluerov_location[1]+2.8, bluerov_location[2]+1),
            (bluerov_orientation[0], bluerov_orientation[1]+0.1, bluerov_orientation[2]+2.8)],
            300+TIME_TO_WAIT: [(bluerov_location[0], bluerov_location[1]+5, bluerov_location[2]),
            (bluerov_orientation[0], bluerov_orientation[1], bluerov_orientation[2]+2.8)]
                       }

        # start everything
        start_pipeline(floor_noise, landscape_texture_dir,
                       bluerov_model_path, bluerov_location,
                       oysters_model_dir, oysters_texture_dir, n_clusters, min_oyster, max_oyster,
                       out_dir, motion_path,save_imu=True, save_scanner=True)
        print("Done")
    except Exception as e:
        print(e)

    # unregister range_scanner module
    #range_scanner.unregister()
