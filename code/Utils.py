import bpy
# import csv
# import matplotlib.pyplot as plt
# import matplotlib.ticker
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.patches as mpatches
# from matplotlib.collections import PatchCollection
from math import floor, ceil
import os
import math
import re
import json
import requests
from simulate import set_motion

# import cv2

import numpy as np
import os
# import pandas as pd

EPSILON = 1e-10
MAX_INT = 1e8



def convert_2_polar(x, y, d):
    theta = np.arctan2(y, x)
    r = d

    return r, theta


# def save_plots(out_dir, out_filename, data, center_offset):

#     # create a new dir inside out_dir named "plots" and save in it
#     plot_dir = os.path.join(out_dir, "plots")
#     if not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)

#     x_coordinates = data[0]
#     y_coordinates = data[1]
#     z_coordinates = data[2]  # we dont care about the z coordinate of the scanned objects
#     distances = data[3]

#     Rs = []
#     ThetaS = []

#     for x, y, d in zip(x_coordinates, y_coordinates, distances):
#         # if x != 0:
#         #     print(x)
#         # if y != 0:
#         #     print(y)
#         # since x, y of the scanned objects are in respect to origin, we shift origin to center_offset
#         # x -= center_offset[0]
#         # y -= center_offset[1]

#         r, theta = convert_2_polar(x, y, d)
#         Rs.append(r)
#         ThetaS.append(theta)

#     plt.polar(ThetaS, Rs)
#     plt.savefig(os.path.join(plot_dir, out_filename))


def save_values(out_dir, out_filename, data):
    """
    param: out_dir
    param: data - list of lists, all lists in the data should have same size
    brief: saves the data in a text file inside the specified out dir
            if the text file is present then it would append the new data, else create a new text file and save in it
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.isfile(os.path.join(out_dir, out_filename)):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(os.path.join(out_dir, out_filename), write_mode) as file:
        for i in range(len(data[0])):
            data_to_write = ""
            for idx, item in enumerate(data):
                if idx == len(data) - 1:
                    end_char = "\n"
                else:
                    end_char = ", "
                data_to_write += str(item[i]) + end_char
            file.write(data_to_write)
    file.close()


def render_img(img_dir,keyframe, camera_name='Camera', save_RGB=True, save_DEPTH=False, save_both=False, save_regular=False, max_depth=45):

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    if save_both:
        save_RGB = True
        save_DEPTH = True

    if save_RGB:
        rgb_dir = os.path.join(img_dir, "RGB_imgs")
        if not os.path.exists(rgb_dir):
            os.makedirs(rgb_dir)
        # save rendered rgb img
        bpy.data.cameras[camera_name].dof.use_dof = False
        bpy.context.scene.camera = bpy.data.objects[camera_name]
        save_path = rgb_dir+"//"+str(keyframe)+'.png'
        r = bpy.context.scene.render
        r.resolution_x = 640
        r.resolution_y = 480
        r.filepath=save_path
        bpy.ops.render.render(write_still=True)
    if save_regular:
        # bpy.data.cameras[camera].dof.use_dof = False
        bpy.context.scene.camera = bpy.data.objects[camera_name]

        save_path = img_dir+"//"+str(keyframe)+'.jpg'
        r = bpy.context.scene.render
        r.resolution_x = 640
        r.resolution_y = 480
        r.filepath=save_path
        bpy.ops.render.render(write_still=True)


    if save_DEPTH:
        depth_dir = os.path.join(img_dir, "DEPTH_imgs")
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir)
        # save rendered depth img
        bpy.data.cameras[camera_name].dof.use_dof = False
        bpy.context.scene.camera = bpy.data.objects[camera_name]
        bpy.context.scene.view_layers["ViewLayer"].use_pass_mist = True

        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links
        for n in tree.nodes:
            tree.nodes.remove(n)
        rl = tree.nodes.new(type="CompositorNodeRLayers")
        composite = tree.nodes.new(type="CompositorNodeComposite")
        composite.location = 200, 0
        links.new(rl.outputs['Mist'], composite.inputs['Image'])
        bpy.context.scene.world.mist_settings.start = 0
        bpy.context.scene.world.mist_settings.depth = max_depth
        scene = bpy.context.scene
        save_path = depth_dir+"//"+str(keyframe)+'.png'
        scene.render.filepath=save_path
        r = bpy.context.scene.render
        r.resolution_x = 960 #640
        r.resolution_y = 720 #480
        r.filepath=save_path
        bpy.ops.render.render(write_still=True)
        bpy.context.scene.view_layers["ViewLayer"].use_pass_mist = False
        bpy.context.scene.use_nodes = False



def get_position(object_name):

    # Mention which object to track
    obj=bpy.context.scene.objects[object_name]
    x=obj.location.x
    y=obj.location.y
    z=obj.location.z

    # Specify rotation rep
    rot_z=obj.rotation_euler.z
    rot_y=obj.rotation_euler.y
    rot_x=obj.rotation_euler.x
    return x,y,z,rot_x,rot_y,rot_z


def arrange_values(r, theta, intensity):

    azimuths = np.radians(np.linspace(0, 360, 10))
    zeniths = np.arange(0, 50, 1)

    values = np.zeros((azimuths.size, zeniths.size))
    # print(values)
    # print(values.shape)

    # l = zip(r, theta, intensity)
    # l.sort()
    # r, theta, intensity = zip(*sorted(zip(r, theta, intensity)))

    # print(r)
    print('\n')
    # print(theta)
    print('\n')
    # print(intensity)


    # plt.contourf()


    # for angle, dist, val in zip(theta, r, intensity):
    #     for idx1, angle_val in enumerate(azimuths):
    #         if angle < angle_val:
    #             for idx2, zenith_vals in enumerate(zeniths):
    #                 if dist < zenith_vals:
    #                     values[idx1][idx2] = val
    #
    # dist, angle = np.meshgrid(zeniths, azimuths)
    #
    # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    # ax.contourf(angle, dist, values, 100, cmap='plasma')

    plt.show()


# def draw_polar_plot(r, theta, intensity):

#     print('max r: ', np.array(r).mean())
#     img = np.ones((510, 510, 3), np.uint8)*255
#     center_x, center_y = int(img.shape[0]/2), int(img.shape[1]/2)
#     cv2.circle(img, (center_x, center_y), 250, (150, 0, 0), -1)
#     x_min = MAX_INT
#     y_min = MAX_INT
#     x_max = -1
#     y_max = -1
#     for i in range(len(r)):
#         x = int(ceil(r[i]*np.cos(theta[i])*5 + center_x))
#         y = int(ceil(r[i]*np.sin(-theta[i])*5 + center_y))
#         intensity_val = intensity[i] * 255
#         if intensity_val > 50:
#             # print(r)
#             color = (0, 0, intensity_val)
#             cv2.circle(img, (x, y), 2, color, -1)
#             # blur = img[max(0, y-10):y+20, max(0, x-10):x+20]
#             # cv2.GaussianBlur(blur, (3, 3), 0, blur)
#             # img[max(0, y-10):y+20, max(0, x-10):x+20] = blur

#             if x <= x_min:
#                 x_min = x
#             elif x >= x_max:
#                 x_max = x

#             if y <= y_min:
#                 y_min = y
#             elif y >= y_max:
#                 y_max = y

#     # img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)

#     blur = img[y_min-5:y_max+10, x_min-5:x_max+10]
#     cv2.GaussianBlur(blur, (5, 5), 0, blur)
#     img[y_min-5:y_max+10, x_min-5:x_max+10] = blur

#     # cv2.GaussianBlur(img, (11, 11), 0, img)
#     cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
#     cv2.imshow('img', img)
#     cv2.waitKey(0)


# def read_csv(csv_file):

#     if not os.path.exists(csv_file):
#         print("FILE DOES NOT EXISTS")
#         exit()

#     file = open(csv_file)
#     csv_reader = csv.reader(file)
#     header = []
#     header = next(csv_reader)
#     # print(header)
#     prev_category_id = 0
#     first_time = True

#     x = []
#     y = []
#     intensity = []
#     distance = []
#     theta = []

#     for row in csv_reader:
#         usable_row = row[0].split(' ')
#         # print(usable_row)
#         # print(usable_row[0])
#         # print(type(float(usable_row[0])))

#         # if first_time:
#         #     prev_category_id = int(float(usable_row[0]))
#         #     first_time = False

#         # category_id = int(float(usable_row[0]))
#         # if prev_category_id != category_id:
#         #     prev_category_id = int(float(usable_row[0]))

#         # print('part ID: {}'.format(int(float(usable_row[1]))))
#         # print('category ID: {}'.format(int(float(usable_row[0]))))
#         theta.append(np.arctan2(float(usable_row[3]) + 8.8342, float(usable_row[2]) - 0.69889 + EPSILON))
#         x.append(float(usable_row[2]))
#         y.append(float(usable_row[3]))
#         # print('x: {}'.format(float(usable_row[2])))
#         # print('y: {}'.format(float(usable_row[3])))
#         # print('z: {}'.format(float(usable_row[4])))
#         # print('distance: {}'.format(float(usable_row[5])))
#         distance.append(float(usable_row[5]))
#         # print('\n')
#         # print('intensity: {}'.format(float(usable_row[6])))

#         intensity.append(float(usable_row[6]))

#     #  shifted origin to the camera position
#     # x = [val - 0.69889 for val in x]
#     # y = [val + 9.1372 for val in y]

#     # r = [min(dist, 25) for dist in distance]
#     r = distance

#     # plt.polar(theta, r, '.')
#     # plt.show()
#     #
#     # arrange_values(r, theta, intensity)
#     # exit()

#     draw_polar_plot(r, theta, intensity)
#     exit()
#     #
#     # azimuths = np.radians(np.linspace(0, 360, 10))
#     # zeniths = np.arange(0, 50, 0.1)
#     #
#     # dist, angle = np.meshgrid(zeniths, azimuths)
#     #
#     # values = np.zeros((azimuths.size, zeniths.size))
#     #
#     # for i in range(len(r)):
#     #     if values[floor(r[i])][floor(theta[i])] < intensity[i]:
#     #         values[floor(r[i])][floor(theta[i])] = intensity[i]
#     #
#     # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
#     # ax.contourf(angle, dist, values, 10, cmap='plasma')
#     #
#     # plt.show()
#     #
#     # exit()
#     #

#     # theta = []
#     # for i in range(len(x)):
#     #     theta.append(np.arctan2(y[i], x[i] + EPSILON))

#     #  working
#     # # plt.axes(projection = 'polar')
#     # plt.polar(theta, r)
#     # plt.show()
#     # exit()

#     # intensity = np.array(intensity)

#     # draw_polar_plot(r, theta, intensity)

#     # arrange_values(r, theta, intensity)
#     #
#     # exit()
#     #
#     # values = intensity.reshape(len(theta), len(r))
#     #
#     # R, Phi = np.meshgrid(r, theta)
#     # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
#     # ax.set_theta_zero_location("N")
#     # ax.set_theta_direction(-1)
#     # # autumn()
#     # cax = ax.contourf(theta, r, values, 30)
#     # # autumn()
#     # cb = fig.colorbar(cax)
#     # cb.set_label("Pixel reflectance")
#     #
#     # plt.show()


#     # I have list of theta, r, intensity
#     # df = pd.DataFrame(list(zip(r, theta, intensity)), columns=['r', 'theta', 'intensity'])
#     #
#     # ntheta = 300#len(theta)
#     # dtheta = 360/ntheta
#     # nradius = 300#len(r)
#     # dradius = max(r)/nradius
#     #
#     # colors = ['#000052', '#0c44ac', '#faf0ca', '#ed0101', '#970005']
#     # cm = LinearSegmentedColormap.from_list('custom', colors, N=10)
#     # cm.set_bad(color='white')
#     #
#     # patches = []
#     # avg_intensity = []
#     #
#     # for nr in range(nradius, 0, -1):
#     #     start_r = (nr - 1)*dradius
#     #     end_r = nr*dradius
#     #
#     #     for nt in range(0, ntheta):
#     #         start_t = nt*dtheta
#     #         end_t = (nt+1)*dtheta
#     #
#     #         stripped = df[(df['r'] >= start_r) & (df['r'] < end_r) &
#     #                       (df['theta'] >= start_t) & (df['theta'] < end_t)]
#     #
#     #         avg_intensity.append(stripped['intensity'].mean())
#     #         wedge = mpatches.Wedge(0, end_r, start_t, end_t)
#     #
#     #         patches.append(wedge)
#     #
#     # f_color = cm([(x - 0.001) / (1 - 0.001) for x in avg_intensity])
#     #
#     # # print(type(f_color))
#     # # print(len(f_color))
#     # # exit()
#     # collection = PatchCollection(patches, linewidth=0.0,
#     #                              edgecolor=['#000000' for x in avg_intensity],
#     #                              facecolor=f_color[0])
#     #
#     # fig = plt.figure(figsize=(10, 10), dpi=100, edgecolor='r', facecolor='w')
#     # ax = fig.add_subplot()
#     # ax.add_collection(collection)
#     # # Clean up the image canvas and save!
#     # plt.axis('equal')
#     # plt.axis('off')
#     # plt.tight_layout()
#     #
#     # # plt.savefig('toronto.png')
#     # plt.show()


# def demo_polar_plot():
#     # -- Generate Data -----------------------------------------
#     # Using linspace so that the endpoint of 360 is included...
#     azimuths = np.radians(np.linspace(0, 360, 10))
#     zeniths = np.arange(0, 60, 5)

#     r, theta = np.meshgrid(zeniths, azimuths)
#     print(r.shape)
#     print(theta.shape)
#     print(r)
#     print(theta)


#     values = np.random.random((azimuths.size, zeniths.size))
#     print(values.shape)
#     # -- Plot... ------------------------------------------------
#     fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
#     ax.contourf(theta, r, values, 10, cmap='plasma')

#     plt.show()


# def demo_polar_plot2():
#     """
#     https://stackoverflow.com/a/49630831
#     """
#     # data = np.array([[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
#     #                  [[0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]]])

#     data = np.array([[[1, 1, 0]]])

#     data = np.repeat(data, 25, axis=1)

#     ax = plt.subplot(111, polar=True)

#     # get coordinates:
#     phi = np.linspace(0, 2 * np.pi, data.shape[1] + 1)
#     r = np.linspace(0, 1, data.shape[0] + 1)
#     Phi, R = np.meshgrid(phi, r)
#     # get color
#     color = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

#     # plot colormesh with Phi, R as coordinates,
#     # and some 2D array of the same shape as the image, except the last dimension
#     # provide colors as `color` argument
#     m = plt.pcolormesh(Phi, R, data[:, :, 0], color=color, linewidth=0)
#     # This is necessary to let the `color` argument determine the color
#     m.set_array(None)

#     plt.show()


if __name__ == "__main__":
    pass

    # x = [11, 12, 13, 14, 15, 10, 10, 10, 10, 10]
    # y = [10, 10, 10, 10, 10, 10, 12, 13, 14, 15]
    # z = []
    # d = [10, 10, 10, 10, 10, 10, 11, 12, 13, 14, 15]
    #
    # center_offset = [0, 0, 0]
    #
    # save_plots("./temp", "temp.png", [x, y, z, d], center_offset)
#     # arrange_values(0,0,0)

#     csv_file1 = "E:\\programming\\github\\SONAR-sim\\blender\\example_scenes\\output\\general_object_sonar\\7_test_sonar_test_frame_130.csv"

#     read_csv(csv_file1)

#     # csv_file2 = "E:\\programming\\github\\SONAR-sim\\blender\\example_scenes\\output\\general_object_sonar\\14_test_sonar_test_frame_129.csv"
#     # csv_file3 = "E:\\programming\\github\\SONAR-sim\\blender\\example_scenes\\output\\general_object_sonar\\15_test_sonar_test_frame_129.csv"
#     # csv_file4 = "E:\\programming\\github\\SONAR-sim\\blender\\example_scenes\\output\\general_object_sonar\\16_test_sonar_test_frame_129.csv"
#     # csv_file5 = "E:\\programming\\github\\SONAR-sim\\blender\\example_scenes\\output\\general_object_sonar\\17_test_sonar_test_frame_129.csv"
#     # csv_file6 = "E:\\programming\\github\\SONAR-sim\\blender\\example_scenes\\output\\general_object_sonar\\18_test_sonar_test_frame_129.csv"
#     # csv_file7 = "E:\\programming\\github\\SONAR-sim\\blender\\example_scenes\\output\\general_object_sonar\\19_test_sonar_test_frame_131.csv"
#     # #
#     # #
#     # csv_files = [csv_file1, csv_file2, csv_file3, csv_file4, csv_file5, csv_file6, csv_file7]
#     # #
#     # for csv_file in csv_files:
#     #     read_csv(csv_file)
#     demo_polar_plot()
#     # demo_polar_plot2()
