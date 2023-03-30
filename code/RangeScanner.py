import numpy as np
import bpy
import sys
import importlib
import time


path = r"..//code//"

if path not in sys.path:
    sys.path.append(path)

import range_scanner
from CreateScene import create_landscape


def run_scanner(context, scannerObject, fov=60):    # Velodyne LiDAR
    return range_scanner.ui.user_interface.scan_rotating(
        context, 
        scannerObject=scannerObject,
        xStepDegree=1, fovX=fov, yStepDegree=1, fovY=2.5, rotationsPerSecond=20,

        reflectivityLower=0.0, distanceLower=0.0, reflectivityUpper=0.0, distanceUpper=99999.9, maxReflectionDepth=10,
        
        enableAnimation=False, frameStart=1, frameEnd=2, frameStep=1, frameRate=1,

        addNoise=False, noiseType='gaussian', mu=0.0, sigma=0.01, noiseAbsoluteOffset=0.0, noiseRelativeOffset=0.0, 

        simulateRain=False, rainfallRate=0.0, 

        addMesh=True,

        #  removed all export capabilites from the code
        exportLAS=False, exportHDF=False, exportCSV=False, exportSingleFrames=False,
        dataFilePath="//output", dataFileName="output_file",
        
        debugLines=False, debugOutput=False, outputProgress=False, measureTime=False, singleRay=False, destinationObject=None, targetObject=None
    )  


def tupleToArray(hit):
    """
    hit[0]  -> categoryID
    hit[1]  -> partID
    hit[2]  -> x location absolute value in world frame
    hit[3]  -> y location absolute value in world frame
    hit[4]  -> z location absolute value in world frame
    hit[5]  -> distance from the center of scanner
    hit[6]  -> intensity
    hit[7]  -> red color
    hit[8]  -> green color
    hit[9]  -> blue color
    """
    # categoryID, partID;X;Y;Z;distance;intensity;red;green;blue;")
    return np.array([
        hit.categoryID, hit.partID,                                             # 0, 1   
        hit.location.x, hit.location.y, hit.location.z,                         # 2, 3, 4
        hit.distance,                                                           # 5
        hit.intensity,                                                          # 6
        hit.color[0], hit.color[1], hit.color[2],                               # 7, 8, 9
    ])

if __name__ == "__main__":
    range_scanner.register()
    
    try:
        context = bpy.context
        scanner_object = context.scene.objects["Camera"]
        floor_noise = 3.5
        base_dir_path = ""
        landscape_texture_dir = base_dir_path + "/data/blender_data/landscape/textures//"
        create_landscape(floor_noise, landscape_texture_dir)
        # print(bpy.context.area)
        scan_values = run_scanner(context, scanner_object)
        mappedData = np.array(list(map(lambda hit: tupleToArray(hit), scan_values))).transpose()

        obj_count = 1
        first_time = True
        part_id = None

        x_coordinates = mappedData[2]
        y_coordinates = mappedData[3]
        z_coordinates = mappedData[4]
        distances     = mappedData[5]
        

        # print(obj_count)
    except Exception as e:
        print(e)
        
    range_scanner.unregister()
    
