import math
from Simulate import set_motion, update_motion
import mathutils
import bpy

# Dict contains the code names of objects in the simulation followed by what they are
object_dict = {
    'BlueROV' : 'The main robot in our simulation'
}

# Defining pre-built functions for use by ChatGPT -->

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

def set_bot_motion(points):
    """
    Sets the motion of an object in the future by making the object move to a certian
    set of points.

    Args:
        points (tuple): The x, y, and z components of the object's motion

    Returns:
        None
    """
    set_motion('BlueROV',points)

def set_roll(object_name, roll_degrees):
    """
    Set the roll angle of the object in Blender

    Args:
        object_name (str): The name of the object
        roll_degrees (float): The roll angle in degrees

    Returns:
        None
    """
    # Convert degrees to radians
    roll = math.radians(roll_degrees)
    
    # Get the object
    obj = bpy.data.objects[object_name]
    
    # Set the Euler angles of the object
    obj.rotation_euler[0] = roll

def set_pitch(object_name, pitch_degrees):
    """
    Set the pitch angle of the object in Blender

    Args:
        object_name (str): The name of the object
        pitch_degrees (float): The pitch angle in degrees

    Returns:
        None
    """
    # Convert degrees to radians
    pitch = math.radians(pitch_degrees)
    
    # Get the object
    obj = bpy.data.objects[object_name]
    
    # Set the Euler angles of the object
    obj.rotation_euler[1] = pitch

def set_yaw(object_name, yaw_degrees):
    """
    Set the yaw angle of the object in Blender

    Args:
        object_name (str): The name of the object
        yaw_degrees (float): The yaw angle in degrees

    Returns:
        None
    """
    # Convert degrees to radians
    yaw = math.radians(yaw_degrees)
    
    # Get the object
    obj = bpy.data.objects[object_name]
    
    # Set the Euler angles of the object
    obj.rotation_euler[2] = yaw



# ______________________________________________________________
# -- Fundamental functions | These are functions ChatGPT isn not aware of by default
def set_euler(object_name, roll_degrees, pitch_degrees, yaw_degrees):
    """
    Set the euler angles of the object in Blender

    Args:
        object_name (str): The name of the object
        roll_degrees (float): The roll angle in degrees
        pitch_degrees (float): The pitch angle in degrees
        yaw_degrees (float): The yaw angle in degrees

    Returns:
        None
    """
    # Convert degrees to radians
    roll = math.radians(roll_degrees)
    pitch = math.radians(pitch_degrees)
    yaw = math.radians(yaw_degrees)
    
    # Get the object
    obj = bpy.data.objects[object_name]
    
    # Set the Euler angles of the object
    obj.rotation_euler = (roll, pitch, yaw)

def set_motion(object_name, points):
    """
    Sets the motion of an object in the future by making the object move to a certian
    set of points.

    Args:
        object_name (str): The name of the object
        points (tuple): The x, y, and z components of the object's motion

    Returns:
        None
    """
    obj = bpy.data.objects[object_name]
    obj.location = obj.location + mathutils.Vector(points)
