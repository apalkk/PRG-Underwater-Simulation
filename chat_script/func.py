import os
import openai
import re
import math 
import mathutils
import bpy

openai.api_key = "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" # API KEY (expired)

# Dict contains the code names of objects in the simulation followed by what they are
object_dict = {
    'BlueROV' : 'The main robot in our simulation'
}

# ---------++ Defining pre-built functions for use by ChatGPT -->

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

def set_roll(roll_degrees):
    """
    Set the roll angle of the bot in the simulation.

    Args:
        object_name (str): The name of the object
        roll_degrees (float): The roll angle in degrees

    Returns:
        None
    """
    # Convert degrees to radians
    roll = math.radians(roll_degrees)
    
    # Get the object
    obj = bpy.data.objects['BlueROV']
    
    # Set the Euler angles of the object
    obj.rotation_euler[0] = roll

def set_pitch(pitch_degrees):
    """
    Set the pitch angle of the bot in the simulation.

    Args:
        object_name (str): The name of the object
        pitch_degrees (float): The pitch angle in degrees

    Returns:
        None
    """
    # Convert degrees to radians
    pitch = math.radians(pitch_degrees)
    
    # Get the object
    obj = bpy.data.objects['BlueROV']
    
    # Set the Euler angles of the object
    obj.rotation_euler[1] = pitch

def set_yaw(yaw_degrees):
    """
    Set the yaw angle of bot in the simulation.

    Args:
        object_name (str): The name of the object
        yaw_degrees (float): The yaw angle in degrees

    Returns:
        None
    """
    # Convert degrees to radians
    yaw = math.radians(yaw_degrees)
    
    # Get the object
    obj = bpy.data.objects['BlueROV']
    
    # Set the Euler angles of the object
    obj.rotation_euler[2] = yaw


# -++ Fundamental functions | These are functions ChatGPT isn not aware of by default
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

def add_object(object_name, object_description, points):
    """
    Adds an object in the simulation to the coordinates provided.

    Args:
        object_name (str): The name of the object
        points (tuple): The x, y, and z components of the object's motion

    Returns:
        None
    """

    object_dict.update({object_name:object_description})
    # Incomplete method !!!!!!!!!!!!
    


# _________________________________________________________________________
# Function to be used by the script are defined below

def ask(chat_history,prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]


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
