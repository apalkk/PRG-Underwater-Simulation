# Underwater-Robotics
We are working under the Perceptions and Robotics Group (PRG) at UMD, to use OpenAI's ChatGPT for applications in robotics. We are working on creating a high level function library which can be controlled by ChatGPT and can then be used to undertake several complex tasks which would otherwise require human intervention. Currently an ongoing project.

- **Note : Add git lfs tracking in the .gitattributes file or via the terminal if python files are bigger than 50MB**
- **Note : Use this [google drive link](https://drive.google.com/drive/folders/1-kSRIKONjX89lnxEH12iVjAsx5-octjC?usp=share_link) to download the blender_data folder to avoid lfs**

## Tasks
- [x] 2D bounding Box of objects from Blender `2.93`
- [x] Integrate IMU with blender
- [x] Integrate LiDAR/SONAR with blender
- [x] Train yolo on the generated data from blender
- [ ] Rover position data with detections on PCL

## Google Colab Notebook
* colab notebook used to train the yolov4-tiny, find it [here](https://colab.research.google.com/drive/1RePfSTb7c1tPAuh_D-ySLhrG78gxkF9D?usp=sharing)
* Modified the colab notebook provided [here](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg)

## Models
* We trained a yoloV4-tiny on a dataset of around 5000 images
* Download the model best weights file from [here](https://drive.google.com/file/d/1ffx9uFeBLUgfymSTHV5pO_OoLnYB7EVT/view?usp=sharing) 
* Copy the model weights in [here](https://github.com/mjoshi07/Underwater-Robotics/tree/main/data/model)

## Blender model
* BlueROV model downlaod from [here](https://github.com/patrickelectric/bluerov_ros_playground)
* Oysters model download from [here](https://drive.google.com/drive/folders/1XY2yMnFDCiSR8H6S84OS8WX1tzu2OnCW?usp=sharing)  

## Proposed Functions For the API Library
**The api library functions are written in ```chat_script/func.py```file**
* get_bot_position() - Returns position of the robot in the form of a tuple containing x,y,z coordinates called points.
* get_position(obj_name) - Returns position in the form of points of any object whose name is passed to the function.
* set_bot_motion(points) - Moves the robot to those set of points at a certain time in the future.
* set_yaw(angle) - Sets the yaw angle for the bot.
* set_pitch(angle) - Sets the pitch angle for the bot.
* set_roll(angle) - Sets the roll angle for the bot.

## Getting Strated
* First create a blender terminal to display cli output and errors via (this)[https://blender.stackexchange.com/questions/265793/launch-system-console-blender-3-0]. You can also create a desktop shortcut on mac if you would like.
* Once blender is started go to File>Open and open the .blend file in the project.
## Common Errors
### Cannot install bpy on python
Each version of blender supports a corresponding version of python. Look [here](https://pypi.org/project/bpy/#description) for more info.
### Editing modules but no changes to the project
Go to Edit>Preferences>Paths and then go to the data subsection where you will see a scripts path. Make sure that corresponds with your working directory.
### Module not found error on importing modules
Blender searches through a list of paths when looking for modules to laod which you can find with ```import sys```and```print(sys.path)```. You can either place your modules on those paths or use <br> 
```
spec = importlib.util.spec_from_file_location("settings", "/Users/aadipalnitkar/Underwater-share/code/settings.py")
```
```
foo = importlib.util.module_from_spec(spec)
```
```
sys.modules["settings"] = foo
```
```
spec.loader.exec_module(foo)
```
```
settings : dict = foo.settings
```
### Editing files outside of blender but no changes to blender files
You must restart blender to let the edits outside of blender take effect.
