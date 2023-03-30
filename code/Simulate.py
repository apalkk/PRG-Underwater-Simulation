import bpy
from math import sin,cos

def set_motion(object_name,points):
    # points is of type : {frame: (x,y,z)}
    obj=bpy.context.scene.objects[object_name]
    #  add random same noise to the location n rotation
    for frame in points:
        obj.location=points[frame][0]
        obj.rotation_euler=points[frame][1]
        obj.keyframe_insert(data_path="location",frame=frame)
        obj.keyframe_insert("rotation_euler", frame = frame)
def update_motion(object_name,frame):
    # points is of type : {frame: (x,y,z)}
    next_frame =frame + 4
    obj=bpy.context.scene.objects[object_name]
    (x,y,z) = obj.location
    (row,pitch,yaw)=obj.rotation_euler
    x+=0.2*cos(yaw)
    y+=0.2*sin(yaw)
    #yaw+=0.1
    print(x,y,z)
    print('--------------------')
    print(row,pitch,yaw)
    obj.rotation_euler=(row,pitch,yaw)
    obj.location=(x,y,z)
    #obj.rotation_euler=points[frame][1]
    obj.keyframe_insert(data_path="location",frame=next_frame)
    obj.keyframe_insert("rotation_euler", frame = next_frame)
