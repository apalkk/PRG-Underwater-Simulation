import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import cv2
from matplotlib import pyplot as plt
import trimesh
from trimesh import Scene
from trimesh.transformations import rotation_matrix, translation_matrix, concatenate_matrices
import bpy

# Function to load and scale the mesh
def load_and_scale_mesh(mesh_path, target_bounding_box_size=1.0):
    mesh = trimesh.load_mesh(mesh_path)
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    bounding_box = mesh.bounding_box.extents
    max_extent = np.max(bounding_box)
    scale_factor = target_bounding_box_size / max_extent
    mesh.apply_scale(scale_factor)
    return mesh

def export_to_stl(filename):
    try:
        bpy.ops.export_mesh.stl(filepath=filename, use_selection=True)
        print(f"Exported to: {filename}")
    except Exception as e:
        print(f"Error exporting to stl: {e}")
        return None
    return filename

def camera_as_planes(scene, obj):
    """
    Return planes in world-space which represent the camera view bounds.
    """
    from mathutils.geometry import normal

    camera = obj.data
    # normalize to ignore camera scale
    matrix = obj.matrix_world.normalized()
    frame = [matrix @ v for v in camera.view_frame(scene=scene)]
    origin = matrix.to_translation()

    planes = []
    from mathutils import Vector
    is_persp = (camera.type != 'ORTHO')
    for i in range(4):
        # find the 3rd point to define the planes direction
        if is_persp:
            frame_other = origin
        else:
            frame_other = frame[i] + matrix.col[2].xyz

        n = normal(frame_other, frame[i - 1], frame[i])
        d = -n.dot(frame_other)
        planes.append((n, d))

    if not is_persp:
        # add a 5th plane to ignore objects behind the view
        n = normal(frame[0], frame[1], frame[2])
        d = -n.dot(origin)
        planes.append((n, d))

    return planes


def side_of_plane(p, v):
    return p[0].dot(v) + p[1]


def is_segment_in_planes(p1, p2, planes):
    dp = p2 - p1

    p1_fac = 0.0
    p2_fac = 1.0

    for p in planes:
        div = dp.dot(p[0])
        if div != 0.0:
            t = -side_of_plane(p, p1)
            if div > 0.0:
                # clip p1 lower bounds
                if t >= div:
                    return False
                if t > 0.0:
                    fac = (t / div)
                    p1_fac = max(fac, p1_fac)
                    if p1_fac > p2_fac:
                        return False
            elif div < 0.0:
                # clip p2 upper bounds
                if t > 0.0:
                    return False
                if t > div:
                    fac = (t / div)
                    p2_fac = min(fac, p2_fac)
                    if p1_fac > p2_fac:
                        return False

    ## If we want the points
    # p1_clip = p1.lerp(p2, p1_fac)
    # p2_clip = p1.lerp(p2, p2_fac)        
    return True


def point_in_object(obj, pt):
    xs = [v[0] for v in obj.bound_box]
    ys = [v[1] for v in obj.bound_box]
    zs = [v[2] for v in obj.bound_box]
    pt = obj.matrix_world.inverted() @ pt
    return (min(xs) <= pt.x <= max(xs) and
            min(ys) <= pt.y <= max(ys) and
            min(zs) <= pt.z <= max(zs))


def object_in_planes(obj, planes):
    from mathutils import Vector

    matrix = obj.matrix_world
    box = [matrix @ Vector(v) for v in obj.bound_box]
    for v in box:
        if all(side_of_plane(p, v) > 0.0 for p in planes):
            # one point was in all planes
            return True

    # possible one of our edges intersects
    edges = ((0, 1), (0, 3), (0, 4), (1, 2),
             (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 7), (5, 6), (6, 7))
    if any(is_segment_in_planes(box[e[0]], box[e[1]], planes)
           for e in edges):
        return True


    return False


def objects_in_planes(objects, planes, origin):
    """
    Return all objects which are inside (even partially) all planes.
    """
    return [obj for obj in objects
            if point_in_object(obj, origin) or
               object_in_planes(obj, planes)]

def select_objects_in_camera():
    from bpy import context
    scene = context.scene
    origin = scene.camera.matrix_world.to_translation()
    planes = camera_as_planes(scene, scene.camera)
    objects_in_view = objects_in_planes(scene.objects, planes, origin)

    for obj in objects_in_view:
        obj.select_set(True)

# Function to generate rays
def gen_rays_at_sonar_for_proj(pose, azi_range, azi_bins, ele_range, pp_arc, **kwargs):
    azis = torch.linspace(azi_range[0], azi_range[1], azi_bins)
    eles = torch.linspace(ele_range[0], ele_range[1], pp_arc)
    pixels_theta, pixels_phi = torch.meshgrid(azis, eles, indexing="ij")
    xs = torch.cos(pixels_phi) * torch.cos(pixels_theta)
    ys = torch.cos(pixels_phi) * torch.sin(pixels_theta)
    zs = torch.sin(pixels_phi)
    p = torch.stack([xs, ys, zs], axis=-1)
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdims=True)
    rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[..., None]).squeeze()
    rays_o = pose[None, :3, 3].expand(rays_v.shape)
    return rays_o.reshape(-1, 3).detach().cpu().numpy(), rays_v.reshape(-1, 3).detach().cpu().numpy()

# Function to project hitpoints
def sonar_project_hitpoints(locations, index_tri, index_ray, mesh, origin, rays_d, cam_info):
    H, W = cam_info["rad_bins"], cam_info["azi_bins"]
    near, far = cam_info["rad_range"]
    pp_arc = cam_info["pp_arc"]
    dists = np.linalg.norm(locations - origin, axis=-1)
    r_res = (far - near) / H
    r_binned = ((dists - near) / r_res).astype(int)
    a_binned = (index_ray / pp_arc).astype(int)

    shading = -(mesh.face_normals[index_tri] * rays_d[index_ray]).sum(axis=-1)

    total = np.zeros((H, W))
    counts = np.zeros((H, W))
    render_rt = np.zeros((H, W))
    rad_img = np.zeros((H, W))

    if len(r_binned) > 0 and len(a_binned) > 0:
        rad_img[r_binned, a_binned] += dists
        np.add.at(total, (r_binned, a_binned), shading)
        np.add.at(counts, (r_binned, a_binned), 1)
        render_rt[counts > 0] = total[counts > 0] / counts[counts > 0]
        render_rt[counts > 0] = render_rt[counts > 0] / rad_img[counts > 0]
    render_rt_out = (render_rt * 255).astype(np.uint8)

    return render_rt_out

# Function to render sonar image
def render_sonar_image(pose, cam_info, mesh):
    ray_origins, ray_directions = gen_rays_at_sonar_for_proj(pose, **cam_info)
    print(f"Ray origins shape: {ray_origins.shape}, Ray directions shape: {ray_directions.shape}")
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )

    if locations.shape[0] == 0:
        print("No intersections found between rays and mesh.")
        return None

    origin = pose[:3, -1].cpu().numpy()
    render_rt = sonar_project_hitpoints(locations, index_tri, index_ray, mesh, origin, ray_directions, cam_info)
    return render_rt

# Function to get the camera pose matrix as a float32 torch tensor
def get_camera_pose_matrix(scene, angle, camera_translation):
    camera_rotation = rotation_matrix(angle, [0, 0, 1])
    camera_transform = concatenate_matrices(camera_translation, camera_rotation)
    scene.camera.transform = camera_transform
    camera_pose = np.linalg.inv(camera_transform)
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = camera_pose[:3, :3]
    pose_matrix[:3, 3] = camera_pose[:3, 3]
    pose_tensor = torch.tensor(pose_matrix, dtype=torch.float32)
    return pose_tensor

def visualize_scene(mesh, ray_origins, ray_directions, camera_pose):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    vertices = mesh.vertices
    faces = mesh.faces
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1],
                    faces, vertices[:, 2], color='cyan', alpha=0.5)

    # Plot the camera
    cam_pos = camera_pose[:3, 3]
    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2],
               color='red', s=100, label='Camera')

    # Plot the rays
    sample_indices = np.random.choice(
        len(ray_origins), size=1000, replace=False)
    sampled_origins = ray_origins[sample_indices]
    sampled_directions = ray_directions[sample_indices]
    for origin, direction in zip(sampled_origins, sampled_directions):
        ax.plot([origin[0], origin[0] + direction[0]],
                [origin[1], origin[1] + direction[1]],
                [origin[2], origin[2] + direction[2]], color='orange', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def sonar_pipeline(camera_distance = 2.0, cam_name="BlueROV", mesh_path="/Users/aadipalnitkar/PRG-Underwater-Simulation/code/untitled.stl"):
    # Example mesh path (replace with your actual mesh file)
    mesh = load_and_scale_mesh(mesh_path)
    # Create a scene and add the mesh to it
    scene = Scene()
    scene.add_geometry(mesh)

    cam = bpy.data.objects[cam_name]
    camera_translation = translation_matrix([camera_distance, 0, 0])
    scene.camera.transform = camera_translation

    # Calculate vertical field of view (vfov) based on camera parameters

    sensor_height = cam.location.z
    focal_length = np.float32(cam.data.lens)  # mm (assumed) f_in_mm = np.float32(cam.lens) 36
    vfov = 2 * np.arctan(sensor_height / (2 * focal_length))
    vfov_degrees = np.degrees(vfov)

    ele_min = np.radians(-vfov_degrees / 2)
    ele_max = np.radians(vfov_degrees / 2)

    cam_info = dict(
        azi_range=[-np.pi / 6, np.pi / 6],
        azi_bins=1024,  # W
        rad_range=[0.01, 3.0],  # Updated range
        rad_bins=17889,  # H
        ele_range=[ele_min, ele_max],
        pp_arc=512,
    )

    # Get the pose matrix at a specific angle (e.g., 0 radians)
    angle = 0
    #pose_matrix = get_camera_pose_matrix(scene, angle, camera_translation)
    pose_matrix = torch.tensor(cam.matrix_world, dtype=torch.float32)
    # Print the pose matrix
    print("Camera Pose Matrix:\n", pose_matrix)

    # Render the sonar image
    render_rt = render_sonar_image(pose_matrix, cam_info, mesh)
    if render_rt is not None:
        cv2.imwrite("/Users/aadipalnitkar/PRG-Underwater-Simulation/code/sonar_img.png", render_rt)
    else:
        print("Rendering failed due to no intersections.")
