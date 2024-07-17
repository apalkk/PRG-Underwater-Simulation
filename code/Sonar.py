import numpy as np
import torch
import matplotlib.pyplot as plt
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

# Example mesh path (replace with your actual mesh file)
mesh_path = 'underwater_scene_for_aerial_image.obj'
mesh = load_and_scale_mesh(mesh_path)

# Create a scene and add the mesh to it
scene = Scene()
scene.add_geometry(mesh)

# Place the camera at a distance to view the entire object
camera_distance = 2.0  # Distance from the object
camera_translation = translation_matrix([camera_distance, 0, 0])
scene.camera.transform = camera_translation

# Calculate vertical field of view (vfov) based on camera parameters
sensor_height = 24.0  # mm (assumed)
focal_length = 36.0  # mm (assumed)
vfov = 2 * np.arctan(sensor_height / (2 * focal_length))
vfov_degrees = np.degrees(vfov)

ele_min = np.radians(-vfov_degrees / 2)
ele_max = np.radians(vfov_degrees / 2)

cam_info = dict(
    azi_range=[-np.pi / 6, np.pi / 6],
    azi_bins=1024,  # W
    rad_range=[0.01, 3.0],  # Updated range
    rad_bins=1024,  # H
    ele_range=[ele_min, ele_max],
    pp_arc=512,
)

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
def get_camera_pose_matrix(scene, angle):
    camera_rotation = rotation_matrix(angle, [0, 0, 1])
    camera_transform = concatenate_matrices(camera_translation, camera_rotation)
    scene.camera.transform = camera_transform
    camera_pose = np.linalg.inv(camera_transform)
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = camera_pose[:3, :3]
    pose_matrix[:3, 3] = camera_pose[:3, 3]
    pose_tensor = torch.tensor(pose_matrix, dtype=torch.float32)
    return pose_tensor

# Get the pose matrix at a specific angle (e.g., 0 radians)
angle = 0
pose_matrix = get_camera_pose_matrix(scene, angle)

# Print the pose matrix
print("Camera Pose Matrix:\n", pose_matrix)

# Render the sonar image
render_rt = render_sonar_image(pose_matrix, cam_info, mesh)
if render_rt is not None:
    print(render_rt)
    plt.imshow(render_rt, cmap='gray')
    plt.show()
else:
    print("Rendering failed due to no intersections.")
