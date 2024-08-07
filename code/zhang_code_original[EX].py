import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import cv2
import trimesh
import json
import os
from models.testing import convert_pose
import pickle

from pyhocon import ConfigFactory
import pyhocon.converter as cvt
import argparse
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def load_neus_cams(cam_path):
    """
    Args:
        cam_path: path to cameras_sphere.npz file
    Returns:
        poses: (n, 3) torch tensor of unit-sphere (relative to scene) scaled camera poses 
    """
    poses = []
    processed_poses = np.load(cam_path)
    for k in processed_poses.keys():
        if "camera_mat_inv" in k: 
            camera_mat_inv = processed_poses[k] 
        if "world_mat" in k and "world_mat_inv" not in k:
            world_mat = processed_poses[k]
            poses.append(np.linalg.inv(camera_mat_inv @ world_mat))
    poses = np.array(poses)

    scale_mat = processed_poses["scale_mat_0"]
    scale = scale_mat[0, 0]
    offset = scale_mat[:3, 3]   
    poses[:, :3, -1] = (poses[:, :3, -1] - offset) / scale 
    poses = torch.from_numpy(poses).cuda() 
    print(scale, offset)
    return poses, scale, offset

def gen_rays_at_sonar_for_proj(
    pose, azi_range, azi_bins, ele_range, pp_arc, **kwargs
):
    """
    Returns:
        rays_o: (azi_bins*pp_arc, 3), numpy array
        rays_d: (azi_bins*pp_arc, 3), numpy array
    """
    azis = torch.linspace(azi_range[0], azi_range[1], azi_bins)
    eles = torch.linspace(ele_range[0], ele_range[1], pp_arc)  
    pixels_theta, pixels_phi = torch.meshgrid(
        azis, eles, indexing="ij"  # careful with indexing here
    )  # azi_bins, pp_arc
    xs = -torch.cos(pixels_phi) * torch.sin(pixels_theta) 
    ys = -torch.sin(pixels_phi)  # sign?
    zs = torch.cos(pixels_phi) * torch.cos(pixels_theta) 
    p = torch.stack([xs, ys, zs], axis=-1)

    rays_v = p / torch.linalg.norm(
        p, ord=2, dim=-1, keepdims=True
    )  # azi_bins, pp_arc, 3
    rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[..., None]).squeeze()
    rays_o = pose[None, :3, 3].expand(
        rays_v.shape
    )  # azi_bins, pp_arc, 3
    return rays_o.reshape(-1, 3).detach().cpu().numpy(), rays_v.reshape(-1, 3).detach().cpu().numpy()


def gen_rays_at_sonar(
    pose, azi_range, azi_bins, rad_range, rad_bins, ele_range, pp_arc
):
    azis = torch.linspace(azi_range[0], azi_range[1], azi_bins)
    rads = torch.linspace(rad_range[0], rad_range[1], rad_bins)
    eles = torch.linspace(ele_range[0], ele_range[1], pp_arc)  
    pixels_r, pixels_theta, pixels_phi = torch.meshgrid(
        rads, azis, eles, indexing="ij"  # careful with indexing here
    )  # rad_bins, azi_bins, pp_arc

    # swap around axes to make consistent with opencv camera conventions
    # how to visualize: align angles in spherical coordinates diagram with
    # sonar expectation for how angles are defined
    # below shares code with gen_random_rays_at_sonar?
    xs = -pixels_r * torch.cos(pixels_phi) * torch.sin(pixels_theta) 
    ys = -pixels_r * torch.sin(pixels_phi)  # sign?
    zs = pixels_r * torch.cos(pixels_phi) * torch.cos(pixels_theta) 

    p = torch.stack([xs, ys, zs], axis=-1)

    rays_v = p / torch.linalg.norm(
        p, ord=2, dim=-1, keepdims=True
    )  # rad_bins, azi_bins, pp_arc, 3
    rays_v = torch.matmul(pose[None, None, None, :3, :3], rays_v[..., None]).squeeze()
    rays_o = pose[None, None, :3, 3].expand(
        rays_v.shape
    )  # rad_bins, azi_bins, pp_arc, 3
    far = pixels_r[..., None]  # rad_bins, azi_bins, pp_arc, 1
    near = torch.ones_like(far) * rad_range[0]  # inefficient?
    ray_info = {
        "rays_o": rays_o.reshape(rad_bins*azi_bins, pp_arc, 3),
        "rays_v": rays_v.reshape(rad_bins*azi_bins, pp_arc, 3),
        "near": near.reshape(rad_bins*azi_bins, pp_arc, 1),
        "far": far.reshape(rad_bins*azi_bins, pp_arc, 1),
    }
    return ray_info

def cvt_neus_coords_to_ho_coords(pts):
    """
    args:
        pts: (..., 3)
    """
    original_shape = pts.shape 
    pts = pts.reshape(-1, 3) 
    cmat = np.array(
        [[0.0, 0.0, 1.0], 
        [-1.0, 0.0, 0.0], 
        [0.0, -1.0, 0.0]])
    out = pts @ cmat.T
    return out.reshape(original_shape)

def cvt_blender_to_neus_coords(pts):
    original_shape = pts.shape 
    pts = pts.reshape(-1, 3) 
    cmat = np.array(
        [[1.0, 0.0, 0.0], 
        [0.0, 0.0, -1.0], 
        [0.0, 1.0, 0.0]])
    out = pts @ cmat.T
    return out.reshape(original_shape)

def cvt_ho_coords_to_neus(pts):
    """
    args:
        pts: (..., 3)
    """
    original_shape = pts.shape 
    pts = pts.reshape(-1, 3) 
    cmat = np.array(
        [[0.0, 0.0, 1.0], 
        [-1.0, 0.0, 0.0], 
        [0.0, -1.0, 0.0]])
    out = pts @ cmat
    return out.reshape(original_shape)

def cvt_neus_mat_to_ho_mat(mat):
    """
    args:
        mat: (3, 3)
    """
    cmat = np.array(
        [[0.0, 0.0, 1.0], 
        [-1.0, 0.0, 0.0], 
        [0.0, -1.0, 0.0]])
    out = cmat @ mat @ cmat.T
    return out

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vfov", type=float)
    parser.add_argument("--gt_mesh_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--H", type=int)
    parser.add_argument("--W", type=int)
    # parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--out_dir", type=str)
    # parser.add_argument("--name", type=str)
    args = parser.parse_args()
    return args    

# TODO: should have made this a function instead of have this weird command line parsing stuff
#       (cruft leftover from multiprocessing speedup experiments) 

def generate_sonar_image_data_from_args(args):
    gt_mesh_path = args.gt_mesh_path
    train_data_path = args.train_data_path    
    out_dir = args.out_dir
    # mesh_name = args.name
    H, W = args.H, args.W
    vfov = args.vfov
    # ckpt_path = args.ckpt_path

    ele_min = math.radians(-vfov/2)
    ele_max = math.radians(vfov/2)
    cam_info = dict(
        azi_range=[-np.pi / 6.0, np.pi / 6.0],
        azi_bins=W,  # ~ batch_size, assuming rad is y-axis
        rad_range=[0.01, 4.0],
        rad_bins=H,
        ele_range=[ele_min, ele_max],
        pp_arc=1024,
    )
    # generate_sonar_image_data(gt_mesh_path, train_data_path, out_dir, mesh_name, H, W, vfov, ckpt_path)
    render_sonar_images(train_data_path, cam_info, out_dir, gt_mesh_path)


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

def generate_sonar_image_data(gt_mesh_path, 
                              train_data_path, 
                              out_dir, 
                              mesh_name, 
                              H, W, vfov, ckpt_path, fix_configs_only=False):
    dataset_name = train_data_path.split("/")[-1]
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/Data", exist_ok=True)
    os.makedirs(f"{out_dir}/imgs", exist_ok=True)

    ele_min = math.radians(-vfov/2)
    ele_max = math.radians(vfov/2)
    cam_info = dict(
        azi_range=[-np.pi / 6.0, np.pi / 6.0],
        azi_bins=W,  # ~ batch_size, assuming rad is y-axis
        rad_range=[0.01, 3.3],
        rad_bins=H,
        ele_range=[ele_min, ele_max],
        pp_arc=1024,
    )

    # handle configs
    # make holoocean config
    config_json_template_path = "/vulcanscratch/kzhang/data/sonar-rgb-neus/neusis/14deg_planeFull/Config.json"
    with open(config_json_template_path, 'r') as f:
        cfg = json.load(f)

    for agents in cfg["agents"][0]["sensors"]:
        if agents["sensor_type"] != "ImagingSonar": continue
        agents["configuration"]["Azimuth"] = math.degrees(2 * cam_info["azi_range"][1])
        agents["configuration"]["Elevation"] = math.degrees(2 * cam_info["ele_range"][1])
        agents["configuration"]["RangeMin"] = cam_info["rad_range"][0]
        agents["configuration"]["RangeMax"] = cam_info["rad_range"][1]
        agents["configuration"]["RangeBins"] = H 
        agents["configuration"]["AzimuthBins"] = W

    cfg["ds_factor"] = 1.0
    cfg["simple_denoise"] = False
    with open(f"{out_dir}/Config.json", "w") as f:
        f.write(json.dumps(cfg, indent=4))

    # TODO: collapse below 4 into one function
        
    # make neusis config 
    neusis_conf_dir = "/vulcanscratch/kzhang/data/sonar-rgb-neus/batch_simulated_data/sim_data_server/sonar_only_confs"
    conf_fname = "/nfshomes/kzhang/old_home/kzhang/research/neusis/confs/template.conf"
    with open(conf_fname) as f: 
        conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)
    conf.put("conf.dataset", out_dir)
    conf.put("conf.expID", f"{ckpt_path}/sonar") 
    new_conf_text = cvt.HOCONConverter.to_hocon(conf)
    with open(f"{neusis_conf_dir}/{mesh_name}_{dataset_name}.conf", "w") as f:
        f.write(new_conf_text)

    # make neusis joint conf
    neusis_joint_conf_dir = "/vulcanscratch/kzhang/data/sonar-rgb-neus/batch_simulated_data/sim_data_server/sonar_joint_confs"
    joint_conf_fname = "/nfshomes/kzhang/old_home/kzhang/research/neusis/confs/joint/joint_template.conf"
    with open(joint_conf_fname) as f: 
        conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)
    conf.put("conf.dataset", out_dir)
    conf.put("conf.expID", f"{ckpt_path}/joint") 
    new_conf_text = cvt.HOCONConverter.to_hocon(conf)
    with open(f"{neusis_joint_conf_dir}/{mesh_name}_{dataset_name}_joint.conf", "w") as f:
        f.write(new_conf_text)

    # make neus joint conf 
    neus_joint_conf_dir = "/vulcanscratch/kzhang/data/sonar-rgb-neus/batch_simulated_data/sim_data_server/rgb_joint_confs"
    joint_conf_fname = "/nfshomes/kzhang/old_home/kzhang/research/neusis/NeuS/confs/joint/joint_template.conf"
    with open(joint_conf_fname) as f: 
        conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)
    conf.put("general.base_exp_dir", f"{ckpt_path}/joint")
    conf.put("dataset.data_dir", train_data_path)
    new_conf_text = cvt.HOCONConverter.to_hocon(conf)
    with open(f"{neus_joint_conf_dir}/{mesh_name}_{dataset_name}_joint.conf", "w") as f:
        f.write(new_conf_text)
    
    # make neus conf 
    neus_conf_dir = "/vulcanscratch/kzhang/data/sonar-rgb-neus/batch_simulated_data/sim_data_server/rgb_only_confs"
    conf_fname = "/nfshomes/kzhang/old_home/kzhang/research/neusis/NeuS/confs/wmask.conf"
    with open(conf_fname) as f: 
        conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)
    conf.put("general.base_exp_dir", f"{ckpt_path}/rgb")
    conf.put("dataset.data_dir", train_data_path)
    new_conf_text = cvt.HOCONConverter.to_hocon(conf)
    with open(f"{neus_conf_dir}/{mesh_name}_{dataset_name}.conf", "w") as f:
        f.write(new_conf_text)

    if fix_configs_only:
        return
    
    render_sonar_images(train_data_path, cam_info, out_dir, gt_mesh_path)


def render_sonar_images(train_data_path, cam_info, out_dir, gt_mesh_path):
    mesh = trimesh.load(gt_mesh_path)
    mesh = as_mesh(mesh)
    # mesh.vertices = cvt_ho_coords_to_neus(mesh.vertices)
    mesh.vertices = cvt_blender_to_neus_coords(mesh.vertices)

    poses, scale, offset = load_neus_cams(f"{train_data_path}/cameras_sphere.npz")
    mesh.vertices = (mesh.vertices - offset) / scale # didn't matter before when these were just dummy values
                                                     # but now it does with most recent data

    for i in range(len(poses)):
        pose = poses[i] 
        # rotate sonar by 90 degrees, about z (optical) axis 
        rot = torch.tensor([[0.0, -1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0]])
        pose[:3, :3] = pose[:3, :3] @ rot
        render_rt = render_sonar_image(pose, cam_info, mesh)
        print(render_rt.sum())
        if render_rt.sum() > 10:
            cv2.imwrite(f"{out_dir}/imgs/{i:0>4d}.png", render_rt)

            curr_data = {}
            curr_pose = convert_pose(pose.cpu().numpy(), "neus_to_ho")
            curr_data["PoseSensor"] = curr_pose
            curr_data["ImagingSonar"] = render_rt
            with open(f"{out_dir}/Data/{i:03d}.pkl", "wb") as f:
                pickle.dump(curr_data, f)
    

def sonar_project_hitpoints(locations, 
                            index_tri, 
                            index_ray, 
                            mesh, 
                            origin, 
                            rays_d,
                            cam_info,):
    """
    Args: 
        origin: (1, 3) np.ndarray 
    """
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

    rad_img[r_binned, a_binned] += dists

    # mask[r_binned, a_binned] = 1
    np.add.at(total, (r_binned, a_binned), shading)
    np.add.at(counts, (r_binned, a_binned), 1)
    # remember to average over the number of rays per bin? 
    render_rt[counts > 0] = total[counts > 0] / counts[counts > 0]
    render_rt[counts > 0] = render_rt[counts > 0] / rad_img[counts > 0]
    render_rt_out = (render_rt*255).astype(np.uint8)

    return render_rt_out

def render_sonar_image(pose, cam_info, mesh):

    # below has issues with multiprocessing
    ray_origins, ray_directions = gen_rays_at_sonar_for_proj(pose, **cam_info)  # rad_bins, azi_bins, pp_arc, 3
    
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False) # (n, 3), (n,), (n,)
    
    origin = pose[:3, -1].cpu().numpy()
    render_rt = sonar_project_hitpoints(locations, 
                                        index_tri, 
                                        index_ray, mesh, origin, ray_directions, cam_info)
    # print(render_rt.sum())
    return render_rt


def get_visible_points(mesh, pose):
    """
    Args:
        mesh: trimesh.Trimesh
        pose: (4, 4) np.ndarray
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    t = pose[:3, 3:4].T
    ray_origins = np.tile(t, (len(mesh.vertices), 1))
    ray_directions = mesh.vertices - ray_origins
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1, keepdims=True) 
    locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False) # (n, 3), (n,), (n,)          
    locations_matched = np.zeros_like(mesh.vertices)
    locations_matched[index_ray] = locations
    distances = np.linalg.norm(locations_matched - mesh.vertices, axis=1)
    visible_points = locations_matched[distances < 1e-6]
    return visible_points 

if __name__ == "__main__":
    args = make_argument_parser()
    generate_sonar_image_data_from_args(args)
