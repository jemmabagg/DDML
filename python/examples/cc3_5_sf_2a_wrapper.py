"""pcFM + Diffusion model wrapper 
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/baggjemm/CaloCloudi")
from src.evaluation import inference

torch.set_num_threads(1)

n_layers = 78

#pcFM Model Path
pcfm_path = os.path.expanduser("/home/baggjemm/PointCountFM_private/results/20260817_133705_TEMPO_PandT_/compiled.pt") #Get the right pcfm model
pcfm = torch.jit.load(pcfm_path, map_location="cpu").eval()

#CaloCloud Model
cc_path = os.path.expanduser("/home/baggjemm/CaloCloudi/data/logs/2026_08_17__15_19_32/checkpoints/2026-08-18_14-35-58_model.pt")
sampler = inference.Sampler.from_model_path(cc_path)

'''#Loading in config file
with open("/home/baggjemm/CaloCloudi/data/logs/2026_08_17__15_19_32/config.yaml") as stream:
    config = yaml.safe_load(stream)

def _sample_cloud(cond, num_points): #num_points is the total number of points wanted for the event
    """
    Function to call the CaloClouds .pt and return the raw cloud in Data coords, columns (x,y,z,energy)
    """
    max_points = int(np.max(num_points))
    cond_t = torch.from_numpy(cond).to(torch.get_default_dtype())
    cloud = _cc(cond_t, max_points)

    return np.asarray(cloud.cpu().numpy()).reshape(1, max_points, 4)

##Sample to physical function, unshift_points taken from CC inference.py
##Trips the generated cloud to pcFM's counts, and maps into detector coordinates + layer indices
def sample_to_physical(points, points_per_layer, config):
    physical_points = np.zeros_like(points)
    point_layer_ids = -np.ones(points.shape[:2], dtype=int)

    total_points_requested = points_per_layer.sum(1)
    point_energy = points[:, :, 3]
    order_by_energy = np.argsort(np.argsort(point_energy, axis=1))
    num_to_remove = np.clip(points.shape[1] - total_points_requested, 0, None)
    remove_mask = order_by_energy < num_to_remove[:, None]

    physical_points[~remove_mask, 3] = points[~remove_mask, 3]

    beyond_detector = 10 * np.max(points[:, :, 2])
    points[:, :, 2][remove_mask] = beyond_detector

    layer_centers = get_layer_centers(config, coordinates="detector")
    points_by_height = np.argsort(np.argsort(points[:, :, 2], axis=1), axis=1)

    n_events, n_layers = points_per_layer.shape
    cumulative_points_per_layer = np.concatenate(
        (np.zeros((n_events, 1), dtype=int), np.cumsum(points_per_layer, axis=1)),
        axis=-1,
    ).astype(int)
    for layer in range(n_layers):
        layer_mask = (points_by_height >= cumulative_points_per_layer[:, [layer]]) & (
            points_by_height < cumulative_points_per_layer[:, [layer + 1]]
        )
        physical_points[layer_mask, 1] = layer_centers[layer]
        point_layer_ids[layer_mask] = layer

    data_low_x = config["data"]["Xmin"]
    detector_low_z = config["data"]["Zmin_in_detector"]
    data_x_range = config["data"]["Xmax"] - data_low_x
    detector_z_range = config["data"]["Zmax_in_detector"] - detector_low_z
    scale_0 = detector_z_range / data_x_range
    physical_points[~remove_mask, 2] = (
        points[~remove_mask, 0] - data_low_x
    ) * scale_0 + detector_low_z

    data_low_y = config["data"]["Ymin"]
    detector_low_x = config["data"]["Xmin_in_detector"]
    data_y_range = config["data"]["Ymax"] - data_low_y
    detector_x_range = config["data"]["Xmax_in_detector"] - detector_low_x
    scale_1 = detector_x_range / data_y_range
    physical_points[~remove_mask, 0] = (
        points[~remove_mask, 1] - data_low_y
    ) * scale_1 + detector_low_x

    physical_points[remove_mask] = 0

    return physical_points, point_layer_ids

#Corrects for the shower not entering perpendicular
def unshift_points(physical_points, point_layer_ids, cond_data_coords, config):
    # defensive programming
    n_events = physical_points.shape[0]
    assert cond_data_coords.shape[0] == n_events
    assert point_layer_ids.shape[0] == n_events
    # cond -> (e, x, y, z) in data
    # cond -> (e, z, x, y) in physical
    direction_vectors = cond_data_coords[:, 1:4]
    normalised_direction_vectors = direction_vectors / np.linalg.norm(
        direction_vectors, axis=1, keepdims=True
    )
    layer_centers = get_layer_centers(config, coordinates="detector")
    layer_centers -= layer_centers[0]
    detetor_x = normalised_direction_vectors[:, 1]
    detetor_z = normalised_direction_vectors[:, 0]
    x_shift_per_layer = layer_centers[:, None] * detetor_x[None, :]
    z_shift_per_layer = layer_centers[:, None] * detetor_z[None, :]

    n_events = physical_points.shape[0]
    real_points = (physical_points[:, :, 3] > 0) & (point_layer_ids >= 0)

    x_shifts = x_shift_per_layer[point_layer_ids, np.arange(n_events)[:, None]]
    physical_points[real_points, 0] += x_shifts[real_points]

    z_shifts = z_shift_per_layer[point_layer_ids, np.arange(n_events)[:, None]]
    physical_points[real_points, 2] += z_shifts[real_points]

    physical_points[~real_points] = 0

    return physical_points

def detector_cell_thickness(config):
    layer_bottom_pos = config["detector"]["layer_bottom_pos"]
    cell_thickness_ecal = config["detector"]["cell_thickness_ecal"]
    cell_thickness_hcal = config["detector"]["cell_thickness_hcal"]
    hcal_start = config["detector"]["hcal_start"]

    cell_thickness = np.full(len(layer_bottom_pos), cell_thickness_ecal, dtype=float)
    cell_thickness[hcal_start:] = cell_thickness_hcal

    return cell_thickness


def get_layer_centers(config, coordinates="data"):
    if coordinates == "data":
        layer_bottom_pos = config["data"]["layer_bottom_pos"]
        cell_thickness = config["data"]["cell_thickness"]
    elif coordinates == "detector":
        layer_bottom_pos = config["detector"]["layer_bottom_pos"]
        cell_thickness = detector_cell_thickness(config)

    layer_bottom_pos = np.array(layer_bottom_pos)
    layer_centers = layer_bottom_pos + cell_thickness / 2

    return layer_centers'''


@torch.inference_mode()
def run_inference(inputs):
    #make the torch dtype float32
    dtype = torch.get_default_dtype()

    #unpack (one shower per call)
    energy = torch.from_numpy(inputs[0]).to(dtype)
    theta_deg = torch.from_numpy(inputs[1]).to(dtype)
    phi_deg = torch.from_numpy(inputs[2]).to(dtype)

    #direction unit vector
    #DDML gives degrees, but torch.sin/cos want radians
    theta = torch.deg2rad(theta_deg)
    phi = torch.deg2rad(phi_deg)
    dx = torch.sin(theta) * torch.cos(phi)
    dy = torch.sin(theta) * torch.sin(phi)
    dz = torch.cos(theta)
    direction = torch.stack([dx, dy, dz], dim=1)

    #cond for pcFM: [e, dx, dy, dz]
    cond = torch.cat([energy.unsqueeze(-1), direction], dim=-1)

    ##pcFM per layer
    ## we get counts per layer and energy per layer from pcFM

    raw = pcfm(cond)
    if raw[-1].shape[1] != n_layers:
        raw = raw[:-1]
    pcfm_out = torch.cat(list(raw), dim=1)

    # hallucination guard (from inference_cond_file.py _check_hallucinations):
    # re-sample if per-layer counts blow up, BEFORE clamping. Without this one
    # bad shower makes total_points huge and can OOM the sampler.
    while pcfm_out[:, :n_layers].max() > 1e4 or pcfm_out[:, :n_layers].min() < -1e4:
        raw = pcfm(cond)
        if raw[-1].shape[1] != n_layers:
            raw = raw[:-1]
        pcfm_out = torch.cat(list(raw), dim=1)

    #post processing (from pcFM)
    pcfm_out = torch.clamp(pcfm_out, min=0.0) # clamp >= 0
    counts_t = (pcfm_out[:, :n_layers] + 0.5).to(torch.int32) #counts via (x + 0.5) -> int
    energy_t = pcfm_out[:, n_layers:2 * n_layers].clone() 
    energy_t[counts_t == 0] = 0.0 #zero the energy per layer where count == 0

    counts_int = counts_t.cpu().numpy().ravel().astype(np.int64)   # (78,)
    energy_per_layer = energy_t.cpu().numpy().ravel()              # (78,) pcFM units
    total_points = int(counts_int.sum())

    #Step 2: Diffusion model, point cloud, driven by pcFM

    cond_np = cond.cpu().numpy()
    points_per_layer = counts_int[None, :]
    energy_pl = energy_per_layer[None, :]

    sample = sampler.sample(cond_np, points_per_layer.sum(1))

    physical_points, point_layer_ids = inference.sample_to_physical(sample, points_per_layer, sampler.config)

    #Rescale each layer's energy to pcFM's energy_per_layer (taken from CC)
    energies = physical_points[:, :, 3].copy()
    diffusion_sum = np.zeros_like(energy_pl)
    for layer in range(energy_pl.shape[1]):
        mask = point_layer_ids == layer
        diffusion_sum[:, layer] = (energies * mask).sum(axis=1)

    ratio = np.divide(energy_pl, diffusion_sum, out=np.zeros_like(energy_pl), where=diffusion_sum!=0)
    scale_per_point = np.take_along_axis(ratio, np.clip(point_layer_ids, 0, None), axis=1)
    scale_per_point[point_layer_ids < 0] = 1.0
    physical_points[:,:,3] = energies * scale_per_point

    #Unshift to true detector positions
    physical_points = inference.unshift_points(physical_points, point_layer_ids, cond_np, sampler.config)

    #extract real hits
    real = (physical_points[0, :, 3] > 0) & (point_layer_ids[0] >= 0)
    x_out = physical_points[0, real, 0]
    y_out = physical_points[0, real, 2]
    layer_out = point_layer_ids[0, real].astype(np.float32)
    e_out = physical_points[0, real, 3] * 1e3 #GeV -> MeV

    order = np.argsort(layer_out, kind="stable")
    x_out = x_out[order]
    y_out = y_out[order]
    layer_out = layer_out[order]
    e_out = e_out[order]
 
    points = np.stack([x_out, y_out, layer_out.astype(np.float32), e_out], axis=1).astype(np.float32)

    header = np.bincount(layer_out, minlength=n_layers).astype(np.float32)
   
    full_report = np.concatenate([header, points.ravel()]).astype(np.float32, copy=False)
    return full_report