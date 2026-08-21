"""pcFM + Diffusion model wrapper 


import os
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, "/home/baggjemm/CaloCloudi")

from src.detector_map import get_layer_centers

torch.set_num_threads(1)

#pcFM Model Path
pcfm_path = os.path.expanduser("/home/baggjemm/PointCountFM_private/results/20260817_133705_TEMPO_PandT_/compiled.pt") #Get the right pcfm model
pcfm = torch.jit.load(pcfm_path, map_location="cpu").eval()

#CaloCloud Model
cc_path = os.path.expanduser("/data/dust/user/dayhallh/data/CaloClouds_diffusion/distilled.ts.pt")
cc = torch.jit.load(cc_path, map_location="cpu").eval()

#Config

config_path = os.path.expanduser("/home/baggjemm/DDML/models/padded_photons.yaml")

with open(config_path) as f:
    config = yaml.safe_load(f)

n_layers = len(config["data"]["layer_bottom_pos"])
feature_dim = config["model"]["feature_dim"]

##CC post-processiong - taken from inference.py in CaloCloudi

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

#Energy rescaling now made into a function 

def energy_corrections(physical_points, point_layer_ids, energy_pl):
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
    return physical_points



@torch.inference_mode()
def run_inference(inputs):

    #make the torch dtype float32
    dtype = torch.get_default_dtype()

    #unpack (one shower per call)
    energy = torch.from_numpy(np.asarray(inputs[0], dtype=np.float32).reshape(-1)).to(dtype)      # (1,)
    theta_deg = torch.from_numpy(np.asarray(inputs[1], dtype=np.float32).reshape(-1)).to(dtype)   # (1,)
    phi_deg = torch.from_numpy(np.asarray(inputs[2], dtype=np.float32).reshape(-1)).to(dtype)     # (1,)

    print("\n===== ML INPUT =====")
    print("energy:", inputs[0])
    print("theta:", inputs[1])
    print("phi:", inputs[2])

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
    num_points = points_per_layer.sum(axis=1)
    max_points = int(np.max(num_points))

    #Henry's work
    noise = torch.randn(1, max_points, feature_dim, dtype=dtype,)

    keep = torch.as_tensor(num_points,dtype=torch.int64,)

    sample = cc(cond,noise,keep,).cpu().numpy()

    #Convert to physical detector coords
    physical_points, point_layer_ids = (sample_to_physical(sample,points_per_layer,config,))

    #Rescale energy
    physical_points = energy_corrections(physical_points,point_layer_ids,energy_pl,)
    print("\npcFM energy per layer:")
    print(energy_pl)

    print("\nFinal diffusion energy per layer:")
    final_layer_energy = np.zeros_like(energy_pl)

    for layer in range(n_layers):
        mask = point_layer_ids == layer
        final_layer_energy[:, layer] = (
            physical_points[:, :, 3] * mask
        ).sum(axis=1)

    print(final_layer_energy)

    print("\nTotal pcFM energy [GeV]:", energy_pl.sum())
    print("Total final energy [GeV]:", final_layer_energy.sum())

    #Unshift to true detector positions
    physical_points = unshift_points(physical_points, point_layer_ids, cond_np, config)

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

    header = np.bincount(layer_out.astype(np.int64), minlength=n_layers).astype(np.float32)

    print("layer_out range:", layer_out.min(), layer_out.max(), "n_hits:", len(layer_out))
    print("header sum:", header.sum(), "n_layers:", n_layers)
   
    full_report = np.concatenate([header, points.ravel()]).astype(np.float32, copy=False)
    return full_report"""

"""pcFM + Diffusion model wrapper 
"""

import os
import sys
import numpy as np
import torch
import yaml
import traceback

# --- debug setup -----------------------------------------------------------
# ddsim runs this Python from the C++ DDML backend, where stdout is buffered.
# A hard crash (or a C++-side abort) discards anything not yet flushed, which
# makes the failure look like it happened earlier than it really did. Force
# line-buffering on stdout, and route stage markers to stderr (which is
# effectively unbuffered) so the LAST line printed is genuinely the last line
# that ran before the crash.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def dbg(*args):
    print("[wrap]", *args, file=sys.stderr, flush=True)


dbg("module import: start")

sys.path.insert(0, "/home/baggjemm/CaloCloudi")

from src.detector_map import get_layer_centers

dbg("imported get_layer_centers OK")

torch.set_num_threads(1)

#pcFM Model Path
pcfm_path = os.path.expanduser("/home/baggjemm/PointCountFM_private/results/20260817_133705_TEMPO_PandT_/compiled.pt") #Get the right pcfm model
dbg("loading pcFM from:", pcfm_path)
pcfm = torch.jit.load(pcfm_path, map_location="cpu").eval()
dbg("pcFM loaded OK")

#CaloCloud Model
cc_path = os.path.expanduser("/data/dust/user/dayhallh/data/CaloClouds_diffusion/distilled.ts.pt")
dbg("loading CaloClouds from:", cc_path)
cc = torch.jit.load(cc_path, map_location="cpu").eval()
dbg("CaloClouds loaded OK")

#Config

config_path = os.path.expanduser("/home/baggjemm/DDML/models/padded_photons.yaml")
dbg("loading config from:", config_path)

with open(config_path) as f:
    config = yaml.safe_load(f)

n_layers = len(config["data"]["layer_bottom_pos"])
feature_dim = config["model"]["feature_dim"]
dbg("config loaded: n_layers =", n_layers, " feature_dim =", feature_dim)
dbg("module import: done")

##CC post-processiong - taken from inference.py in CaloCloudi

def sample_to_physical(points, points_per_layer, config):
    dbg("  sample_to_physical: enter | points", points.shape,
        "points_per_layer", points_per_layer.shape)
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
    dbg("  sample_to_physical: layer_centers", layer_centers.shape)
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

    dbg("  sample_to_physical: exit")
    return physical_points, point_layer_ids

def unshift_points(physical_points, point_layer_ids, cond_data_coords, config):
    dbg("  unshift_points: enter")
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

    dbg("  unshift_points: exit")
    return physical_points

#Energy rescaling now made into a function 

def energy_corrections(physical_points, point_layer_ids, energy_pl):
    dbg("  energy_corrections: enter")
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
    dbg("  energy_corrections: exit")
    return physical_points



@torch.inference_mode()
def run_inference(inputs):
  try:
    dbg("run_inference: ENTER")

    #make the torch dtype float32
    dtype = torch.get_default_dtype()

    #unpack (one shower per call)
    energy = torch.from_numpy(np.asarray(inputs[0], dtype=np.float32).reshape(-1)).to(dtype)      # (1,)
    theta_deg = torch.from_numpy(np.asarray(inputs[1], dtype=np.float32).reshape(-1)).to(dtype)   # (1,)
    phi_deg = torch.from_numpy(np.asarray(inputs[2], dtype=np.float32).reshape(-1)).to(dtype)     # (1,)

    print("\n===== ML INPUT =====")
    print("energy:", inputs[0])
    print("theta:", inputs[1])
    print("phi:", inputs[2])
    dbg("inputs unpacked | energy", energy.shape, "theta", theta_deg.shape, "phi", phi_deg.shape)

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
    dbg("cond built:", cond.shape, cond.dtype, "->", cond.cpu().numpy())

    ##pcFM per layer
    ## we get counts per layer and energy per layer from pcFM

    dbg("calling pcfm ...")
    raw = pcfm(cond)
    dbg("pcfm returned | type", type(raw),
        "| shapes", [tuple(t.shape) for t in raw] if isinstance(raw, (list, tuple)) else raw.shape)
    if raw[-1].shape[1] != n_layers:
        raw = raw[:-1]
    pcfm_out = torch.cat(list(raw), dim=1)
    dbg("pcfm_out:", pcfm_out.shape, "| expected width 2*n_layers =", 2 * n_layers)

    # hallucination guard (from inference_cond_file.py _check_hallucinations):
    # re-sample if per-layer counts blow up, BEFORE clamping. Without this one
    # bad shower makes total_points huge and can OOM the sampler.
    guard_iter = 0
    while pcfm_out[:, :n_layers].max() > 1e4 or pcfm_out[:, :n_layers].min() < -1e4:
        guard_iter += 1
        dbg("hallucination guard RESAMPLE #", guard_iter,
            "| max", float(pcfm_out[:, :n_layers].max()),
            "min", float(pcfm_out[:, :n_layers].min()))
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
    dbg("counts_int:", counts_int.shape, "| min", counts_int.min(),
        "max", counts_int.max(), "| total_points", total_points)

    #Step 2: Diffusion model, point cloud, driven by pcFM

    cond_np = cond.cpu().numpy()
    points_per_layer = counts_int[None, :]
    energy_pl = energy_per_layer[None, :]
    num_points = points_per_layer.sum(axis=1)
    max_points = int(np.max(num_points))
    dbg("Step 2 setup | num_points", num_points, "max_points", max_points)

    #Henry's work
    noise = torch.randn(1, max_points, feature_dim, dtype=dtype,)

    keep = torch.as_tensor(num_points,dtype=torch.int64,)

    dbg("calling cc | cond", cond.shape, "noise", noise.shape, "keep", keep,
        "| dtypes:", cond.dtype, noise.dtype, keep.dtype)
    sample = cc(cond,noise,keep,).cpu().numpy()
    dbg("cc returned | sample", sample.shape, sample.dtype)

    #Convert to physical detector coords
    physical_points, point_layer_ids = (sample_to_physical(sample,points_per_layer,config,))
    dbg("physical_points", physical_points.shape, "| point_layer_ids", point_layer_ids.shape)

    #Rescale energy
    physical_points = energy_corrections(physical_points,point_layer_ids,energy_pl,)
    print("\npcFM energy per layer:")
    print(energy_pl)

    print("\nFinal diffusion energy per layer:")
    final_layer_energy = np.zeros_like(energy_pl)

    for layer in range(n_layers):
        mask = point_layer_ids == layer
        final_layer_energy[:, layer] = (
            physical_points[:, :, 3] * mask
        ).sum(axis=1)

    print(final_layer_energy)

    print("\nTotal pcFM energy [GeV]:", energy_pl.sum())
    print("Total final energy [GeV]:", final_layer_energy.sum())

    #Unshift to true detector positions
    physical_points = unshift_points(physical_points, point_layer_ids, cond_np, config)

    #extract real hits
    real = (physical_points[0, :, 3] > 0) & (point_layer_ids[0] >= 0)
    x_out = physical_points[0, real, 0]
    y_out = physical_points[0, real, 2]
    layer_out = point_layer_ids[0, real].astype(np.float32)
    e_out = physical_points[0, real, 3] * 1e3 #GeV -> MeV
    dbg("real hits extracted | n_real", int(real.sum()))

    order = np.argsort(layer_out, kind="stable")

    x_out = x_out[order]
    y_out = y_out[order]
    layer_out = layer_out[order]
    e_out = e_out[order]
 
    points = np.stack([x_out, y_out, layer_out.astype(np.float32), e_out], axis=1).astype(np.float32)

    header = np.bincount(layer_out.astype(np.int64), minlength=n_layers).astype(np.float32)

    print("layer_out range:", layer_out.min(), layer_out.max(), "n_hits:", len(layer_out))
    print("header sum:", header.sum(), "n_layers:", n_layers)
   
    full_report = np.concatenate([header, points.ravel()]).astype(np.float32, copy=False)
    dbg("run_inference: EXIT | full_report", full_report.shape)
    return full_report
  except Exception:
    dbg("run_inference CRASHED - traceback follows:")
    traceback.print_exc()
    sys.stderr.flush()
    raise