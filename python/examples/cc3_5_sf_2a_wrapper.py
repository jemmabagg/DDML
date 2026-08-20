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
pcfm_path = os.path.expanduser("/home/baggjemm/PointCountFM_private/results/20260729_164316_PointCountFM/compiled.pt") #Get the right pcfm model
pcfm = torch.jit.load(pcfm_path, map_location="cpu").eval()

#CaloCloud Model
cc_path = os.path.expanduser("/home/baggjemm/CaloCloudi/data/logs/2026_08_17__15_19_32/checkpoints/2026-08-18_14-35-58_model.pt")
sampler = inference.Sampler.from_model_path(cc_path)


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

    header = np.bincount(layer_out.astype(np.int64), minlength=n_layers).astype(np.float32)
   
    full_report = np.concatenate([header, points.ravel()]).astype(np.float32, copy=False)
    return full_report