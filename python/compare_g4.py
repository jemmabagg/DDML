#!/usr/bin/env python3
"""
Compare Geant4 ground truth vs fast-sim (pcFM+CaloClouds) -- hit level AND reco level.
 
Hit-level panels (from EcalBarrelCollection, SimCalorimeterHit): longitudinal energy
profile and per-layer hit multiplicity -- the quantities pcFM actually predicts.
 
Reco-level panels (from PandoraPFOs): reconstructed energy per event and PFO count --
what the ildc reconstruction (the _REC file) newly gives you over the SIM level.
 
Before plotting anything, the script prints -- per file -- which collections it found and
the truth incident-photon energy from MCParticles, so you can see whether the two samples
are actually comparable before reading anything into the shapes.
 
Run inside key4hep on Maxwell:
    python compare_g4_reco.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.root_io import Reader
 
# ------------------------------- config -------------------------------
# EDIT these two paths to your files.
GEANT4_FILE  = "/home/baggjemm/data_production/10k_batches/ground_truth.edm4hep.root"  # <-- CONFIRM this path
FASTSIM_FILE = "/home/baggjemm/ILDConfig/StandardConfig/production/dummyOutput_REC.edm4hep.root"
 
SIM_COLL = "EcalBarrelCollection"      # SimCalorimeterHit     (raw deposits, hit level)
PFO_COLL = "PandoraPFOs"               # ReconstructedParticle (reco level)
MC_COLL  = "MCParticles"               # MCParticle            (truth)
 
N_ECAL_LAYERS = 30
OUTPUT_PNG    = "compare_g4_reco.png"
 
# Leave None to auto-detect from file metadata. If auto-detect fails, paste the
# CellIDEncoding string here, e.g. "system:5,side:-2,module:8,stave:4,layer:9,..."
ENCODING = None
# ----------------------------------------------------------------------
 
 
def get_encoding(reader, collection):
    """Pull the CellIDEncoding string from the file's metadata frame."""
    try:
        frame = reader.get("metadata")[0]
        return frame.get_parameter(f"{collection}__CellIDEncoding")
    except Exception as e:
        print(f"  [warn] could not auto-read encoding: {e}")
        return None
 
 
def parse_fields(encoding):
    """Parse a DD4hep bitfield encoding string into {name: (offset, width)}."""
    fields, offset = {}, 0
    for tok in encoding.split(","):
        p = tok.split(":")
        name = p[0].strip()
        if len(p) == 2:
            off, width = offset, int(p[1])
        else:
            off, width = int(p[1]), int(p[2])
        width = abs(width)
        fields[name] = (off, width)
        offset = off + width
    return fields
 
 
def make_layer_decoder(encoding):
    """Return (decode_fn, field_names). decode_fn is None if there's no 'layer' field."""
    fields = parse_fields(encoding)
    if "layer" not in fields:
        return None, list(fields.keys())
    off, width = fields["layer"]
    mask = (1 << width) - 1
    return (lambda cid: (cid >> off) & mask), list(fields.keys())
 
 
def process(filename, decode_layer, n_layers):
    """Loop events; accumulate hit-level, reco-level, and truth quantities."""
    reader = Reader(filename)
    try:
        events = reader.get("events")
    except Exception as e:
        sys.exit(f"[fatal] cannot read 'events' from {filename}: {e}")
 
    per_layer_E  = np.zeros(n_layers)
    per_layer_N  = np.zeros(n_layers)
    per_layer_E2 = np.zeros(n_layers)      # for the per-event stddev of the profile
    tot_E, tot_N, cell_E = [], [], []      # hit level
    reco_E, n_pfo = [], []                 # reco level
    truth_E = []                           # truth
    available, n_ev = None, 0
 
    for frame in events:
        if available is None:
            try:
                available = list(frame.getAvailableCollections())
            except Exception:
                available = []
 
        # ---- hit level (sim deposits) ----
        if SIM_COLL in available:
            e_sum, n_hits = 0.0, 0
            ev_layer_E = np.zeros(n_layers)
            for hit in frame.get(SIM_COLL):
                e = hit.getEnergy()                 # GeV
                e_sum += e
                n_hits += 1
                cell_E.append(e)
                if decode_layer is not None:
                    L = decode_layer(hit.getCellID())
                    if 0 <= L < n_layers:
                        per_layer_E[L] += e
                        per_layer_N[L] += 1
                        ev_layer_E[L]  += e
            per_layer_E2 += ev_layer_E ** 2
            tot_E.append(e_sum)
            tot_N.append(n_hits)
 
        # ---- reco level (Pandora PFOs) ----
        if PFO_COLL in available:
            es = [p.getEnergy() for p in frame.get(PFO_COLL)]   # GeV
            reco_E.append(sum(es))
            n_pfo.append(len(es))
 
        # ---- truth (incident photon) ----
        if MC_COLL in available:
            cand = [p.getEnergy() for p in frame.get(MC_COLL)
                    if p.getPDG() == 22 and p.getGeneratorStatus() == 1]
            if cand:
                truth_E.append(max(cand))
 
        n_ev += 1
 
    n = max(n_ev, 1)
    mean_E = per_layer_E / n
    var_E  = np.maximum(per_layer_E2 / n - mean_E ** 2, 0.0)
    sem_E  = np.sqrt(var_E) / np.sqrt(n)
    return {
        "mean_layer_E": mean_E,
        "sem_layer_E":  sem_E,
        "mean_layer_N": per_layer_N / n,
        "tot_E":   np.array(tot_E),
        "tot_N":   np.array(tot_N),
        "cell_E":  np.array(cell_E),
        "reco_E":  np.array(reco_E),
        "n_pfo":   np.array(n_pfo),
        "truth_E": np.array(truth_E),
        "available": available or [],
        "n_ev": n_ev,
    }
 
 
def report(tag, d):
    """Print, per file, what's actually in it -- the check before any plot."""
    print(f"\n[{tag}] events = {d['n_ev']}")
    print(f"  collections present : "
          f"sim({SIM_COLL})={SIM_COLL in d['available']}  "
          f"pfo({PFO_COLL})={PFO_COLL in d['available']}  "
          f"mc({MC_COLL})={MC_COLL in d['available']}")
    if len(d["truth_E"]):
        t = d["truth_E"]
        print(f"  truth photon E [GeV]: mean={t.mean():.2f}  range=[{t.min():.2f}, {t.max():.2f}]")
    else:
        print(f"  truth photon E [GeV]: (no status-1 photon found in {MC_COLL})")
    if len(d["reco_E"]):
        print(f"  reco  Sum-E   [GeV] : mean={d['reco_E'].mean():.2f}  (mean #PFO={d['n_pfo'].mean():.1f})")
 
 
def overlay_hist(ax, g4, fs, xlabel, title, bins=40, logy=False, integer=False):
    data = [x for x in (g4, fs) if len(x)]
    if not data:
        ax.text(0.5, 0.5, "no data\nin either file", ha="center", va="center")
        ax.set_axis_off()
        return
    lo = min(x.min() for x in data)
    hi = max(x.max() for x in data)
    if integer:
        b = np.arange(int(lo), int(hi) + 2) - 0.5
    else:
        b = np.linspace(lo, hi, bins) if hi > lo else np.linspace(lo, lo + 1, bins)
    if len(g4):
        ax.hist(g4, bins=b, density=True, histtype="step", color="black",
                label=f"Geant4 (N={len(g4)})")
    if len(fs):
        ax.hist(fs, bins=b, density=True, histtype="step", color="tab:red",
                label=f"fast-sim (N={len(fs)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("normalized")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend()
 
 
def main():
    # resolve the layer decoder once (readouts match between the two files)
    decode_layer = None
    enc = ENCODING
    if enc is None:
        print("Auto-detecting CellIDEncoding...")
        enc = get_encoding(Reader(FASTSIM_FILE), SIM_COLL) \
            or get_encoding(Reader(GEANT4_FILE), SIM_COLL)
    if enc is not None:
        decode_layer, field_names = make_layer_decoder(enc)
        if decode_layer is None:
            print(f"  [warn] no 'layer' field in encoding. fields: {field_names}"
                  f"  -> per-layer panels will be skipped")
        else:
            print("  encoding OK, layer field found")
    else:
        print("  [warn] no encoding -> per-layer panels skipped. Get it manually with:")
        print("    python -c \"from podio.root_io import Reader; "
              "r=Reader('dummyOutput_REC.edm4hep.root'); "
              f"print(r.get('metadata')[0].get_parameter('{SIM_COLL}__CellIDEncoding'))\"")
        print("  then paste it into ENCODING at the top.")
 
    print(f"\nReading Geant4 : {GEANT4_FILE}")
    g4 = process(GEANT4_FILE, decode_layer, N_ECAL_LAYERS)
    print(f"Reading fastsim: {FASTSIM_FILE}")
    fs = process(FASTSIM_FILE, decode_layer, N_ECAL_LAYERS)
 
    report("Geant4", g4)
    report("fast-sim", fs)
 
    if fs["n_ev"] < 50:
        print(f"\n  *** {fs['n_ev']} fast-sim events is a plumbing test, not a validation. ***")
        print("      Curves will be statistically noisy; re-run on the full sample.")
    if PFO_COLL not in g4["available"]:
        print(f"\n  *** Geant4 file has no {PFO_COLL}: it's a SIM file, not reco'd through the")
        print("      same ildc chain. The two reco panels will carry fast-sim only. Either")
        print("      run reco on it the same way, or compare at hit level only.")
 
    # ------------------------------ plotting ------------------------------
    layers = np.arange(N_ECAL_LAYERS)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    G4 = dict(color="black",   label=f"Geant4 (N={g4['n_ev']})")
    FS = dict(color="tab:red", label=f"fast-sim (N={fs['n_ev']})")
 
    # (0,0) longitudinal energy profile -- hit level, pcFM's core claim
    a = ax[0, 0]
    if decode_layer is not None:
        a.errorbar(layers, g4["mean_layer_E"], yerr=g4["sem_layer_E"], fmt="o-", ms=3, **G4)
        a.errorbar(layers, fs["mean_layer_E"], yerr=fs["sem_layer_E"], fmt="s-", ms=3, **FS)
        a.set_xlabel("ECAL layer"); a.set_ylabel("mean energy / layer [GeV]")
        a.set_title("Longitudinal profile (hit level)"); a.legend()
    else:
        a.text(0.5, 0.5, "per-layer skipped\n(no encoding)", ha="center", va="center"); a.set_axis_off()
 
    # (0,1) per-layer hit multiplicity -- hit level
    a = ax[0, 1]
    if decode_layer is not None:
        a.plot(layers, g4["mean_layer_N"], "o-", ms=3, **G4)
        a.plot(layers, fs["mean_layer_N"], "s-", ms=3, **FS)
        a.set_xlabel("ECAL layer"); a.set_ylabel("mean hits / layer")
        a.set_title("Hit multiplicity vs layer (hit level)"); a.legend()
    else:
        a.text(0.5, 0.5, "per-layer skipped\n(no encoding)", ha="center", va="center"); a.set_axis_off()
 
    # (0,2) cell energy spectrum -- hit level, log-y
    overlay_hist(ax[0, 2], g4["cell_E"], fs["cell_E"],
                 "cell energy [GeV]", "Cell energy spectrum (hit level)", logy=True)
 
    # (1,0) reconstructed energy per event -- RECO level, the new thing
    overlay_hist(ax[1, 0], g4["reco_E"], fs["reco_E"],
                 "reconstructed Sum E(PFO) / event [GeV]", "Reconstructed energy (reco level)")
 
    # (1,1) number of PFOs per event -- reco level
    overlay_hist(ax[1, 1], g4["n_pfo"], fs["n_pfo"],
                 "# PFOs / event", "PFO multiplicity (reco level)", integer=True)
 
    # (1,2) total visible (sim) energy per event -- hit level
    overlay_hist(ax[1, 2], g4["tot_E"], fs["tot_E"],
                 "total visible energy / event [GeV]", "Total sim energy (hit level)")
 
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=130)
    print(f"\nSaved {OUTPUT_PNG}")
 
 
if __name__ == "__main__":
    main()