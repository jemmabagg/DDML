"""
plot_showers.py
 
Validation plots for the pcFM -> CaloClouds fast simulation vs a Geant4 reference.

"""
 
import numpy as np
import matplotlib.pyplot as plt
 
N_LAYERS = 78          # 30 ECAL + 48 HCAL, ILD barrel geometry
ECAL_HCAL_BOUNDARY = 30
ENERGY_UNIT = "MeV"    # wrapper emits MeV; VERIFY against a known input (see note below)
 
 
# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
class Showers:
    def __init__(self, event_id, layer, energy, x, y,
                 n_events, incident_energy=None, label="model"):
        self.event_id = np.asarray(event_id).astype(int)
        self.layer = np.asarray(layer).astype(int)
        self.energy = np.asarray(energy, dtype=float)      # MeV
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n_events = int(n_events)
        self.incident_energy = incident_energy
        self.label = label
 
    # --- per-event scalars -------------------------------------------------
    def total_energy_per_event(self):
        out = np.zeros(self.n_events)
        np.add.at(out, self.event_id, self.energy)
        return out
 
    def hits_per_event(self):
        return np.bincount(self.event_id, minlength=self.n_events)
 
    # --- per-layer profiles (need valid layer indices) --------------------
    def mean_energy_per_layer(self):
        """Longitudinal profile: mean deposited energy per layer, per event."""
        m = self.layer >= 0
        prof = np.zeros(N_LAYERS)
        lay = np.clip(self.layer[m], 0, N_LAYERS - 1)
        np.add.at(prof, lay, self.energy[m])
        return prof / self.n_events
 
    def mean_hits_per_layer(self):
        """Occupancy profile: mean hit count per layer, per event. (pcFM's job.)"""
        m = self.layer >= 0
        lay = np.clip(self.layer[m], 0, N_LAYERS - 1)
        return np.bincount(lay, minlength=N_LAYERS)[:N_LAYERS] / self.n_events
 
    # --- transverse --------------------------------------------------------
    def radius_per_hit(self):
        """Distance of each hit from its event's energy-weighted CoG."""
        cx = np.zeros(self.n_events); cy = np.zeros(self.n_events)
        es = np.zeros(self.n_events)
        np.add.at(cx, self.event_id, self.x * self.energy)
        np.add.at(cy, self.event_id, self.y * self.energy)
        np.add.at(es, self.event_id, self.energy)
        es[es == 0] = 1.0
        cx /= es; cy /= es
        dx = self.x - cx[self.event_id]
        dy = self.y - cy[self.event_id]
        return np.sqrt(dx * dx + dy * dy)
 
 
# ---------------------------------------------------------------------------
# Loaders  
# ------------------------------------------------------------------------
 
def parse_cellid_encoding(desc):
    """DD4hep encoding string -> {field: (offset, width, signed)}.
    Handles both 'name:width' (auto offset) and 'name:offset:width' tokens;
    a negative width means the field is signed (e.g. 'x:-16')."""
    fields, offset = {}, 0
    for tok in desc.replace(" ", "").split(","):
        p = tok.split(":")
        name = p[0]
        if len(p) == 2:
            off, width = offset, int(p[1])
        else:
            off, width = int(p[1]), int(p[2])
        fields[name] = (off, abs(width), width < 0)
        offset = off + abs(width)
    return fields
 
 
def decode_field(cellid, field):
    off, width, signed = field
    cellid = np.asarray(cellid, dtype=np.uint64)
    val = ((cellid >> np.uint64(off)) & np.uint64((1 << width) - 1)).astype(np.int64)
    if signed:
        val = np.where(val & (1 << (width - 1)), val - (1 << width), val)
    return val
 
 
def load_from_root(path, collection=None, encoding=None, label="model"):
    """
    Read an edm4hep root file directly with uproot.
 
    collection : the calo hit collection name (run once with collection=None to
                 print the branch list, then pass the right one).
    encoding   : the cellID encoding string, e.g.
                 "system:5,module:3,stave:4,tower:5,layer:6,x:32:-16,y:-16".
                 Get it from `podio-dump <file>.root` (printed per collection) or
                 from Henry. Without it, layer is set to -1 and the per-layer
                 plots stay empty (energy/multiplicity/radial still work).
    """
    import uproot
    tree = uproot.open(path)["events"]
    if collection is None:
        print("Available branches -- pick your calo hit collection:")
        for k in tree.keys():
            print("  ", k)
        raise SystemExit("Re-call with collection='<YourCaloCollection>'.")
 
    energy = tree[f"{collection}.energy"].array(library="np")     # jagged, per event
    px = tree[f"{collection}.position.x"].array(library="np")
    py = tree[f"{collection}.position.y"].array(library="np")
    cid = tree[f"{collection}.cellID"].array(library="np")
 
    n_events = len(energy)
    ev = np.repeat(np.arange(n_events), [len(a) for a in energy])
    ef = np.concatenate(energy) if n_events else np.array([])
    xf = np.concatenate(px) if n_events else np.array([])
    yf = np.concatenate(py) if n_events else np.array([])
    cf = (np.concatenate(cid).astype(np.uint64) if n_events
          else np.array([], dtype=np.uint64))
 
    if encoding is not None:
        fields = parse_cellid_encoding(encoding)
        if "layer" not in fields:
            raise KeyError(f"no 'layer' field in encoding; found {list(fields)}")
        layer = decode_field(cf, fields["layer"])
    else:
        print("WARNING: no encoding given -> layer = -1 (per-layer plots empty). "
              "Get the string from `podio-dump` or Henry.")
        layer = np.full(len(ef), -1)
 
    return Showers(ev, layer, ef, xf, yf, n_events, label=label)
 
 
# ILD ECAL barrel Si readout: layer sits at bits 16-21 (6 bits wide)
ENC = ("system:0:5,module:5:3,stave:8:4,tower:12:4,layer:16:6,"
       "wafer:22:6,slice:28:4,cellX:32:-16,cellY:48:-16")
 
 
def load_barrel(path, label):
    """The ECAL barrel Si readout is split into Even/Odd collections; load both
    and merge into one Showers object. event_id keeps each hit tied to its event,
    so per-event and per-layer sums stay correct across the merge."""
    parts = [load_from_root(path, c, encoding=ENC, label=label)
             for c in ("ECalBarrelSiHitsEven", "ECalBarrelSiHitsOdd")]
    return Showers(
        event_id=np.concatenate([p.event_id for p in parts]),
        layer=np.concatenate([p.layer for p in parts]),
        energy=np.concatenate([p.energy for p in parts]),
        x=np.concatenate([p.x for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        n_events=parts[0].n_events,
        label=label,
    )
 
 
# ---------------------------------------------------------------------------
# Plots  --  each overlays model vs reference
# ---------------------------------------------------------------------------
def _finish(ax, xlabel, ylabel, title, path):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    ax.figure.tight_layout(); ax.figure.savefig(path, dpi=150)
    print("wrote", path); plt.close(ax.figure)
 
 
def plot_longitudinal(model, ref=None, path="prof_longitudinal.png"):
    fig, ax = plt.subplots(); L = np.arange(N_LAYERS)
    ax.step(L, model.mean_energy_per_layer(), where="mid", label=model.label)
    if ref is not None:
        ax.step(L, ref.mean_energy_per_layer(), where="mid", ls="--", label=ref.label)
    ax.axvline(ECAL_HCAL_BOUNDARY, color="grey", lw=0.8, alpha=0.6)
    _finish(ax, "layer", f"mean E / event [{ENERGY_UNIT}]",
            "Longitudinal shower profile", path)
 
 
def plot_hits_per_layer(model, ref=None, path="prof_hits_per_layer.png"):
    fig, ax = plt.subplots(); L = np.arange(N_LAYERS)
    ax.step(L, model.mean_hits_per_layer(), where="mid", label=model.label)
    if ref is not None:
        ax.step(L, ref.mean_hits_per_layer(), where="mid", ls="--", label=ref.label)
    ax.axvline(ECAL_HCAL_BOUNDARY, color="grey", lw=0.8, alpha=0.6)
    _finish(ax, "layer", "mean hits / event", "Occupancy profile (pcFM)", path)
 
 
def plot_multiplicity(model, ref=None, path="hist_multiplicity.png"):
    fig, ax = plt.subplots(); m = model.hits_per_event()
    bins = np.histogram_bin_edges(m, bins=40)
    ax.hist(m, bins=bins, density=True, histtype="step", label=model.label)
    if ref is not None:
        ax.hist(ref.hits_per_event(), bins=bins, density=True, histtype="step",
                ls="--", label=ref.label)
    _finish(ax, "hits / event", "normalized", "Total hit multiplicity (pcFM)", path)
 
 
def plot_total_energy(model, ref=None, path="hist_total_energy.png"):
    fig, ax = plt.subplots(); m = model.total_energy_per_event()
    bins = np.histogram_bin_edges(m, bins=40)
    ax.hist(m, bins=bins, density=True, histtype="step", label=model.label)
    if ref is not None:
        ax.hist(ref.total_energy_per_event(), bins=bins, density=True,
                histtype="step", ls="--", label=ref.label)
    _finish(ax, f"total visible E / event [{ENERGY_UNIT}]", "normalized",
            "Total deposited energy", path)
 
 
def plot_hit_energy_spectrum(model, ref=None, path="hist_hit_energy.png"):
    fig, ax = plt.subplots()
    lo = max(model.energy[model.energy > 0].min(), 1e-3)
    bins = np.logspace(np.log10(lo), np.log10(model.energy.max()), 50)
    ax.hist(model.energy, bins=bins, density=True, histtype="step", label=model.label)
    if ref is not None:
        ax.hist(ref.energy, bins=bins, density=True, histtype="step",
                ls="--", label=ref.label)
    ax.set_xscale("log"); ax.set_yscale("log")
    _finish(ax, f"hit energy [{ENERGY_UNIT}]", "normalized",
            "Per-hit energy spectrum", path)
 
 
def plot_radial(model, ref=None, path="hist_radial.png"):
    fig, ax = plt.subplots(); r = model.radius_per_hit()
    bins = np.histogram_bin_edges(r, bins=40)
    ax.hist(r, bins=bins, weights=model.energy, density=True,
            histtype="step", label=model.label)
    if ref is not None:
        ax.hist(ref.radius_per_hit(), bins=bins, weights=ref.energy,
                density=True, histtype="step", ls="--", label=ref.label)
    _finish(ax, "radius from CoG [mm]", "normalized energy",
            "Transverse energy profile", path)
 
 
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_FILE = "dummyOutput_edm4hep.root"
    GEANT_FILE = None          # set to your Geant4 reference .root when you have it
 
    model = load_barrel(MODEL_FILE, "pcFM+CaloClouds")
    ref = load_barrel(GEANT_FILE, "Geant4") if GEANT_FILE else None
 
    # --- sanity check before plotting --------------------------------------
    print("events:            ", model.n_events)
    print("total hits:        ", len(model.energy))
    print("hits/event mean:   ", round(model.hits_per_event().mean(), 1))
    print("layer range:       ", int(model.layer.min()), "-", int(model.layer.max()))
    print("total E/event mean:", round(model.total_energy_per_event().mean(), 3),
          ENERGY_UNIT, "(if ~1000x too small, energy is in GeV -> add *1e3 in load_from_root)")
 
    # --- plots -------------------------------------------------------------
    plot_longitudinal(model, ref)
    plot_hits_per_layer(model, ref)
    plot_multiplicity(model, ref)
    plot_total_energy(model, ref)
    plot_hit_energy_spectrum(model, ref)
    plot_radial(model, ref)