"""
plot_showers.py
 
Validation plots for the pcFM -> CaloClouds fast simulation vs a Geant4 reference.

"""
 
import numpy as np
import matplotlib.pyplot as plt
 
N_LAYERS = 78          # 30 ECAL + 48 HCAL, ILD barrel geometry
ECAL_HCAL_BOUNDARY = 30
ENERGY_UNIT = "MeV"    # wrapper emits MeV; VERIFY against a known input (see note below)
 
# Data container
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
 
    # per-event scalars
    def total_energy_per_event(self):
        out = np.zeros(self.n_events)
        np.add.at(out, self.event_id, self.energy)
        return out
 
    def hits_per_event(self):
        return np.bincount(self.event_id, minlength=self.n_events)
 
    # per-layer profiles
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

    def _per_event_layer(self, values):
        """(n_events, N_LAYERS) matrix; entry = sum of `values` for that (event, layer)."""
        m = self.layer >= 0
        ev = self.event_id[m]
        lay = np.clip(self.layer[m], 0, N_LAYERS - 1)
        mat = np.zeros((self.n_events, N_LAYERS))
        np.add.at(mat, (ev, lay), values[m])
        return mat

    def sem_energy_per_layer(self):
        mat = self._per_event_layer(self.energy)
        return mat.std(axis=0, ddof=1) / np.sqrt(self.n_events)

    def sem_hits_per_layer(self):
        mat = self._per_event_layer(np.ones_like(self.energy))
        return mat.std(axis=0, ddof=1) / np.sqrt(self.n_events)
 
    # transverse 
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
 
 
# Loaders
 
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
 
# Plot functions
def _finish(ax, xlabel, ylabel, title, path):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    ax.figure.tight_layout(); ax.figure.savefig(path, dpi=150)
    print("wrote", path); plt.close(ax.figure)
 

#Energy per layer plot
def plot_longitudinal(model, ref=None, path="prof_longitudinal.png"):
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    L = np.arange(N_LAYERS)

    mean1 = model.mean_energy_per_layer()
    ax_main.step(L, mean1, where="mid", label=model.label)

    if ref is not None:
        mean2 = ref.mean_energy_per_layer()
        ax_main.step(L, mean2, where="mid", ls="--", label=ref.label)

        sem1 = model.sem_energy_per_layer()   # standard error per layer
        sem2 = ref.sem_energy_per_layer()

        ratio = mean1 / mean2
        ratio_unc = ratio * np.sqrt((sem1/mean1)**2 + (sem2/mean2)**2)

        ax_ratio.axhline(1.0, color="r", ls="--")
        ax_ratio.errorbar(L, ratio, yerr=ratio_unc,
                          fmt="ko", capsize=3, markersize=3)

    ax_main.set_ylabel(f"mean E / event [{ENERGY_UNIT}]")
    ax_main.legend()
    ax_ratio.set_ylabel("gen / real")
    ax_ratio.set_xlabel("layer")

    fig.subplots_adjust(hspace=0.05)
    fig.suptitle("Longitudinal shower profile")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
#Number of hits per layer plot
def plot_hits_per_layer(model, ref=None, path="prof_hits_per_layer.png"):
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    L = np.arange(N_LAYERS)

    mean1 = model.mean_hits_per_layer()
    ax_main.step(L, mean1, where="mid", label=model.label)

    if ref is not None:
        mean2 = ref.mean_hits_per_layer()
        ax_main.step(L, mean2, where="mid", ls="--", label=ref.label)
        
        sem1 = model.sem_hits_per_layer()   # standard error per layer
        sem2 = ref.sem_hits_per_layer()

        ratio = mean1 / mean2
        ratio_unc = ratio * np.sqrt((sem1/mean1)**2 + (sem2/mean2)**2)

        ax_ratio.axhline(1.0, color="r", ls="--")
        ax_ratio.errorbar(L, ratio, yerr=ratio_unc,
                            fmt="ko", capsize=3, markersize=3)
    
    ax_main.set_ylabel(f"mean hit / event")
    ax_main.legend()
    ax_ratio.set_ylabel("gen / real")
    ax_ratio.set_xlabel("layer")

    fig.subplots_adjust(hspace=0.05)
    fig.suptitle("Occupancy Profile")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
if __name__ == "__main__":
    MODEL_FILE = "/home/baggjemm/DDML/scripts/dummyOutput_edm4hep.root"
    GEANT_FILE = "/home/baggjemm/data_production/10k_batches/ground_truth.edm4hep.root"
 
    model = load_barrel(MODEL_FILE, "pcFM+CaloClouds")
    ref = load_barrel(GEANT_FILE, "Geant4") if GEANT_FILE else None
 
  
    print("events:            ", model.n_events)
    print("total hits:        ", len(model.energy))
    print("hits/event mean:   ", round(model.hits_per_event().mean(), 1))
    print("layer range:       ", int(model.layer.min()), "-", int(model.layer.max()))
    print("total E/event mean:", round(model.total_energy_per_event().mean(), 3),
          ENERGY_UNIT, "(if ~1000x too small, energy is in GeV -> add *1e3 in load_from_root)")
    model_mean_E = model.total_energy_per_event().mean()
    ref_mean_E = ref.total_energy_per_event().mean()

    energy_ratio = ref_mean_E / model_mean_E

    print("\n===== ENERGY CALIBRATION =====")
    print("Mean model deposited energy:", model_mean_E, ENERGY_UNIT)
    print("Mean Geant4 deposited energy:", ref_mean_E, ENERGY_UNIT)
    print("Geant4 / model energy ratio:", energy_ratio)
 
    # plots
    plot_longitudinal(model, ref)
    plot_hits_per_layer(model, ref)
