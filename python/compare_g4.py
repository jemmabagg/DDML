#!/usr/bin/env python3
"""
Overlay comparison of two REC files at PFO level.

Usage:
    python3 compare_two.py FAST_REC.edm4hep.root FULL_REC.edm4hep.root

Both files must contain MCParticles and PandoraPFOs (both must have been
through ILD reconstruction). If a file's printed efficiency is ~0, it has
no PFOs - it is sim-level, not reco-level, and does not belong here.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from podio import root_io


PHOTON_PDG = 22
N_EBINS = 25          # histogram bins for the energy distributions


# ============================================================
# Process one REC file -> per-event arrays
# ============================================================

def process(rec_file, label):

    print(f"\nOpening [{label}]: {rec_file}")
    events = root_io.Reader(rec_file).get("events")
    print(f"  events: {len(events)}")

    true_E, reco_E, mult = [], [], []

    for event in events:

        mc = event.get("MCParticles")
        pfos = event.get("PandoraPFOs")

        prim = [
            p for p in mc
            if abs(p.getPDG()) == PHOTON_PDG
            and p.parents_size() == 0
        ]
        if not prim:
            continue

        reco_photons = [
            p for p in pfos
            if abs(p.getPDG()) == PHOTON_PDG
        ]

        true_E.append(prim[0].getEnergy())
        mult.append(len(reco_photons))
        reco_E.append(
            sum(p.getEnergy() for p in reco_photons)
            if reco_photons else np.nan
        )

    d = {
        "label": label,
        "true_E": np.asarray(true_E),
        "reco_E": np.asarray(reco_E),
        "mult": np.asarray(mult),
    }

    eff = np.isfinite(d["reco_E"]).mean()
    print(f"  reco efficiency: {eff:.3f}")
    print(f"  true E range: {d['true_E'].min():.1f} - "
          f"{d['true_E'].max():.1f} GeV")

    return d


# ============================================================
# Main
# ============================================================

if len(sys.argv) < 3:
    sys.exit("usage: compare_two.py FAST_REC.root FULL_REC.root")

fast = process(sys.argv[1], "fast sim")
full = process(sys.argv[2], "full sim (Geant4)")

# reconstructed photon energy, matched events only
ef = fast["reco_E"][np.isfinite(fast["reco_E"])]
eg = full["reco_E"][np.isfinite(full["reco_E"])]


# ------------------------------------------------------------
# FIGURE 1: reco-energy histograms (normalised) + fast/full ratio
# ------------------------------------------------------------

emax = max(ef.max(), eg.max()) * 1.05
edges = np.linspace(0, emax, N_EBINS + 1)
centres = 0.5 * (edges[:-1] + edges[1:])

# raw counts (for errors) and totals (for normalisation)
nf, _ = np.histogram(ef, bins=edges)
ng, _ = np.histogram(eg, bins=edges)
Nf, Ng = nf.sum(), ng.sum()

# fraction of events per bin
hf = nf / Nf
hg = ng / Ng

fig, (ax_main, ax_ratio) = plt.subplots(
    2, 1, sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

ax_main.step(centres, hf, where="mid", label=fast["label"])
ax_main.step(centres, hg, where="mid", ls="--", label=full["label"])
ax_main.set_ylabel("fraction of events")
ax_main.legend()

# ratio only where both bins are populated
both = (nf > 0) & (ng > 0)
ratio = hf[both] / hg[both]
# Poisson relative errors added in quadrature
ratio_unc = ratio * np.sqrt(1.0 / nf[both] + 1.0 / ng[both])

ax_ratio.axhline(1.0, color="r", ls="--")
ax_ratio.errorbar(centres[both], ratio, yerr=ratio_unc,
                  fmt="ko", capsize=3, markersize=3)
ax_ratio.set_ylabel("fast / full")
ax_ratio.set_xlabel("reconstructed photon energy [GeV]")

fig.subplots_adjust(hspace=0.05)
fig.suptitle("Reconstructed photon energy: fast vs full sim")
fig.savefig("energy_fast_vs_full.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------
# FIGURE 2: multiplicity histograms (normalised) + ratio
# ------------------------------------------------------------

max_mult = int(max(fast["mult"].max(), full["mult"].max()))
mbins = np.arange(-0.5, max_mult + 1.5, 1)
mcentres = np.arange(0, max_mult + 1)

mf_raw, _ = np.histogram(fast["mult"], bins=mbins)
mg_raw, _ = np.histogram(full["mult"], bins=mbins)
mf = mf_raw / mf_raw.sum()
mg = mg_raw / mg_raw.sum()

fig, (ax_main, ax_ratio) = plt.subplots(
    2, 1, sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

ax_main.step(mcentres, mf, where="mid", label=fast["label"])
ax_main.step(mcentres, mg, where="mid", ls="--", label=full["label"])
ax_main.set_ylabel("fraction of events")
ax_main.legend()

both_m = (mf_raw > 0) & (mg_raw > 0)
mratio = mf[both_m] / mg[both_m]
mratio_unc = mratio * np.sqrt(1.0 / mf_raw[both_m] + 1.0 / mg_raw[both_m])

ax_ratio.axhline(1.0, color="r", ls="--")
ax_ratio.errorbar(mcentres[both_m], mratio, yerr=mratio_unc,
                  fmt="ko", capsize=3, markersize=3)
ax_ratio.set_ylabel("fast / full")
ax_ratio.set_xlabel("number of reconstructed photons")
ax_ratio.set_xticks(mcentres)

fig.subplots_adjust(hspace=0.05)
fig.suptitle("Photon reconstruction multiplicity: fast vs full sim")
fig.savefig("multiplicity_fast_vs_full.png", dpi=150, bbox_inches="tight")
plt.close(fig)


print("\nSaved:")
print("  energy_fast_vs_full.png")
print("  multiplicity_fast_vs_full.png")
 