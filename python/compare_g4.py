#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from podio import root_io


# ============================================================
# Configuration
# ============================================================

REC_FILE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/baggjemm/ILDConfig/StandardConfig/production/"
         "finalOutput_debug_REC.edm4hep.root"
)

PHOTON_PDG = 22

# True-energy bins in GeV
ENERGY_BINS = np.array([
    0, 5, 10, 20, 30, 40, 50,
    60, 70, 80, 100, 120, 150, 200,
])


# ============================================================
# Open reconstructed file
# ============================================================

print(f"Opening: {REC_FILE}")

reader = root_io.Reader(REC_FILE)
events = reader.get("events")

print(f"Number of events: {len(events)}")


# ============================================================
# Storage
# ============================================================

truth_energies = []
reco_energies = []
reco_multiplicities = []

matched_events = 0
unmatched_events = 0


# ============================================================
# Event loop
# ============================================================

for i, event in enumerate(events):

    mc_particles = event.get("MCParticles")
    pfos = event.get("PandoraPFOs")


    # --------------------------------------------------------
    # Find primary truth photon
    # --------------------------------------------------------

    truth_photons = []

    for particle in mc_particles:

        if abs(particle.getPDG()) != PHOTON_PDG:
            continue

        # Incoming photon should have no parents
        if particle.parents_size() == 0:
            truth_photons.append(particle)


    if len(truth_photons) == 0:
        print(f"Event {i}: no primary truth photon found")
        continue


    # We expect one incoming photon
    truth_photon = truth_photons[0]
    true_energy = truth_photon.getEnergy()


    # --------------------------------------------------------
    # Reconstructed photons in this event
    # --------------------------------------------------------

    reco_photons = [
        p for p in pfos
        if abs(p.getPDG()) == PHOTON_PDG
    ]

    n_reco = len(reco_photons)

    truth_energies.append(true_energy)
    reco_multiplicities.append(n_reco)


    # --------------------------------------------------------
    # Store event
    #
    # reco energy = SUM over all reco photons, so a shower that
    # fragments into several PFOs still has its energy counted.
    # (Use max(...) instead for leading-photon energy.)
    # --------------------------------------------------------

    if n_reco > 0:

        reco_energy = sum(
            p.getEnergy() for p in reco_photons
        )

        reco_energies.append(reco_energy)
        matched_events += 1

        print(
            f"Event {i:3d}: "
            f"true E = {true_energy:8.3f} GeV | "
            f"reco photons = {n_reco:2d} | "
            f"reco E = {reco_energy:8.3f} GeV"
        )

    else:

        reco_energies.append(np.nan)
        unmatched_events += 1

        print(
            f"Event {i:3d}: "
            f"true E = {true_energy:8.3f} GeV | "
            f"reco photons = 0 | NO RECO"
        )


# ============================================================
# Convert to numpy
# ============================================================

truth_energies = np.asarray(truth_energies)
reco_energies = np.asarray(reco_energies)
reco_multiplicities = np.asarray(reco_multiplicities)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 65)
print("Photon reconstruction summary")
print("=" * 65)

print(f"Events with truth photon:        {len(truth_energies)}")
print(f"Events with >=1 reco photon:     {matched_events}")
print(f"Events with no reco photon:      {unmatched_events}")

if len(truth_energies) > 0:
    efficiency = matched_events / len(truth_energies)
    print(f"Photon reconstruction efficiency: {efficiency:.3f}")

print()
print("Photon multiplicity:")

for n in range(int(reco_multiplicities.max()) + 1):
    count = np.sum(reco_multiplicities == n)
    fraction = count / len(reco_multiplicities)
    print(
        f"  {n:2d} reconstructed photons: "
        f"{count:3d} events ({fraction:.1%})"
    )


# ============================================================
# Matched events for energy analysis
# ============================================================

matched = np.isfinite(reco_energies)

true_E = truth_energies[matched]
reco_E = reco_energies[matched]

if len(true_E) == 0:
    print("\nNo events with a reconstructed photon - nothing to plot.")
    sys.exit(0)

response = reco_E / true_E


# ============================================================
# FIGURE 1
#
# Main:  photon energy distributions (reco vs truth)
# Ratio: mean reco/true response, binned in true energy
# ============================================================

fig, (ax_main, ax_ratio) = plt.subplots(
    2, 1, sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

# --- main panel: energy distributions as step histograms ---
max_energy = max(np.max(true_E), np.max(reco_E))
hist_bins = np.linspace(0, max_energy * 1.05, 20)
centres = 0.5 * (hist_bins[:-1] + hist_bins[1:])

reco_counts, _ = np.histogram(reco_E, bins=hist_bins)
truth_counts, _ = np.histogram(true_E, bins=hist_bins)

ax_main.step(centres, reco_counts, where="mid", label="ILD reconstructed")
ax_main.step(centres, truth_counts, where="mid", ls="--", label="Geant4 truth")

ax_main.set_ylabel("events")
ax_main.legend()

# --- ratio panel: mean reco/true per true-energy bin ---
bin_centres = []
mean_response = []
sem_response = []

for low, high in zip(ENERGY_BINS[:-1], ENERGY_BINS[1:]):
    values = response[(true_E >= low) & (true_E < high)]
    if len(values) == 0:
        continue
    bin_centres.append(0.5 * (low + high))
    mean_response.append(np.mean(values))
    sem_response.append(np.std(values) / np.sqrt(len(values)))  # SEM on the mean

bin_centres = np.asarray(bin_centres)
mean_response = np.asarray(mean_response)
sem_response = np.asarray(sem_response)

ax_ratio.axhline(1.0, color="r", ls="--")
ax_ratio.errorbar(bin_centres, mean_response, yerr=sem_response,
                  fmt="ko", capsize=3, markersize=3)

ax_ratio.set_ylabel(r"$E_{\mathrm{reco}} / E_{\mathrm{true}}$")
ax_ratio.set_xlabel("photon energy [GeV]")

fig.subplots_adjust(hspace=0.05)
fig.suptitle("Photon energy response")
fig.savefig("photon_energy_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)


#============================================================
# FIGURE 2 : photon reconstruction multiplicity
# ============================================================

fig, ax = plt.subplots()

max_mult = int(reco_multiplicities.max())
bins = np.arange(-0.5, max_mult + 1.5, 1)

ax.hist(reco_multiplicities, bins=bins, histtype="step", lw=2)

ax.set_xlabel("number of reconstructed photons")
ax.set_ylabel("events")
ax.set_xticks(np.arange(0, max_mult + 1))

fig.suptitle("Photon reconstruction multiplicity")
fig.savefig("photon_reco_multiplicity.png", dpi=150, bbox_inches="tight")
plt.close(fig)