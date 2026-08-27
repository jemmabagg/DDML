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
# Top:    True vs reconstructed photon energy
# Bottom: Reco / true energy vs true energy
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 9),
    gridspec_kw={"height_ratios": [2, 1]},
    sharex=False,
)


# ------------------------------------------------------------
# Top panel: energy distributions
# ------------------------------------------------------------

max_energy = max(np.max(true_E), np.max(reco_E))
hist_bins = np.linspace(0, max_energy * 1.05, 20)

ax1.hist(
    true_E, bins=hist_bins,
    histtype="step", linewidth=2,
    label="Geant4 truth",
)

ax1.hist(
    reco_E, bins=hist_bins,
    histtype="step", linewidth=2,
    label="ILD reconstructed",
)

ax1.set_ylabel("Number of events")
ax1.set_title("Photon energy reconstruction")
ax1.legend()


# ------------------------------------------------------------
# Bottom panel: energy response
# ------------------------------------------------------------

bin_centres = []
mean_response = []
std_response = []
n_per_bin = []

for low, high in zip(ENERGY_BINS[:-1], ENERGY_BINS[1:]):

    mask = (true_E >= low) & (true_E < high)
    values = response[mask]

    if len(values) == 0:
        continue

    bin_centres.append(0.5 * (low + high))
    mean_response.append(np.mean(values))
    std_response.append(np.std(values))
    n_per_bin.append(len(values))


bin_centres = np.asarray(bin_centres)
mean_response = np.asarray(mean_response)
std_response = np.asarray(std_response)
n_per_bin = np.asarray(n_per_bin)


# Mean response with standard deviation (= resolution) as error bars
ax2.errorbar(
    bin_centres, mean_response, yerr=std_response,
    fmt="ko", capsize=3, markersize=3,
)

# Perfect reconstruction
ax2.axhline(1.0, linestyle="--", linewidth=1.5)

ax2.set_xlabel("True photon energy [GeV]")
ax2.set_ylabel(r"$E_{\mathrm{reco}} / E_{\mathrm{true}}$")
ax2.set_title("Photon energy response")


plt.tight_layout()
plt.savefig("photon_energy_comparison.png", dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# FIGURE 2
#
# Photon reconstruction multiplicity
# ============================================================

plt.figure(figsize=(7, 5))

max_mult = int(reco_multiplicities.max())
bins = np.arange(-0.5, max_mult + 1.5, 1)

plt.hist(reco_multiplicities, bins=bins, edgecolor="black")

plt.xlabel("Number of reconstructed photons")
plt.ylabel("Number of events")
plt.title("Photon reconstruction multiplicity")
plt.xticks(np.arange(0, max_mult + 1))

plt.tight_layout()
plt.savefig("photon_reco_multiplicity.png", dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# Final output
# ============================================================

print()
print("=" * 65)
print("Plots saved:")
print("  photon_energy_comparison.png")
print("  photon_reco_multiplicity.png")
print("=" * 65)