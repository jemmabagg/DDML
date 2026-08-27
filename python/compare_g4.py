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
    0,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    100,
    120,
    150,
    200,
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
    links = event.get("MCTruthRecoLink")


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

        print(
            f"Event {i}: no primary truth photon found"
        )

        continue


    # We expect one incoming photon
    truth_photon = truth_photons[0]

    true_energy = truth_photon.getEnergy()


    # --------------------------------------------------------
    # Count reconstructed photons
    # --------------------------------------------------------

    reco_photons = []

    for pfo in pfos:

        if abs(pfo.getPDG()) == PHOTON_PDG:
            reco_photons.append(pfo)

    n_reco = len(reco_photons)


    # --------------------------------------------------------
    # Find reconstructed particle matched to truth photon
    # --------------------------------------------------------

    best_reco = None
    best_weight = -1.0

    for link in links:

        reco = link.getFrom()
        truth = link.getTo()

        # Only consider links to our incoming truth photon
        if truth.getPDG() != truth_photon.getPDG():
            continue

        # Make sure this is actually our truth particle.
        #
        # Comparing the object's identity is not always reliable
        # through cppyy, so compare its energy as well.
        if not np.isclose(
            truth.getEnergy(),
            truth_photon.getEnergy(),
            rtol=1e-6,
            atol=1e-6
        ):
            continue

        weight = link.getWeight()

        if weight > best_weight:

            best_weight = weight
            best_reco = reco


    # --------------------------------------------------------
    # Store event
    # --------------------------------------------------------

    truth_energies.append(true_energy)
    reco_multiplicities.append(n_reco)


    if best_reco is not None:

        reco_energy = best_reco.getEnergy()

        reco_energies.append(reco_energy)

        matched_events += 1

        print(
            f"Event {i:3d}: "
            f"true E = {true_energy:8.3f} GeV | "
            f"reco photons = {n_reco:2d} | "
            f"matched reco E = {reco_energy:8.3f} GeV | "
            f"link weight = {best_weight:.3g}"
        )

    else:

        reco_energies.append(np.nan)

        unmatched_events += 1

        print(
            f"Event {i:3d}: "
            f"true E = {true_energy:8.3f} GeV | "
            f"reco photons = {n_reco:2d} | "
            f"NO MATCH"
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

print(
    f"Events with truth photon:       "
    f"{len(truth_energies)}"
)

print(
    f"Events with matched reco photon: "
    f"{matched_events}"
)

print(
    f"Events without matched photon:   "
    f"{unmatched_events}"
)

if len(truth_energies) > 0:

    efficiency = matched_events / len(truth_energies)

    print(
        f"Photon reconstruction efficiency: "
        f"{efficiency:.3f}"
    )

print()
print("Photon multiplicity:")

for n in range(int(reco_multiplicities.max()) + 1):

    count = np.sum(
        reco_multiplicities == n
    )

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

response = reco_E / true_E


# ============================================================
# FIGURE 1
#
# Top:
#   True vs reconstructed photon energy
#
# Bottom:
#   Reco / true energy vs true energy
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 9),
    gridspec_kw={"height_ratios": [2, 1]},
    sharex=False
)


# ------------------------------------------------------------
# Top panel: energy distributions
# ------------------------------------------------------------

# Use common energy range
max_energy = max(
    np.max(true_E),
    np.max(reco_E)
)

hist_bins = np.linspace(
    0,
    max_energy * 1.05,
    20
)

ax1.hist(
    true_E,
    bins=hist_bins,
    histtype="step",
    linewidth=2,
    label="Geant4 truth"
)

ax1.hist(
    reco_E,
    bins=hist_bins,
    histtype="step",
    linewidth=2,
    label="ILD reconstructed"
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

for low, high in zip(
    ENERGY_BINS[:-1],
    ENERGY_BINS[1:]
):

    mask = (
        (true_E >= low)
        & (true_E < high)
    )

    values = response[mask]

    if len(values) == 0:
        continue

    centre = 0.5 * (low + high)

    bin_centres.append(centre)
    mean_response.append(np.mean(values))
    std_response.append(np.std(values))
    n_per_bin.append(len(values))


bin_centres = np.asarray(bin_centres)
mean_response = np.asarray(mean_response)
std_response = np.asarray(std_response)
n_per_bin = np.asarray(n_per_bin)


# Plot mean response with standard deviation
ax2.errorbar(bin_centres,mean_response,yerr=std_response,fmt="ko", capsize=3, markersize=3)

# Perfect reconstruction
ax2.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
)

ax2.set_xlabel("True photon energy [GeV]")
ax2.set_ylabel(r"$E_{\mathrm{reco}} / E_{\mathrm{true}}$")
ax2.set_title("Photon energy response")


plt.tight_layout()

plt.savefig(
    "photon_energy_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
#
# Photon reconstruction multiplicity
# ============================================================

plt.figure(figsize=(7, 5))

max_mult = int(
    reco_multiplicities.max()
)

bins = np.arange(
    -0.5,
    max_mult + 1.5,
    1
)

plt.hist(
    reco_multiplicities,
    bins=bins,
    edgecolor="black"
)

plt.xlabel("Number of reconstructed photons")
plt.ylabel("Number of events")
plt.title("Photon reconstruction multiplicity")

plt.xticks(
    np.arange(
        0,
        max_mult + 1
    )
)

plt.tight_layout()

plt.savefig(
    "photon_reco_multiplicity.png",
    dpi=200,
    bbox_inches="tight"
)

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