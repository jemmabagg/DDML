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
         "dummyOutput_debug_REC.edm4hep.root"
)

PHOTON_PDG = 22

# Energy bins in GeV
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
# Open file
# ============================================================

print(f"Opening: {REC_FILE}")

reader = root_io.Reader(REC_FILE)
events = reader.get("events")

print(f"Number of events: {len(events)}")


# ============================================================
# Loop over events
# ============================================================

truth_energies = []
reco_energies = []
reco_multiplicities = []

for i, event in enumerate(events):

    # --------------------------------------------------------
    # Truth photon
    # --------------------------------------------------------

    mc_particles = event.get("MCParticles")

    truth_photons = []

    for particle in mc_particles:

        if abs(particle.getPDG()) != PHOTON_PDG:
            continue

        # Primary photon = photon with no parents
        if particle.parents_size() == 0:
            truth_photons.append(particle)

    if len(truth_photons) == 0:
        print(f"Event {i}: no primary truth photon found")
        continue

    if len(truth_photons) > 1:
        print(
            f"Event {i}: found {len(truth_photons)} "
            "primary photons; using first."
        )

    truth_photon = truth_photons[0]

    true_energy = truth_photon.getEnergy()


    # --------------------------------------------------------
    # Reconstructed photons
    # --------------------------------------------------------

    pfos = event.get("PandoraPFOs")

    reco_photons = []

    for pfo in pfos:

        if abs(pfo.getType()) == PHOTON_PDG:
            reco_photons.append(pfo)

    n_reco = len(reco_photons)

    # Total reconstructed photon energy
    reco_energy = sum(
        pfo.getEnergy()
        for pfo in reco_photons
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    truth_energies.append(true_energy)
    reco_energies.append(reco_energy)
    reco_multiplicities.append(n_reco)


    print(
        f"Event {i:3d}: "
        f"true E = {true_energy:8.3f} GeV | "
        f"reco photons = {n_reco:2d} | "
        f"reco E = {reco_energy:8.3f} GeV"
    )


# ============================================================
# Convert to numpy
# ============================================================

truth_energies = np.asarray(truth_energies)
reco_energies = np.asarray(reco_energies)
reco_multiplicities = np.asarray(reco_multiplicities)


if len(truth_energies) == 0:
    raise RuntimeError("No truth photons were found.")


# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("Photon reconstruction summary")
print("=" * 60)

print(f"Events with truth photon:    {len(truth_energies)}")
print(
    f"Events with >=1 reco photon: "
    f"{np.sum(reco_multiplicities > 0)}"
)

efficiency = np.mean(reco_multiplicities > 0)

print(
    f"Reconstruction efficiency:   "
    f"{efficiency:.3f}"
)

print()
print("Reconstructed photon multiplicity:")

for n in range(int(reco_multiplicities.max()) + 1):

    count = np.sum(reco_multiplicities == n)

    print(
        f"  {n:2d} photons: {count:3d} events"
    )


# ============================================================
# Plot 1: Photon multiplicity
# ============================================================

plt.figure(figsize=(7, 5))

max_mult = int(reco_multiplicities.max())

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
plt.title("Reconstructed photon multiplicity")

plt.xticks(
    np.arange(0, max_mult + 1)
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "photon_reco_multiplicity.png",
    dpi=200
)

plt.close()


# ============================================================
# Plot 2: Reconstructed / true energy
# ============================================================

# Only events with at least one reconstructed photon
has_reco = reco_energies > 0

response = (
    reco_energies[has_reco]
    / truth_energies[has_reco]
)

response_true_energy = truth_energies[has_reco]


# ------------------------------------------------------------
# Calculate mean response in true-energy bins
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
        (response_true_energy >= low)
        & (response_true_energy < high)
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


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.errorbar(
    bin_centres,
    mean_response,
    yerr=std_response,
    fmt="o",
    capsize=3,
    label="ILD reconstruction"
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label="Reco = true"
)

plt.xlabel("True photon energy [GeV]")
plt.ylabel("Reconstructed energy / true energy")
plt.title("Photon energy response")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "photon_energy_response.png",
    dpi=200
)

plt.close()


# ============================================================
# Plot 3: Response distribution
# ============================================================

plt.figure(figsize=(7, 5))

plt.hist(
    response,
    bins=30,
    edgecolor="black"
)

plt.axvline(
    1.0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Reconstructed energy / true energy")
plt.ylabel("Number of events")
plt.title("Photon energy response distribution")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "photon_energy_response_distribution.png",
    dpi=200
)

plt.close()


print()
print("Plots saved:")
print("  photon_reco_multiplicity.png")
print("  photon_energy_response.png")
print("  photon_energy_response_distribution.png")