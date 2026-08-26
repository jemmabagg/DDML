#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import ROOT

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

REC_FILE = "/home/baggjemm/ILDConfig/StandardConfig/production/dummyOutput_debug_REC.edm4hep.root"

# EDM4hep PDG code for photon
PHOTON_PDG = 22

# Number of bins for the energy-response plot
ENERGY_BINS = np.array([
    0, 5, 10, 20, 30, 40, 50,
    60, 70, 80, 100, 120, 150,
    200, 300
])


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_truth_photons(mc_particles):
    """
    Find incoming/primary photons in MCParticles.

    We select photons (PDG = 22) which have no parents.
    This is intended to pick out the incident photon rather
    than photons produced inside the shower.
    """

    photons = []

    for p in mc_particles:
        if abs(p.getPDG()) != PHOTON_PDG:
            continue

        # Primary particle: no parents
        if p.parents_size() == 0:
            photons.append(p)

    return photons


def get_reco_photons(pfos):
    """
    Select reconstructed photons from PandoraPFOs.
    EDM4hep particle type 22 corresponds to photon.
    """

    photons = []

    for pfo in pfos:
        if abs(pfo.getType()) == PHOTON_PDG:
            photons.append(pfo)

    return photons


# ------------------------------------------------------------
# Open file
# ------------------------------------------------------------

print(f"Opening: {REC_FILE}")

reader = ROOT.podio.create(ROOT.podio.FrameReader(REC_FILE))

events = reader.get(ROOT.podio.Frame.EVENT)

print(f"Number of events: {events.size()}")


# ------------------------------------------------------------
# Event loop
# ------------------------------------------------------------

truth_energies = []
reco_energies = []
reco_multiplicity = []

n_events_with_truth_photon = 0
n_events_with_reco_photon = 0

for i, event in enumerate(events):

    # Get collections
    try:
        mc_particles = event.get("MCParticles")
    except Exception:
        print("ERROR: Could not find MCParticles collection.")
        print("Check the collection names with:")
        print("  podio-dump", REC_FILE)
        sys.exit(1)

    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        print("ERROR: Could not find PandoraPFOs collection.")
        print("Check the collection names with:")
        print("  podio-dump", REC_FILE)
        sys.exit(1)

    # --------------------------------------------------------
    # Truth photon
    # --------------------------------------------------------

    truth_photons = get_truth_photons(mc_particles)

    if len(truth_photons) == 0:
        print(f"Event {i}: no primary truth photon found")
        continue

    if len(truth_photons) > 1:
        print(
            f"Event {i}: found {len(truth_photons)} primary photons; "
            "using the first one."
        )

    truth_photon = truth_photons[0]

    true_energy = truth_photon.getEnergy()

    # --------------------------------------------------------
    # Reconstructed photons
    # --------------------------------------------------------

    reco_photons = get_reco_photons(pfos)

    n_reco = len(reco_photons)

    # Total reconstructed photon energy in the event.
    #
    # This is preferable to simply taking the first photon,
    # because a shower can potentially be split into multiple
    # reconstructed photon PFOs.
    reco_energy = sum(p.getEnergy() for p in reco_photons)

    truth_energies.append(true_energy)
    reco_energies.append(reco_energy)
    reco_multiplicity.append(n_reco)

    n_events_with_truth_photon += 1

    if n_reco > 0:
        n_events_with_reco_photon += 1

    print(
        f"Event {i:3d}: "
        f"true E = {true_energy:8.3f} GeV, "
        f"reco photons = {n_reco:2d}, "
        f"reco E = {reco_energy:8.3f} GeV"
    )


# ------------------------------------------------------------
# Convert to numpy
# ------------------------------------------------------------

truth_energies = np.asarray(truth_energies)
reco_energies = np.asarray(reco_energies)
reco_multiplicity = np.asarray(reco_multiplicity)


if len(truth_energies) == 0:
    raise RuntimeError("No truth photons found.")


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("Photon reconstruction summary")
print("=" * 60)

print(f"Events with truth photon:       {n_events_with_truth_photon}")
print(f"Events with >=1 reco photon:    {n_events_with_reco_photon}")

efficiency = (
    n_events_with_reco_photon / n_events_with_truth_photon
)

print(f"Photon reconstruction efficiency: {efficiency:.3f}")
print()

print("Reconstructed photon multiplicity:")
for n in range(reco_multiplicity.max() + 1):
    count = np.sum(reco_multiplicity == n)
    print(f"  {n:2d} photons: {count:3d} events")


# ============================================================
# Plot 1: reconstructed photon multiplicity
# ============================================================

plt.figure(figsize=(7, 5))

max_mult = int(reco_multiplicity.max())

bins = np.arange(-0.5, max_mult + 1.5, 1)

plt.hist(
    reco_multiplicity,
    bins=bins,
    edgecolor="black"
)

plt.xlabel("Number of reconstructed photons")
plt.ylabel("Number of events")
plt.title("Reconstructed photon multiplicity")

plt.xticks(range(max_mult + 1))
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "photon_reco_multiplicity.png",
    dpi=200
)

plt.show()


# ============================================================
# Plot 2: reconstructed / true energy vs true energy
# ============================================================

# Only calculate response where at least one photon
# was reconstructed.
has_reco = reco_energies > 0

response = reco_energies[has_reco] / truth_energies[has_reco]
response_true_energy = truth_energies[has_reco]


bin_centres = []
mean_response = []
std_response = []
n_per_bin = []

for low, high in zip(ENERGY_BINS[:-1], ENERGY_BINS[1:]):

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
# Energy response plot
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
plt.ylabel(r"Reconstructed energy / true energy")
plt.title("Photon energy response")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "photon_energy_response.png",
    dpi=200
)

plt.show()


# ============================================================
# Plot 3: optional — response distribution
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

plt.xlabel(r"Reconstructed energy / true energy")
plt.ylabel("Number of events")
plt.title("Photon energy response distribution")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "photon_energy_response_distribution.png",
    dpi=200
)

plt.show()