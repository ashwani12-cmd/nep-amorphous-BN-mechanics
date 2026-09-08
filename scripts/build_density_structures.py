from ase.io import read, write
import numpy as np

# -----------------------------
# Read CIF and make supercell
# -----------------------------
atoms = read("aBN_bulk_mp-1244991.cif")
atoms = atoms.repeat((5,5,5))

mass = atoms.get_masses().sum()       # amu
vol0 = atoms.get_volume()             # A^3
rho0 = mass * 1.66054 / vol0          # g/cm^3

print("Atoms:", len(atoms))
print("Original density (g/cm^3):", rho0)

# -----------------------------
# Target densities (g/cm^3)
# -----------------------------
target_densities = [0.8, 1.0, 1.3, 1.6, 2.0, 2.3]

# -----------------------------
# Generate structures
# -----------------------------
for rho_t in target_densities:

    a = atoms.copy()

    scale = (rho0 / rho_t) ** (1.0 / 3.0)

    a.set_cell(a.cell * scale, scale_atoms=True)

    mass = a.get_masses().sum()
    vol = a.get_volume()
    rho = mass * 1.66054 / vol

    fname = f"aBN_density_{rho_t:.2f}.data"
    write(fname, a, format="lammps-data", atom_style="atomic")

    print(f"\nWritten: {fname}")
    print(f"Target density: {rho_t:.2f} g/cm^3")
    print(f"Actual density: {rho:.3f} g/cm^3")

