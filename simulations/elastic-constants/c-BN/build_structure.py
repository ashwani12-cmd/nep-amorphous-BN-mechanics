from ase.io import read, write

# --------------------------------------------------
# Read CIF file
# --------------------------------------------------
atoms = read("cBN_mp-1639.cif")

print("Number of atoms:", len(atoms))
print("Cell:")
print(atoms.cell)

# --------------------------------------------------
# OPTIONAL: make supercell (recommended)
# Uncomment if needed
# --------------------------------------------------
atoms = atoms.repeat((12,12,12))

# --------------------------------------------------
# Write LAMMPS data file
# --------------------------------------------------
write(
    "c-BN_bulk.data",
    atoms,
    format="lammps-data",
    atom_style="atomic"
)

print("LAMMPS data file written: c-BN_bulk.data")

