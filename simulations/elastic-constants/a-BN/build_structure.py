from ase.io import read, write

# --------------------------------------------------
# Read CIF file
# --------------------------------------------------
atoms = read("aBN_bulk_mp-1244991.cif")

print("Number of atoms:", len(atoms))
print("Cell:")
print(atoms.cell)

# --------------------------------------------------
# OPTIONAL: make supercell (recommended)
# Uncomment if needed
# --------------------------------------------------
atoms = atoms.repeat((5,5,5))

# --------------------------------------------------
# Write LAMMPS data file
# --------------------------------------------------
write(
    "aBN_bulk.data",
    atoms,
    format="lammps-data",
    atom_style="atomic"
)

print("LAMMPS data file written: aBN_bulk.data")

