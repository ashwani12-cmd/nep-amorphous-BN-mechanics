import numpy as np
from ase.io import read, write 
# --------------------------------------------------
# Read BULK h-BN from MP (mp-7991)
# --------------------------------------------------
atoms = read("hBN_monolayer_mp-7991.cif")

print("Original atoms:", len(atoms))
print("Original cell:\n", atoms.cell)

# --------------------------------------------------
# Sort atoms by z and keep ONE layer
# Bulk has 2 layers → keep lower half
# --------------------------------------------------
z = atoms.positions[:, 2]
order = np.argsort(z)

atoms = atoms[order[:len(atoms)//2]]

print("Atoms after monolayer extraction:", len(atoms))

# --------------------------------------------------
# Add vacuum along z
# --------------------------------------------------
vacuum = 20.0  # Å
cell = atoms.cell.copy()
cell[2] = [0, 0, vacuum]
atoms.set_cell(cell)
atoms.center(axis=2)

# --------------------------------------------------
# In-plane supercell (5×5)
# --------------------------------------------------
atoms = atoms.repeat((5, 5, 1))

# --------------------------------------------------
# Write LAMMPS data
# --------------------------------------------------
write(
    "hBN_monolayer.data",
    atoms,
    format="lammps-data",
    atom_style="atomic"
)

print("✅ Monolayer h-BN written: hBN_monolayer.data")

