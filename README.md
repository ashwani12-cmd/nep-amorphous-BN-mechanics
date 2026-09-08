# NEP for the Mechanical Response of Amorphous Boron Nitride

Companion code, input decks, trained potential, processed data, and figures for:

> **Benchmarking Neural Equivariant Potentials for the Mechanical Response of Amorphous Boron Nitride**
> Ashwani Kushwaha, Amit Singh
> Department of Mechanical Engineering, IIT Bombay, Mumbai 400076, India

We train a neuroevolution potential (NEP) for the B–N system on a publicly available
DFT reference dataset and benchmark it against previously reported deep-learning
results for amorphous boron nitride (a-BN). The potential is used in LAMMPS to
compute elastic constants, radial/angular distribution functions, and uniaxial
stress–strain response of a-BN across a range of densities.

---

## Repository layout

```
nep-amorphous-BN-mechanics/
├── potential/                 Trained NEP potential (nep.txt) — use this in LAMMPS / GPUMD
├── training/                  NEP training input, learning curve, and energy/force/stress parity data
├── structures/
│   ├── cif/                   Source crystal structures from the Materials Project
│   └── lammps-data/           LAMMPS data files: a-BN cells at 6 densities, plus c-BN and h-BN references
├── simulations/
│   ├── tension/               LAMMPS input decks for uniaxial tension + batch driver
│   └── elastic-constants/     LAMMPS `in.elastic` protocol for a-BN, c-BN, h-BN
├── scripts/                   Structure builders and plotting scripts (Python)
├── data/                      Processed results: stress–strain curves, RDF, ADF  (see data/README.md)
├── figures/                   Final paper-quality figures
├── notebooks/                 Jupyter notebooks for post-processing
└── manuscript/                LaTeX source (Elsevier elsarticle) + figures
```

## Requirements

**Analysis / plotting** (Python):

```bash
pip install -r requirements.txt
```

**Simulations** (external, not bundled):

| Tool | Used for | Notes |
|------|----------|-------|
| [GPUMD](https://github.com/brucefan1983/GPUMD) (`nep` executable) | Training the NEP potential | GPU required |
| [LAMMPS](https://www.lammps.org/) with the `NEP` pair style | MD: tension, elastic constants, RDF/ADF | Build with the `ML-NEP` / GPUMD `nep` interface |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | Building LAMMPS data files from CIF | Installed via `requirements.txt` |

## Reference DFT dataset

The NEP is trained on the publicly available DFT dataset of boron nitride
configurations (hexagonal, cubic, and amorphous BN; energies, forces, virials)
reported by:

> S.-P. Ju, C.-C. Huang, H.-Y. Chen, *Illuminating the mechanical responses of
> amorphous boron nitride through deep learning: A molecular dynamics study*,
> Computational Materials Science **232** (2024) 112664.

The dataset is **not redistributed here**. `training/train.xyz` and
`training/test.xyz` are the train/test splits in extended-XYZ form as fed to
GPUMD; obtain the original data from the reference above.

## Reproducing the results

### 1. Train the potential (optional — `potential/nep.txt` is provided)

```bash
cd training
# needs train.xyz + nep.in in the working directory
nep            # or: sbatch submit.sh   (SLURM + GPU)
```

`nep.in` — NEP4 with ZBL, cutoffs 6/4 Å, for elements `B N`, 300k generations.
`loss.out` columns: `gen  total  RMSE_E_tr  RMSE_F_tr  RMSE_V_tr  ...`.
`{energy,force,stress,virial}_{train,test}.out` are two-column
`NEP_prediction  DFT_reference` files used for the parity plots.

### 2. Build a-BN structures at target densities

```bash
cd scripts
python build_density_structures.py   # reads structures/cif/aBN_bulk_mp-1244991.cif
```

Generates `aBN_density_{0.80,1.00,1.30,1.60,2.00,2.30}.data` (5×5×5 supercell
isotropically scaled to each target density in g/cm³). Copies of the outputs are
already in `structures/lammps-data/`.

### 3. Uniaxial tension

```bash
cd simulations/tension
cp ../../potential/nep.txt .
cp ../../structures/lammps-data/aBN_density_1.00.data .
lmp -in tension_from_density_cell.in     # per-density deck (sets B/N masses explicitly)
# or Tension.in for the small-cell a-BN_*.data structures
```

Each run relaxes the cell, equilibrates with NPT, then applies incremental
tensile strain along *z*, appending one line per step to `Strain_Stress.dat`.
`run_all.sh` loops the six density folders.

### 4. Elastic constants

```bash
cd simulations/elastic-constants/c-BN     # or h-BN
cp ../../../potential/nep.txt .
lmp -in in.elastic                        # standard LAMMPS elastic-constant protocol
```

`in.elastic` pulls in `init.mod`, `potential.mod`, `displace.mod`. For a-BN,
first build a bulk cell with `a-BN/build_structure.py`.

### 5. Figures

```bash
cd scripts
python plot_stress_strain.py          # per-density σ–ε panels
python plot_moduli_vs_density.py      # E, strength, toughness, ductility vs density
python plot_atomic_strain_frames.py 1.00   # von Mises atomic-strain maps (needs raw .cfg frames)
```

The notebooks in `notebooks/` reproduce the tensile analysis and the h-BN
reference checks end-to-end.

## What is *not* in this repository

Raw MD output — per-frame dump files (`*.cfg`), `log.lammps`, `HEA.log`,
`out.dat`, SLURM logs — is excluded (see `.gitignore`); it is bulky and fully
regenerable from the inputs here. `scripts/plot_atomic_strain_frames.py` and
`plot_strain_panels_by_species.py` need those frames, so run the tension
simulations first if you want the atomic-strain maps.

## License

Code, input decks, and documentation are released under the MIT License
(`LICENSE`). The manuscript text and figures under `manuscript/` are © the
authors. The reference DFT dataset is the property of its original authors.

## Citing

See `CITATION.cff`. Please cite the paper above and the NEP method
(Fan et al., *J. Chem. Phys.* / *Phys. Rev. B*) if you use the potential.
