# Processed data

## `tension/stress_strain_rho_<density>.dat`

Uniaxial-tension output for 5×5×5 a-BN cells at the labelled density (g/cm³),
one line per strain increment, written by the LAMMPS decks in
`simulations/tension/`. Whitespace-delimited `key: value` pairs:

| Field | Meaning | Units |
|-------|---------|-------|
| `lz`   | current box length along the loading axis | Å |
| `Vol`  | current cell volume | Å³ |
| `PE`   | total potential energy | eV |
| `Pxx`, `Pyy`, `Pzz` | stress components (tension positive; `-pressure/10⁴`) | GPa |

Engineering strain is `(lz - lz₀) / lz₀` with `lz₀` the first row; the tensile
stress is `Pzz`.

`tension/small-cell/Strain_Stress_run{1,2,3}.dat` — same format, for the smaller
`a-BN_{0.79,1.3,2.1}.data` cells (`simulations/tension/Tension.in`).

## `rdf/rdf_<phase>.dat`, `adf/adf_<phase>.dat`

LAMMPS `fix ave/time` output for `compute rdf` (radial) and the bond-angle
distribution, time-averaged over the production run at 0 GPa / 300 K / NVT for
amorphous (`a-BN`), cubic (`c-BN`), and hexagonal (`h-BN`) boron nitride.
Header lines start with `#`; data columns:

| Column | Meaning |
|--------|---------|
| 1 | bin index |
| 2 | bin coordinate (distance in Å for RDF, angle in degrees for ADF) |
| 3 | g(r) / normalised P(θ) |
| 4 | running coordination number (RDF only) |
