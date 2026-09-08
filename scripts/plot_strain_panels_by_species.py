"""
plot_shear_strain_v2.py — with B/N atom-type analysis
Adds: per-species shear strain statistics, B vs N comparison plots, 1.5 colorbar tick
"""
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────
REF_FILE   = '00stress_1.cfg'
FRAMES     = [130, 261, 289, 318, 359, 421]   # ρ=0.80 — change per density
LABELS     = ['(I)', '(II)\nYield', '(III)', '(IV)\nUltimate', '(V)', '(VI)\nFracture']
CUTOFF     = 4.5
CMAP_MAX   = 1.5
TYPE_NAMES = {1: 'B', 2: 'N'}      # mass 1=10.81 B, mass 2=14.007 N from Tension.in
OUTPUT     = 'shear_strain_panels.png'
OUTPUT_BN  = 'shear_strain_B_vs_N.png'

# ────────────────────────────────────────────
def read_dump(fname):
    """Returns positions (sorted by id), types, box lengths."""
    with open(fname) as f:
        lines = f.readlines()
    i, natoms, box = 0, 0, np.zeros((3,2))
    pos, types = None, None
    while i < len(lines):
        if lines[i].startswith('ITEM: NUMBER OF ATOMS'):
            natoms = int(lines[i+1]); i += 2
        elif lines[i].startswith('ITEM: BOX BOUNDS'):
            for d in range(3):
                parts = lines[i+1+d].split()
                box[d] = [float(parts[0]), float(parts[1])]
            i += 4
        elif lines[i].startswith('ITEM: ATOMS'):
            cols = lines[i].split()[2:]
            iid, ity = cols.index('id'), cols.index('type')
            ix, iy, iz = cols.index('x'), cols.index('y'), cols.index('z')
            data = np.array([lines[i+1+k].split() for k in range(natoms)], dtype=float)
            order = np.argsort(data[:, iid])
            data = data[order]
            pos = data[:, [ix, iy, iz]]
            types = data[:, ity].astype(int)
            i += 1 + natoms
        else:
            i += 1
    L = box[:,1] - box[:,0]
    return pos - box[:,0], types, L

def pbc_disp(d, L):
    return d - L * np.round(d / L)

def atomic_shear_strain(pos_ref, L_ref, pos_cur, L_cur, cutoff):
    N = len(pos_ref)
    pr = pos_ref % L_ref
    tree = cKDTree(pr, boxsize=L_ref)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    nbrs = [[] for _ in range(N)]
    for a, b in pairs:
        nbrs[a].append(b); nbrs[b].append(a)
    eta_vm = np.zeros(N)
    I3 = np.eye(3)
    for i in range(N):
        nb = nbrs[i]
        if len(nb) < 3: continue
        d_ref = pbc_disp(pos_ref[nb] - pos_ref[i], L_ref)
        d_cur = pbc_disp(pos_cur[nb] - pos_cur[i], L_cur)
        V = d_ref.T @ d_ref
        W = d_ref.T @ d_cur
        try: J = np.linalg.solve(V, W)
        except np.linalg.LinAlgError: continue
        eta = 0.5 * (J @ J.T - I3)
        eta_vm[i] = np.sqrt(
            eta[0,1]**2 + eta[0,2]**2 + eta[1,2]**2 +
            ((eta[0,0]-eta[1,1])**2 + (eta[0,0]-eta[2,2])**2 + (eta[1,1]-eta[2,2])**2)/6)
    return eta_vm

# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────
print("Reading reference...")
pos_ref, types, L_ref = read_dump(REF_FILE)
nB, nN = (types==1).sum(), (types==2).sum()
print(f"  {len(pos_ref)} atoms: {nB} B, {nN} N")

STRAINS = [f*0.002 for f in FRAMES]

# Storage for B/N statistics
stats = {'strain': [], 'mean_B': [], 'mean_N': [], 'std_B': [], 'std_N': [],
         'frac_hi_B': [], 'frac_hi_N': [], 'eta_all': []}
HI_THRESHOLD = 0.8   # "highly strained" atom definition

# ── Figure 1: strain panels (same as before, + 1.5 tick) ──
fig, axes = plt.subplots(1, len(FRAMES), figsize=(2.3*len(FRAMES), 7))
plt.subplots_adjust(bottom=0.14, wspace=0.05, top=0.90)
cmap = matplotlib.colormaps['jet']
norm = Normalize(0, CMAP_MAX)

for ax, frame, strain, label in zip(axes, FRAMES, STRAINS, LABELS):
    fname = f'00stress_{frame}.cfg'
    print(f"Processing {fname}...")
    pos, _, L = read_dump(fname)
    eta = atomic_shear_strain(pos_ref, L_ref, pos, L, CUTOFF)

    # Per-species stats
    eB, eN = eta[types==1], eta[types==2]
    stats['strain'].append(strain)
    stats['mean_B'].append(eB.mean());  stats['mean_N'].append(eN.mean())
    stats['std_B'].append(eB.std());    stats['std_N'].append(eN.std())
    stats['frac_hi_B'].append((eB > HI_THRESHOLD).mean()*100)
    stats['frac_hi_N'].append((eN > HI_THRESHOLD).mean()*100)
    stats['eta_all'].append((eB.copy(), eN.copy()))

    order = np.argsort(pos[:,1])
    ax.scatter(pos[order,0], pos[order,2], c=eta[order], cmap=cmap, norm=norm,
               s=6, edgecolors='none')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.5)
    ax.set_title(f'{label} {strain:.3f}', fontsize=9, fontweight='bold')

cax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=cax, orientation='horizontal')
cb.set_label('Shear Strain', fontsize=11)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])     # ← 1.5 included
plt.savefig(OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved → {OUTPUT}")

# ── Figure 2: B vs N analysis (3 panels) ──
fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.4))
st = stats

# (a) Mean shear strain per species vs applied strain
ax2[0].errorbar(st['strain'], st['mean_B'], yerr=st['std_B'], fmt='o-',
                color='#E24B4A', capsize=3, label='B', markersize=6)
ax2[0].errorbar(st['strain'], st['mean_N'], yerr=st['std_N'], fmt='s-',
                color='#378ADD', capsize=3, label='N', markersize=6)
ax2[0].set_xlabel('Applied strain ε'); ax2[0].set_ylabel('Mean atomic shear strain ⟨η⟩')
ax2[0].set_title('(a) Mean shear strain: B vs N'); ax2[0].legend(); ax2[0].grid(alpha=0.3)

# (b) Fraction of highly-strained atoms per species
ax2[1].plot(st['strain'], st['frac_hi_B'], 'o-', color='#E24B4A', label='B', markersize=6)
ax2[1].plot(st['strain'], st['frac_hi_N'], 's-', color='#378ADD', label='N', markersize=6)
ax2[1].set_xlabel('Applied strain ε')
ax2[1].set_ylabel(f'Atoms with η > {HI_THRESHOLD} (%)')
ax2[1].set_title('(b) Highly strained atom fraction'); ax2[1].legend(); ax2[1].grid(alpha=0.3)

# (c) Strain distribution histograms at the fracture frame
eB_f, eN_f = st['eta_all'][-1]
bins = np.linspace(0, CMAP_MAX, 60)
ax2[2].hist(eB_f, bins=bins, alpha=0.6, color='#E24B4A', label='B', density=True)
ax2[2].hist(eN_f, bins=bins, alpha=0.6, color='#378ADD', label='N', density=True)
ax2[2].set_xlabel('Atomic shear strain η')
ax2[2].set_ylabel('Probability density')
ax2[2].set_title(f'(c) η distribution at fracture (ε={st["strain"][-1]:.2f})')
ax2[2].legend(); ax2[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_BN, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved → {OUTPUT_BN}")

# Print stats table
print(f"\n{'ε':>7} {'⟨η⟩_B':>8} {'⟨η⟩_N':>8} {'%hi_B':>7} {'%hi_N':>7}")
for k in range(len(st['strain'])):
    print(f"{st['strain'][k]:>7.3f} {st['mean_B'][k]:>8.3f} {st['mean_N'][k]:>8.3f} "
          f"{st['frac_hi_B'][k]:>7.1f} {st['frac_hi_N'][k]:>7.1f}")
