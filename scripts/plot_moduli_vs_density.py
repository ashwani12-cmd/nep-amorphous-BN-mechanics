"""
Plot mechanical properties vs density for a-BN paper
Generates separate PDFs: E_vs_density, strength_vs_density, ductility_vs_density, toughness_vs_density
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 20,
    "axes.linewidth": 2.0,
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
})

def apply_style(ax):
    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2)
    ax.tick_params(axis='both', which='minor', direction='in', length=5, width=1.5)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='same')

def parse_and_analyze(fpath, rho):
    data = []
    with open(fpath) as f:
        for line in f:
            m = re.findall(r'lz:\s*([\d.]+).*?Pzz:\s*([-\d.e+]+)', line)
            if m:
                data.append((float(m[0][0]), float(m[0][1])))
    data = np.array(data)
    L0 = data[0, 0]
    strain = (data[:, 0] - L0) / L0
    stress = data[:, 1]
    stress_s = smooth(stress)

    # Young's modulus — adaptive linear fit
    idx_ult = np.argmax(stress_s)
    sig30 = 0.3 * stress_s[idx_ult]
    lin_end = strain[np.argmax(stress_s > sig30)]
    lin = (strain > 0.005) & (strain < min(lin_end, 0.15))
    if lin.sum() < 5:
        lin = (strain > 0.005) & (strain < 0.05)
    E, b = np.polyfit(strain[lin], stress[lin], 1)

    # Yield: 0.2% offset, must be before ultimate
    offset = E * (strain - 0.002) + b
    diff = stress_s - offset
    sc = np.where(np.diff(np.sign(diff)) != 0)[0]
    valid = sc[(strain[sc] > 0.02) & (sc < idx_ult) &
               (stress_s[sc] > 0.2 * stress_s[idx_ult])]
    if len(valid) > 0:
        idx_yield = valid[0]
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = np.abs(stress_s - (E*strain+b)) / np.abs(E*strain+b+1e-9)
        cand = np.where((strain > 0.02) & (dev > 0.05) &
                        (np.arange(len(strain)) < idx_ult) &
                        (stress_s > 0.2*stress_s[idx_ult]))[0]
        idx_yield = cand[0] if len(cand) else idx_ult // 2

    # Toughness: area under stress-strain curve up to fracture
    post = np.arange(idx_ult, len(stress_s))
    b50 = post[stress_s[post] < 0.5 * stress_s[idx_ult]]
    idx_frac = b50[0] if len(b50) else len(stress_s) - 1
    toughness = np.trapz(stress_s[:idx_frac], strain[:idx_frac])

    return {
        'rho': rho,
        'E': E,
        'yield_stress': stress[idx_yield],
        'yield_strain': strain[idx_yield],
        'ult_stress': stress[idx_ult],
        'ult_strain': strain[idx_ult],
        'frac_strain': strain[idx_frac],
        'toughness': toughness,
    }

# ============================================================
# Analyze all 6 densities
# ============================================================
densities = {1: 0.80, 2: 1.00, 3: 1.30, 4: 1.60, 5: 2.00, 6: 2.30}
results = []
for i, rho in densities.items():
    r = parse_and_analyze(f'Strain_Stress_{i}.dat', rho)
    results.append(r)
    print(f"rho={rho}: E={r['E']:.1f}, sig_y={r['yield_stress']:.2f}, "
          f"sig_ult={r['ult_stress']:.2f}, eps_ult={r['ult_strain']:.3f}, "
          f"T={r['toughness']:.2f}")

rhos = [r['rho'] for r in results]
Es = [r['E'] for r in results]
sig_y = [r['yield_stress'] for r in results]
sig_ult = [r['ult_stress'] for r in results]
eps_ult = [r['ult_strain'] for r in results]
eps_frac = [r['frac_strain'] for r in results]
tough = [r['toughness'] for r in results]

# ============================================================
# Figure 1: Young's modulus vs density
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(rhos, Es, 'o-', color='#1f77b4', markersize=20, markeredgecolor='k',
        markeredgewidth=2.0)
ax.set_xlabel('Density, ($\\rho$) [g/cm$^3$]',fontsize=20)
ax.set_ylabel("Young's modulus, ($E$) [GPa]",fontsize=20)
ax.set_xlim(0.6, 2.5)
ax.set_ylim(bottom=0)
apply_style(ax)
plt.tight_layout()
plt.savefig('E_vs_density.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
plt.savefig('E_vs_density.png', format='png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved E_vs_density")

# ============================================================
# Figure 2: Strength vs density (yield + ultimate)
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(rhos, sig_ult, 's-', color='#E24B4A', markersize=20, markeredgecolor='k',
        markeredgewidth=2.0, label='Ultimate strength, $\\sigma_{ult}$')
ax.plot(rhos, sig_y, '^-', color='#EF9F27', markersize=20, markeredgecolor='k',
        markeredgewidth=2.0, label='Yield strength, $\\sigma_{Y}$')
ax.set_xlabel('Density, ($\\rho$) [g/cm$^3$]',fontsize=20)
ax.set_ylabel('Stress, ($\\sigma$) [GPa]',fontsize=20)
ax.set_xlim(0.6, 2.5)
ax.set_ylim(bottom=0)
ax.legend()
apply_style(ax)
plt.tight_layout()
plt.savefig('strength_vs_density.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
plt.savefig('strength_vs_density.png', format='png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved strength_vs_density")

# ============================================================
# Figure 3: Ductility vs density (strain at ultimate + fracture)
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(rhos, eps_ult, 'd-', color='#1D9E75', markersize=20, markeredgecolor='k',
        markeredgewidth=2.0, label='Strain at ultimate, $\\varepsilon_{ult}$')
ax.plot(rhos, eps_frac, 'v-', color='#534AB7', markersize=20, markeredgecolor='k',
        markeredgewidth=2.0, label='Strain at fracture, $\\varepsilon_{f}$')
ax.set_xlabel('Density, ($\\rho$ [g/cm$^3$])',fontsize=20)
ax.set_ylabel('Strain, ($\\varepsilon$)',fontsize=20)
ax.set_xlim(0.6, 2.5)
ax.set_ylim(bottom=0)
ax.legend()
apply_style(ax)
plt.tight_layout()
plt.savefig('ductility_vs_density.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
plt.savefig('ductility_vs_density.png', format='png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved ductility_vs_density")

# ============================================================
# Figure 4: Toughness vs density
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(rhos, tough, 'D-', color='#D85A30', markersize=20, markeredgecolor='k',
        markeredgewidth=2.0)
ax.set_xlabel('Density, $\\rho$ [g/cm$^3$]',fontsize=20)
ax.set_ylabel('Toughness [GJ/m$^3$]',fontsize=20)
ax.set_xlim(0.6, 2.5)
ax.set_ylim(bottom=0)
apply_style(ax)
plt.tight_layout()
plt.savefig('toughness_vs_density.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
plt.savefig('toughness_vs_density.png', format='png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved toughness_vs_density")

# ============================================================
# Print summary table for paper
# ============================================================
print("\n" + "="*80)
print("SUMMARY TABLE FOR PAPER")
print("="*80)
print(f"{'rho':>6} {'E':>8} {'sig_y':>8} {'sig_ult':>8} {'eps_ult':>8} {'eps_f':>8} {'T':>8}")
print(f"{'g/cm3':>6} {'GPa':>8} {'GPa':>8} {'GPa':>8} {'':>8} {'':>8} {'GJ/m3':>8}")
print("-"*80)
for r in results:
    print(f"{r['rho']:>6.2f} {r['E']:>8.1f} {r['yield_stress']:>8.2f} "
          f"{r['ult_stress']:>8.2f} {r['ult_strain']:>8.3f} "
          f"{r['frac_strain']:>8.3f} {r['toughness']:>8.2f}")

print("\n" + "="*80)
print("LATEX TABLE")
print("="*80)
print(r"\begin{table}[h!]")
print(r"\centering")
print(r"\caption{Mechanical properties of a-BN at varying densities.}")
print(r"\label{tab:mech_props}")
print(r"\begin{tabular}{ccccccc}")
print(r"\hline")
print(r"$\rho$ & $E$ & $\sigma_y$ & $\sigma_{ult}$ & $\varepsilon_{ult}$ & $\varepsilon_f$ & Toughness \\")
print(r"(g/cm$^3$) & (GPa) & (GPa) & (GPa) & & & (GJ/m$^3$) \\")
print(r"\hline")
for r in results:
    print(f"{r['rho']:.2f} & {r['E']:.1f} & {r['yield_stress']:.2f} & "
          f"{r['ult_stress']:.2f} & {r['ult_strain']:.3f} & "
          f"{r['frac_strain']:.3f} & {r['toughness']:.2f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

print("\nAll done!")
