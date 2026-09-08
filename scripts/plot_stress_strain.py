"""
plot_ss_separate_clear.py — 6 separate stress-strain with clean labels
Usage:  cd ~/home_computer/conference-paper/tensile && python3 plot_ss_separate_clear.py
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
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 2.0,
    "lines.linewidth": 2.5,
})

def apply_style(ax):
    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2)
    ax.tick_params(axis='both', which='minor', direction='in', length=5, width=1.5)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='same')

def parse_dat(fpath):
    data = []
    with open(fpath) as f:
        for line in f:
            m = re.findall(r'lz:\s*([\d.]+).*?Pzz:\s*([-\d.e+]+)', line)
            if m:
                data.append((float(m[0][0]), float(m[0][1])))
    data = np.array(data)
    L0 = data[0, 0]
    return (data[:, 0] - L0) / L0, data[:, 1]

def auto_detect(strain, stress_s):
    ntot = len(strain)
    idx_ult = np.argmax(stress_s)
    post = np.arange(idx_ult, ntot)
    b50 = post[stress_s[post] < 0.5 * stress_s[idx_ult]]
    idx_frac = b50[0] if len(b50) else ntot - 1
    frames = sorted(set([
        min(max(5, idx_ult // 10), ntot - 1),
        min(max(10, idx_ult // 3), ntot - 1),
        min(max(15, 2 * idx_ult // 3), ntot - 1),
        min(idx_ult, ntot - 1),
        min((idx_ult + idx_frac) // 2, ntot - 1),
        min(idx_frac + 30, ntot - 1)
    ]))
    while len(frames) < 6:
        frames.append(min(frames[-1] + 20, ntot - 1))
    return sorted(set(frames))[:6]

densities = {1: 0.80, 2: 1.00, 3: 1.30, 4: 1.60, 5: 2.00, 6: 2.30}
panels    = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
roman     = ['I', 'II', 'III', 'IV', 'V', 'VI']
colors    = ['#378ADD', '#EF9F27', '#1D9E75', '#E24B4A', '#D85A30', '#791F1F']

for (i, rho), panel in zip(densities.items(), panels):

    strain, stress = parse_dat(f'Strain_Stress_{i}.dat')
    stress_s = smooth(stress)
    frames = auto_detect(strain, stress_s)

    xmax_data = strain[frames[-1]] * 1.08
    xmax = min(max(xmax_data, 0.5), strain[-1])
    mask = strain <= xmax
    ymax = stress_s[mask].max()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(strain[mask], stress[mask], '-', color='#1f77b4', lw=1.0, alpha=0.35)
    ax.plot(strain[mask], stress_s[mask], '-', color='#1f77b4', lw=2.5,
            label=f'density = {rho:.2f}')

    for k, f in enumerate(frames):
        if f >= len(strain) or strain[f] > xmax:
            continue
        sx, sy = strain[f], stress_s[f]
        ax.scatter([sx], [sy], color=colors[k], s=120, zorder=5,
                   edgecolor='k', linewidth=2.5)
        # Put label below if dot is near the top, else above
        if sy > 0.85 * ymax:
            ax.text(sx, sy - 0.08 * ymax, roman[k],
                    fontsize=18, fontweight='bold', color=colors[k],
                    ha='center', va='top')
        else:
            ax.text(sx, sy + 0.06 * ymax, roman[k],
                    fontsize=20, fontweight='bold', color=colors[k],
                    ha='center', va='bottom')

    ax.set_xlabel('Strain ($\\varepsilon$) [%]',fontsize=20)
    ax.set_ylabel('Stress ($\\sigma$) [GPa]',fontsize=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)
    apply_style(ax)
    plt.tight_layout()

    rho_str = f'{rho:.2f}'.replace('.', '')
    plt.savefig(f'ss_{rho_str}.pdf', format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"rho={rho}: frames={frames}, "
          f"strains=[{', '.join(f'{strain[f]:.3f}' for f in frames)}] -> ss_{rho_str}.pdf")

print("\nDone! All 6 saved.")
