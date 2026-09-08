"""
Usage: python3 plot_shear_all.py 0.80
Run from inside each density folder. Takes density as command-line arg.
"""
import os, re, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rcParams['axes.linewidth'] = 0.8

RHO = sys.argv[1] if len(sys.argv) > 1 else 'unknown'
OUTPDF = f'{RHO}_shear_strain.pdf'
OUTPNG = f'{RHO}_shear_strain.png'
REF_FILE = '00stress_1.cfg'
CUTOFF = 4.5
CMAP_MAX = 1.5
ETA_MAX = 3.0

def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='same')

# Auto-detect frames
ss = []
with open('Strain_Stress.dat') as f:
    for line in f:
        m = re.findall(r'lz:\s*([\d.]+).*?Pzz:\s*([-\d.e+]+)', line)
        if m: ss.append((float(m[0][0]), float(m[0][1])))
ss = np.array(ss)
L0 = ss[0,0]
strain_all = (ss[:,0] - L0) / L0
stress_all = smooth(ss[:,1])
ntot = len(strain_all)

idx_ult = np.argmax(stress_all)
post = np.arange(idx_ult, ntot)
b50 = post[stress_all[post] < 0.3 * stress_all[idx_ult]]
idx_frac = b50[0] if len(b50) else ntot - 1

FRAMES = sorted(set([
    min(max(5, idx_ult//10), ntot-1),
    min(max(10, idx_ult//3), ntot-1),
    min(max(15, 2*idx_ult//3), ntot-1),
    min(idx_ult, ntot-1),
    min((idx_ult+idx_frac)//2, ntot-1),
    min(idx_frac+30, ntot-1)
]))
while len(FRAMES) < 6:
    FRAMES.append(min(FRAMES[-1]+20, ntot-1))
FRAMES = sorted(set(FRAMES))[:6]

tag = ['', '\nYield', '', '\nUltimate', '', '\nFracture']
LABELS = [f'({["I","II","III","IV","V","VI"][k]}) {strain_all[f]:.3f}{tag[k]}'
          for k, f in enumerate(FRAMES)]

print(f"ρ={RHO}, frames={FRAMES}")
print(f"strains={[round(strain_all[f],3) for f in FRAMES]}")

def read_dump(fname):
    with open(fname) as f: lines = f.readlines()
    i, na, box = 0, 0, np.zeros((3,2))
    pos, tp = None, None
    while i < len(lines):
        if lines[i].startswith('ITEM: NUMBER'):
            na = int(lines[i+1]); i += 2
        elif lines[i].startswith('ITEM: BOX'):
            for d in range(3):
                p = lines[i+1+d].split(); box[d]=[float(p[0]),float(p[1])]
            i += 4
        elif lines[i].startswith('ITEM: ATOMS'):
            c = lines[i].split()[2:]
            ii,it,ix,iy,iz = c.index('id'),c.index('type'),c.index('x'),c.index('y'),c.index('z')
            data = np.array([lines[i+1+k].split() for k in range(na)], dtype=float)
            data = data[np.argsort(data[:,ii])]
            pos = data[:,[ix,iy,iz]]; tp = data[:,it].astype(int)
            i += 1+na
        else: i += 1
    L = box[:,1]-box[:,0]
    return pos-box[:,0], tp, L

def pbc_disp(d,L): return d - L*np.round(d/L)

def calc_eta(pr, Lr, pc, Lc, cut):
    N = len(pr)
    pw = np.clip(np.where(np.mod(pr,Lr)>=Lr, np.mod(pr,Lr)-Lr, np.mod(pr,Lr)), 0, np.nextafter(Lr,0))
    tree = cKDTree(pw, boxsize=Lr)
    pairs = tree.query_pairs(cut, output_type='ndarray')
    nb = [[] for _ in range(N)]
    for a,b in pairs: nb[a].append(b); nb[b].append(a)
    eta = np.full(N, np.nan); I3 = np.eye(3)
    for i in range(N):
        if len(nb[i]) < 4: continue
        dr = pbc_disp(pr[nb[i]]-pr[i], Lr)
        dc = pbc_disp(pc[nb[i]]-pc[i], Lc)
        V = dr.T@dr
        if np.linalg.cond(V)>1e8: continue
        J = np.linalg.solve(V, dr.T@dc)
        e = 0.5*(J@J.T - I3)
        v = np.sqrt(e[0,1]**2+e[0,2]**2+e[1,2]**2+
                    ((e[0,0]-e[1,1])**2+(e[0,0]-e[2,2])**2+(e[1,1]-e[2,2])**2)/6)
        if v < ETA_MAX: eta[i] = v
    return eta

print("Reading reference...")
pos_ref, tp, L_ref = read_dump(REF_FILE)
print(f"  {len(pos_ref)} atoms")

data = []
vlabels = []
for k,frame in enumerate(FRAMES):
    fn = f'00stress_{frame}.cfg'
    if not os.path.exists(fn): print(f"  SKIP {fn}"); continue
    pos,_,L = read_dump(fn)
    if len(pos)!=len(pos_ref): print(f"  SKIP {fn} (mismatch)"); continue
    eps = strain_all[frame]
    print(f"  {fn} ε={eps:.3f}...")
    eta = calc_eta(pos_ref, L_ref, pos, L, CUTOFF)
    print(f"    valid={np.isfinite(eta).sum()}/{len(eta)}")
    data.append((pos, eta, eps))
    vlabels.append(LABELS[k])

nf = len(data)
fig, axes = plt.subplots(1, nf, figsize=(20, 12))
if nf==1: axes=[axes]
plt.subplots_adjust(wspace=0.05, bottom=0.12, top=0.90, left=0.01, right=0.99)
cmap = matplotlib.colormaps['jet']
norm = Normalize(0, CMAP_MAX)

for ax,(pos,eta,eps),lbl in zip(axes, data, vlabels):
    ep = np.where(np.isfinite(eta), eta, CMAP_MAX)
    o = np.argsort(pos[:,1])
    ax.scatter(pos[o,0], pos[o,2], c=ep[o], cmap=cmap, norm=norm,
               s=30, edgecolors='none', rasterized=True)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.8)
    ax.set_title(lbl, fontsize=22, fontweight='bold', pad=10)

cax = fig.add_axes([0.15, 0.035, 0.7, 0.015])
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), cax=cax, orientation='horizontal')
cb.set_label('Atomic von Mises shear strain $\\eta$', fontsize=30)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
cb.ax.tick_params(labelsize=30)

plt.savefig(OUTPDF, format='pdf', bbox_inches='tight', facecolor='white')
plt.savefig(OUTPNG, format='png', dpi=600, bbox_inches='tight', facecolor='white')
print(f"Saved → {OUTPDF} + {OUTPNG}")
