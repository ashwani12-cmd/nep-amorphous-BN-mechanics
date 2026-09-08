#!/bin/bash
for d in 1 2 3 4 5 6; do
    cd ~/home_computer/conference-paper/5X/$d
    rho=$(ls aBN_density_*.data | sed 's/aBN_density_//;s/.data//')
    echo "=== Folder $d (ρ=${rho}) ==="
    python3 ~/home_computer/conference-paper/5X/plot_shear_all.py ${rho}
done
