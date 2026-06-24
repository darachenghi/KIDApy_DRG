

import sys
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import load_abundances, Network
from assembly import assembly
from DRG_Cluster import DRG_c
from solver import QuadraticSolver

HERE = Path(__file__).resolve().parent
SAVE_DIR = HERE / "cluster_reduced_solved" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = HERE.parent
ABUNDANCES_PATH = REPO_ROOT/ "networks" / "kida.uva.2024" / "abundances.in"

CLUSTER_PATH = REPO_ROOT/ "cluster_data" / "cluster_0000.npy"
YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-6

NETWORK_PATH    = REPO_ROOT / "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
FULL_SAVE_DIR = HERE / "Full Order Solve"
FULL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

#Environment Variables
env = dict(
    T       = 12.0,    # gas temperature [K]
    nH      = 1e4,     # total H number density [cm⁻³]
    Av      = 10.0,    # visual extinction [mag]
    uv_flux = 1.0,     # FUV field scaling (1 = standard Draine field)
)

#LOAD NETWORK
net = Network(grains=True)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()

#INITIAL CONDITIONS
abund = load_abundances(str(ABUNDANCES_PATH))
x0 = np.zeros(len(net.species), dtype=np.float64)
for name, val in abund.items():
    if name in net.species_map:
        x0[net.species_map[name]] = val

#GET MATRIX A, B
A, B = net.get_operators(env)

#INTEGRATE
t_eval = np.logspace(0, np.log10(1e6 * YEAR), 300)

solver = QuadraticSolver()
t, y = solver.solve(
    A, B,
    t_span=(t_eval[0], t_eval[-1]),
    x0=x0,
    atol=ATOL,
    rtol=RTOL,
    t_eval=None,
)

#SAVE
out_root = str(FULL_SAVE_DIR / "Cluster_0000")
solver.save(out_root, t, y, col_names=net.species)
print(f"\n  saved = {out_root}.npy")
print(f"  saved = {out_root}.csv")
