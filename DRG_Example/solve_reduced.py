
import sys
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import load_abundances
from assembly import assembly
from solver import QuadraticSolver

HERE = Path(__file__).resolve().parent
SAVE_DIR = HERE / "kida_reduced_solved" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = HERE.parent
ABUNDANCES_PATH = REPO_ROOT/ "networks" / "kida.uva.2024" / "abundances.in"

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3

eps = [ 0.001, 0.01, 0.1, 0.2]
t_eval = np.logspace(0, np.log10(1e6 * YEAR), 300)

for e in eps:
    file_name = f"kida_reduced_net_eps{e}.json"
    file_path = HERE/"reduced_networks"/file_name

    with open(file_path) as f:
        data = json.load(f)

    reactions = data["reactions"]
    species = data["species"]
    species_map = {species:i for i, species in enumerate(species)}
    rates = data["rates"]

    abund = load_abundances(str(ABUNDANCES_PATH))
    abund["e-"] = sum(val for name,val in abund.items() if name.endswith("+"))
    x0 = np.zeros(len(species), dtype=np.float64)
    for name, val in abund.items():
        if name in species_map:
            x0[species_map[name]] = val

    asb = assembly(grains=True)
    A,B = asb.get_operators(reactions, species_map, rates)

    print(f'Epsilon: {e}')
    print(f"  A shape = {A.shape}, nnz = {A.nnz}")
    print(f"  B shape = {B.shape}, nnz = {B.nnz}")

    solver = QuadraticSolver()
    t, y = solver.solve(
        A, B,
        t_span=(t_eval[0], t_eval[-1]),
        x0=x0,
        atol=ATOL,
        rtol=RTOL,
        t_eval=t_eval,
    )

    out_root = str(SAVE_DIR/f"{str(e).replace('.', 'p')}eps")
    solver.save(out_root, t, y, col_names= species_map.keys())
    print(f"  saved = {out_root}.csv")