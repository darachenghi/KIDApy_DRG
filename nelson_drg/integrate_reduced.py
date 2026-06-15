
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
SAVE_DIR = HERE / "nelson_reduced_solved" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3

eps = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7, 0.8, 0.9]
t_eval = np.logspace(0, np.log10(1e6 * YEAR), 300)

for e in eps:
    file_name = f"nelson_reduced_net_eps{e}.json"
    file_path = HERE/"reduced_net"/file_name

    with open(file_path) as f:
        data = json.load(f)

    reactions = data["reactions"]
    species_map = data["species"]
    rates = data["rates"]
    x0 = np.array(data["initial abundances"]) 

    asb = assembly()
    A,B = asb.get_operators(reactions, species_map, rates)

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