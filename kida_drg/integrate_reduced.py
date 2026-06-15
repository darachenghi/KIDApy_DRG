
import sys
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import load_abundances
from assembly import assembly
from solver import QuadraticSolver


YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3


network_path = Path("/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/reduced_net/kida_reduced_net_eps0.01_max.json")
with open(network_path) as f:
    data = json.load(f)

reactions = data["reactions"]
species_map = data["species"]
rates = data["rates"]
x0 = np.array(data["initial abundances"]) 

asb = assembly()
A,B = asb.get_operators(reactions, species_map, rates)

t_eval = np.logspace(0, np.log10(1e6 * YEAR), 1000)

solver = QuadraticSolver()
t, y = solver.solve(
    A, B,
    t_span=(t_eval[0], t_eval[-1]),
    x0=x0,
    atol=ATOL,
    rtol=RTOL,
    t_eval=t_eval,
)

out_root =Path("/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/kida_reduced_solved/kida_reduced_eps0.01_max_new2")
solver.save(out_root, t, y, col_names= species_map.keys())
print(f"  saved = {out_root}.csv")