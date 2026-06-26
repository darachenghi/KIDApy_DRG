import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAVE_DIR = HERE / "Plots" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
FULL_ORDER_SOLVE = HERE/"Full_networks"/"nelson.csv"

YEAR = 3600 * 24 * 365.25
folder_path = HERE/"kida_reduced_solved"
eps =  [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,0.06,0.075, 0.1,0.15, 0.2,0.3, 0.4, 0.5, 0.6, 0.7,0.8 ,0.9]

species = ["CO", "C+", "O+", "O", "e-"]

for s in species:
    df = pd.read_csv(FULL_ORDER_SOLVE)
    true_conc = df[s]
    true_conc_l2 = np.linalg.norm(true_conc, 2)
    max_errors = []
    rel_errors = []

    for e in eps:
        e = str(e).replace('.', 'p')
        f = f'{folder_path}/{e}eps.csv'
        df = pd.read_csv(f)
        eps_conc = df[s]
        diff = true_conc - eps_conc
        diff_l2 = np.linalg.norm(diff,2)
        rel_errors.append(diff_l2/true_conc_l2)
        max_errors.append(np.linalg.norm(diff, np.inf))


    plt.figure()
    plt.plot(eps, max_errors, marker = 'o')
    plt.xlabel("Epsilon")
    plt.ylabel('Max Error')
    plt.title(f'{s}')
    file = SAVE_DIR/f'max_error_{s}'
    plt.savefig(file, dpi = 700)