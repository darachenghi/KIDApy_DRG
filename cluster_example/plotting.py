import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAVE_DIR = HERE / "Plots" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
FULL_ORDER_SOLVE = "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/cluster_example/Full Order Solve/Full Order Solve/kida_uva_2024_point.csv"

#Plotting Quantity of Interests
YEAR = 3600 * 24 * 365.25
folder_path = HERE/"cluster_reduced_solved"
eps = [0.00001,0.001]

label = ["full order"] + [f'eps = {e}' for e in eps]

species = ["CO", "C+", "O+", "O"]

for s in species:
    plt.figure()
    df = pd.read_csv(FULL_ORDER_SOLVE)
    plt.plot(df["t"]/YEAR, df[s])

    for e in eps:
        e = str(e).replace('.', 'p')
        f = f'{folder_path}/{e}eps.csv'
        df = pd.read_csv(f)
        plt.plot(df["t"]/YEAR, df[s], linestyle = 'dashed')

    plt.loglog()
    plt.xlabel("Time (Years)")
    plt.ylabel('Abundance per H')
    plt.legend(label)
    plt.title(f'{s}')
    file = SAVE_DIR/f'{s}'
    plt.savefig(file)