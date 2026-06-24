import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAVE_DIR = HERE / "Plots" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)

#Plotting Quantity of Interests
YEAR = 3600 * 24 * 365.25
folder_path = HERE/"cluster_reduced_solved"
eps = [0.001, 0.01, 0.05, 0.1]

label = [f'eps = {e}' for e in eps]

species = ["CO", "C+", "O+", "O", "e-"]

for s in species:
    plt.figure()

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