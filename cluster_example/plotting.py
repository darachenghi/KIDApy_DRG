import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAVE_DIR = HERE / "Plots" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)

import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAVE_DIR = HERE / "Plots" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
FULL_ORDER_SOLVE = HERE/"Full_networks"/"kida_uva_2024_point.csv"


#Plotting Epsilon Plot
folder = HERE/"reduced_networks"
eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.9]
species_length = []

plt.figure(figsize = (6,5))

for e in eps:
    file_name = f"cluster_0000_reduced_net_eps{e}.json"
    file = folder/file_name
    with open(file) as f:
        data = json.load(f)

    species = data["species"]
    n_species = len(species)
    species_length.append(n_species)

plt.plot(eps, species_length, marker = 'o')

plt.xlabel("Epsilon")
plt.ylabel("Number of Species")
plt.title("DRG Reduction (Cluster 0000)")
file_name = "cluster_0000_eps"
plt.savefig(SAVE_DIR/file_name, dpi = 700, transparent = True)

'''
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
'''