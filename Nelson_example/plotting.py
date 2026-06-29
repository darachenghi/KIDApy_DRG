import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAVE_DIR = HERE / "Plots" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
FULL_ORDER_SOLVE = HERE/"Full_networks"/"nelson.csv"


#Plotting Epsilon Plot
folder = HERE/"reduced_networks"
eps = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
species_length = []

plt.figure(figsize = (6,5))

for e in eps:
    file_name = f"nelson_reduced_net_eps{e}.json"
    file = folder/file_name
    with open(file) as f:
        data = json.load(f)

    species = data["species"]
    n_species = len(species)
    species_length.append(n_species)

plt.plot(eps, species_length, marker = 'o')

plt.xlabel("Epsilon")
plt.ylabel("Number of Species")
file_name = "kida_eps"
plt.savefig(SAVE_DIR/file_name, dpi = 700)

#Plotting Quantity of Interests
YEAR = 3600 * 24 * 365.25
folder_path = HERE/"nelson_reduced_solved"
eps = [ 0.01, 0.2, 0.7]
labels = ["No Reduction (14 species)", "eps = 0.01 (12 species)", "eps = 0.2 (11 species)", "eps = 0.7 (9 species)"]

species = ["CO", "e-"]

for s in species:
    plt.figure(figsize = (5.5,4.5))
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
    plt.legend(labels)
    plt.title(f'{s}')
    file = SAVE_DIR/f'{s}'
    plt.savefig(file, dpi = 700)