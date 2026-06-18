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
eps = [0.001, 0.005,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
species_length = []

plt.figure()

for e in eps:
    file_name = f"kida_reduced_net_eps{e}.json"
    file = folder/file_name
    with open(file) as f:
        data = json.load(f)

    species = data["species"]
    n_species = len(species)
    species_length.append(n_species)

plt.plot(eps, species_length, marker = 'o')

plt.xlabel("Epsilon")
plt.ylabel("Number of Species")
plt.title("DRG Reduction of KIDA Network")
file_name = "kida_eps"
plt.savefig(SAVE_DIR/file_name)


#Plotting Quantity of Interests
YEAR = 3600 * 24 * 365.25
folder_path = HERE/"kida_reduced_solved"
eps = ['0p001','0p01', '0p1','0p2']

species = ["CO", "C+", "O+", "O"]

for s in species:
    plt.figure()
    df = pd.read_csv(FULL_ORDER_SOLVE)
    plt.plot(df["t"]/YEAR, df[s])

    for e in eps:
        f = f'{folder_path}/{e}eps.csv'
        df = pd.read_csv(f)
        plt.plot(df["t"]/YEAR, df[s], linestyle = 'dashed')

    plt.loglog()
    plt.xlabel("Time (Years)")
    plt.ylabel('Abundance per H')
    plt.legend(["full order", "eps = 0.001", "eps = 0.01", "eps = 0.1", "eps = 0.2"])
    plt.title(f'{s}')
    file = SAVE_DIR/f'{s}'
    plt.savefig(file)