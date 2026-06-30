import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAVE_DIR = HERE / "Plots" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)


#Plotting Epsilon Plot
folder = HERE/"reduced_networks"
eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
species_length = []

plt.figure()

for e in eps:
    file_name = f"reduced_net_eps{e}.json"
    file = folder/file_name
    with open(file) as f:
        data = json.load(f)

    species = data["species"]
    n_species = len(species)
    species_length.append(n_species)

plt.plot(eps, species_length, marker = 'o')

plt.xlabel("Epsilon")
plt.ylabel("Number of Species")
plt.title("DRG Reduction of KIDA Network (Cluster 0000)")
file_name = "cluster0000_eps"
plt.savefig(SAVE_DIR/file_name)