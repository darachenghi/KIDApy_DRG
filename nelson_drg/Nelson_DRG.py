
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json

from parser import Network, load_abundances
from solver import QuadraticSolver
from DRG_union import DRG_u

#Path and settings
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT / "networks" / "nelson" / "gas_reactions.in"
ABUNDANCES_PATH = REPO_ROOT / "networks" / "nelson" / "abundances.in"

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3

#Load network
net = Network(grains = False)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()

#Inital Conditions
abund = load_abundances(str(ABUNDANCES_PATH))
abund["e-"] = sum(val for name,val in abund.items() if name.endswith("+"))
x0 = np.zeros(len(net.species), dtype=np.float64)
for name, val in abund.items():
    if name in net.species_map:
        x0[net.species_map[name]] = val

#Environment
env = dict(
    T = 10.0,
    nH = 2e4,
    Av = 15.0,
    uv_flux = 1.0,
)

A, B = net.get_operators(env)

#solve 
t_eval = np.logspace(0, np.log10(1e6 * YEAR), 120)
solver = QuadraticSolver()
t,y = solver.solve(
    A, B,
    t_span = (t_eval[0], t_eval[-1]),
    x0 = x0,
    t_eval = t_eval,
    atol = ATOL,
    rtol = RTOL,
)
#Get reaction rates
reaction_rates = net.reaction_rates(env)

#Search Parameters
sources = ['CO', 'e-']
source_indices = [net.species_map[i] for i in sources]
eps = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7, 0.8, 0.9]
species_eps = []

print(f'Source Species: {sources}')

#Reduce Network
drg = DRG_u()

for e in eps:
    drg.reduce_net(net.reactions, net.species_map,reaction_rates, y, source_indices,dropped,eps = e)
    reactions = drg.reduced_rxns
    species = drg.reduced_species
    n_species = len(drg.reduced_species)
    species_eps.append(species)
    print(f'Tolerance: {e}')
    print(f'Number of reactions in reduced network: {len(drg.reduced_rxns)}')
    print(f'Number of species in reduced network: {n_species}')

    reduced_folder = HERE
    file_name = f"nelson_reduced_net_eps{e}.json"
    file_path = reduced_folder/file_name
    data = {"method": "union", "epsilon": e, "reactions": reactions, 
            "species": net.species_map, "rates": reaction_rates, "initial abundances": x0.tolist()}
    with open(file_path, "w") as f:
        json.dump(data, f, indent = 2)

data_eps = {"epsilon": eps, "species": species_eps}
eps_folder = HERE
file_name_eps = "nelson_eps_results_union.json"
file_path_eps = eps_folder/file_name_eps

with open(file_path_eps, "w") as f_eps:
    json.dump(data_eps, f_eps, indent = 2)

#Plotting the Directed Graph
'''
G =  drg.Graph
pos = nx.circular_layout(G)
labels = {net.species_map[i]: i for i in net.species_map.keys()}
plt.figure(figsize=(6, 4))
nx.draw(G, pos,labels = labels, with_labels=True, node_color='lightblue', 
        node_size=700, arrowstyle='-|>', arrowsize=15)
plt.title("Directed Graph")
plt.savefig("Nelson Directed Graph")
'''