
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
from DRG_subset import DRG_sub

#Path and settings
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/networks/kida.uva.2024/gas_reactions_kida.uva.2024.in"
ABUNDANCES_PATH = "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/networks/kida.uva.2024/abundances.in"

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3

#Load network
net = Network(grains = True)
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
    T       = 10.0,    # gas temperature [K]
    nH      = 1e4,     # total H number density [cm⁻³]
    Av      = 10.0,    # visual extinction [mag]
    uv_flux = 1.0,     # FUV field scaling (1 = standard Draine field)
)

A, B = net.get_operators(env)

#solve 
t_eval = np.logspace(0, np.log10(1e6 * YEAR), 300)
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
sources = ['CO', 'C+', 'O', 'O+']
source_indices = [net.species_map[i] for i in sources]
eps = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
species_eps = []

print(f'Source Species: {sources}')

#Reduce Network
drg = DRG_sub()

for e in eps:
    drg.reduce_net(net.reactions, net.species_map,reaction_rates, y, source_indices,dropped,eps = e)
    reactions = drg.reduced_rxns
    species = drg.reduced_species
    n_species = len(drg.skeletal_species)
    species_eps.append(species)

    print(f'Tolerance: {e}')
    print(f'Number of reactions in reduced network: {len(drg.reduced_rxns)}')
    print(f'Number of species in reduced network: {n_species}')

    reduced_folder = "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/reduced_net"
    file_name = f"kida_reduced_net_eps{e}_sub.json"
    file_path = f"{reduced_folder}/{file_name}"
    data = {"method": "subset", "epsilon": e, "reactions": reactions, 
            "species": drg.skeletal_species, "rates": drg.skeletal_rates}
    with open(file_path, "w") as f:
        json.dump(data, f, indent = 2)

data_eps = {"epsilon": eps, "species": species_eps}
eps_folder = "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/epsilon"
file_name_eps = "kida_eps_results_sub.json"
file_path_eps = f'{eps_folder}/{file_name_eps}'

with open(file_path_eps, "w") as f_eps:
    json.dump(data_eps, f_eps, indent = 2)
