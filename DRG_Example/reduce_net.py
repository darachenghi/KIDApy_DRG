
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
import time

from parser import Network, load_abundances
from solver import QuadraticSolver
from DRG import DRG

# Paths and settings

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT/ "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
ABUNDANCES_PATH = REPO_ROOT/ "networks" / "kida.uva.2024" / "abundances.in"
SAVE_DIR = HERE / "reduced_networks" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR_FULL = HERE / "Full_networks" 
SAVE_DIR_FULL.mkdir(parents=True, exist_ok=True)

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3

# Load network
net = Network(grains=True)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()

# Initial conditions

abund = load_abundances(str(ABUNDANCES_PATH))
abund["e-"] = sum(val for name, val in abund.items() if name.endswith("+"))  # charge neutrality
x0 = np.zeros(len(net.species), dtype=np.float64)
for name, val in abund.items():
    if name in net.species_map:
        x0[net.species_map[name]] = val

print("\nNetwork")
print(f"  species  = {len(net.species)}")
print(f"  reactions = {len(net.reactions)}")


# Environment

env = dict(
    T       = 10.0,    # gas temperature [K]
    nH      = 1e4,     # total H number density [cm⁻³]
    Av      = 10.0,    # visual extinction [mag]
    uv_flux = 1.0,     # FUV field scaling (1 = standard Draine field)
)

A, B = net.get_operators(env)

# Integrate

t_eval = np.logspace(0, np.log10(1e6 * YEAR), 300)

solver = QuadraticSolver()
t, y = solver.solve(
    A, B,
    t_span=(t_eval[0], t_eval[-1]),
    x0=x0,
    atol=ATOL,
    rtol=RTOL,
    t_eval=t_eval,
)

out_root = str(SAVE_DIR_FULL / "kida_uva_2024_point")
solver.save(out_root, t, y, col_names=net.species)

reactions = net._select_multirange_entries(net.reactions, env["T"])  # dedupe multi-temperature-range entries (matches get_operators)
reaction_rates = net.reaction_rates(reactions, env) #function to get list of reaction rates
species_map = net.species_map

#Sources
sources = ['CO', 'C+', 'O+', 'O','e-']
eps = [0.2]

#Reduce Network
drg = DRG()

for e in eps:
    start = time.perf_counter()
    drg.reduce_net(reactions, species_map, reaction_rates, y, sources,dropped,eps = e)
    end = time.perf_counter()
    reduced_reactions = drg.reduced_rxns
    species = drg.reduced_species
    n_species = len(species)

    print(f'\nTolerance: {e}')
    print(f'Number of reactions in reduced network: {len(reduced_reactions)}')
    print(f'Number of species in reduced network: {n_species}')
    print(f'Time to reduce: {end-start} seconds')

    file_name = f"kida_reduced_net_eps{e}.json"
    file_path = SAVE_DIR/file_name

    data = {"epsilon": e, "reactions": reduced_reactions,
            "species": species, "rates": drg.reduced_rates}
    
    #with open(file_path, "w") as f:
        #json.dump(data, f, indent = 2)
