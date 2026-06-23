import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import json

from parser import Network, load_abundances
from DRG import DRG
from solver import QuadraticSolver

HERE = Path(__file__).resolve().parent
SAVE_DIR = HERE / "reduced_networks" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT/ "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
ABUNDANCES_PATH = REPO_ROOT/ "networks" / "kida.uva.2024" / "abundances.in"
CLUSTER_PATH = REPO_ROOT/ "cluster_data" / "cluster_0000.npy"
SAVE_DIR_FULL = HERE/"Full Order Solve"/"kida"
SAVE_DIR_FULL.mkdir(parents=True, exist_ok=True)

data = np.load(CLUSTER_PATH)
print(data.shape)
test_data = data[0,7:]
env_data = data[0,2:7]

"""idx | t | nH | T | Tgrain | Av | uv_flux | species 0 | species 1 | ... | species 577."""

#Load Network
net = Network(grains = True)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()

#Get Environment
def get_env(env_splice):
    env_splice = np.delete(env_splice, 2)
    env_var = [ 'nH', 'T', 'Av','uv_flux']
    env = {}
    for i, var in enumerate(env_var):
        env[var] = env_splice[i]
    return env

env = get_env(env_data)

#Get Reactions
reactions = net._select_multirange_entries(net.reactions, env["T"])  # dedupe multi-temperature-range entries (matches get_operators)
reaction_rates = net.reaction_rates(reactions, env) #function to get list of reaction rates
species_map = net.species_map
species = net.species

#Remove Dropped from Cluster Data
def rm_dropped(test_data, dropped):
    dropped_idx = [species_map[i] for i in dropped]
    for i in dropped_idx:
        test_data = np.delete(test_data, i)
    return test_data

sources = ['CO', 'C+', 'O+', 'O', 'e-']
eps = [0.00001]

test_data = rm_dropped(test_data, dropped)

#Full order solve for comparison
YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3
A, B = net.get_operators(env)

#Abundances
abund = load_abundances(str(ABUNDANCES_PATH))
abund["e-"] = sum(val for name,val in abund.items() if name.endswith("+"))
x0 = np.zeros(len(species), dtype=np.float64)
for name, val in abund.items():
    if name in species_map:
        x0[species_map[name]] = val

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

#Reduce Net
drg = DRG()

for e in eps:
    drg.reduce_net(reactions, species_map, reaction_rates, test_data.T, sources,dropped,eps = e)
    reduced_reactions = drg.reduced_rxns
    species = drg.reduced_species
    n_species = len(species)

    print(f'\nTolerance: {e}')
    print(f'Number of reactions in reduced network: {len(reduced_reactions)}')
    print(f'Number of species in reduced network: {n_species}')

    file_name = f"cluster_0000_reduced_net_eps{e}.json"
    file_path = SAVE_DIR/file_name

    data = {"epsilon": e, "reactions": reduced_reactions,
            "species": species, "rates": drg.reduced_rates}
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent = 2)

