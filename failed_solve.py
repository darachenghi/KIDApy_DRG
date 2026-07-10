
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

from parser import Network
from DRG_Par import DRG_p
from concurrent.futures import ProcessPoolExecutor, as_completed
from assembly import assembly
from solver import QuadraticSolver

HERE = Path(__file__).resolve().parent
NETWORK_PATH    = HERE/ "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
RED_NET_DIR = HERE/"reduced_networks_0025"
CLUSTER_DIR = HERE.parent/"clusters_params_only"
SAVE_DIR = HERE/"reduced_solutions"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

#LOADS NETWORK
net = Network(grains = True)
net.load_from_disk(str(NETWORK_PATH))
cluster_species_map = net.species_map #indexing of species in original cluster results
dropped = net.drop_passive_species() 
net._get_stoich(dropped) #adds stoichiometric coefficient dictionary to each reaction

#Define Source Species and Tolerance
eps = 0.001
sources = ['CO', 'C+', 'O+', 'O', 'e-']

cluster_num = "0025"
failed_int = 0

CLUSTER_PATH = CLUSTER_DIR/f"cluster_{cluster_num}.npy"
cluster = np.load(CLUSTER_PATH)
cluster_interval = cluster[failed_int*20:failed_int*20 +20, :]

###########################################SOLVES REDUCED NETWORK##########################################

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3

RED_NET_PATH = RED_NET_DIR/f"reduced_net_eps{eps}.json"
with open(RED_NET_PATH) as f:
    data = json.load(f)

species = data["species"] 
init_reactions = data["reactions"]
species_map = {s: i for i, s in enumerate(species)}

init_state = cluster_interval[0, :] 

env_array = np.delete(init_state[2:7], 2)
env_var = ['nH', 'T', 'Av', 'uv_flux']
env = {var: env_array[i] for i, var in enumerate(env_var)} #environment for point solve


reactions = net._select_multirange_entries(init_reactions, env["T"])
rates = net.reaction_rates(reactions, env)

x0 = np.zeros(len(species), dtype=np.float64)

for s, sidx in cluster_species_map.items(): #first row of cluster chunk gives initial conditions 
    if s in species_map:
        idx7 = sidx + 7
        x0[species_map[s]] = init_state[idx7]

asb = assembly(grains=True)
A, B = asb.get_operators(reactions, species_map, rates)

t_eval = cluster_interval[:, 1]

solver = QuadraticSolver()
t, y = solver.solve(A, B,
                    x0=x0,
                    atol=ATOL,
                    rtol=RTOL,
                    method="BDF",
                    t_eval=t_eval,
                    t_span=(t_eval[0], t_eval[-1]))

out_root = str(SAVE_DIR / f"reduced_sol_eps{str(eps).replace('.', 'p')}")
solver = QuadraticSolver()
solver.save(out_root, t, y, col_names=species_map.keys()) #save full solution for all species in reduced network
print(f"eps={eps}: saved {out_root}.csv")

###########################################PLOTS REDUCED NETWORK##########################################

for s in sources:

    true_sol = cluster_interval[:, cluster_species_map[s] + 7]
    red_sol = y[species_map[s], :]

    plt.figure(figsize=(5.5, 4.5))
    plt.plot(t_eval / YEAR, true_sol, label="true")
    plt.plot(t / YEAR, red_sol, linestyle="dashed", label="reduced")

    plt.xlabel("Time (Years)")
    plt.ylabel("Abundance per H")
    plt.title(s)
    plt.legend()

    out_file = SAVE_DIR/ f"{s}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()