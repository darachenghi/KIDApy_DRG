import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import json
import time

from parser import Network
from DRG_Cluster import DRG_c

HERE = Path(__file__).resolve().parent
SAVE_DIR = HERE / "reduced_networks" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = HERE.parent

NETWORK_PATH    = REPO_ROOT/ "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
CLUSTER_PATH = REPO_ROOT/ "cluster_data" / "cluster_0000.npy"

#Load Network
net = Network(grains = True)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()

#Load Cluster
cluster = np.load(CLUSTER_PATH)
cluster = cluster[0,:]

#Define Source Species and Tolerance
sources = ['CO', 'C+', 'O+', 'O', 'e-']
eps = [0.001]

#Reduce Network
drg = DRG_c()

for e in eps:
    start = time.perf_counter()
    drg.reduce_net(net,cluster, sources,dropped,eps = e)
    end = time.perf_counter()
    reduced_reactions = drg.reduced_rxns
    species = drg.reduced_species
    n_species = len(species)

    print(f'\nTolerance: {e}')
    print(f'Number of reactions in reduced network: {len(reduced_reactions)}')
    print(f'Number of species in reduced network: {n_species}')
    print(f'Time to reduce: {end - start} seconds')

    file_name = f"test_cluster_0000_reduced_net_eps{e}.json"
    file_path = SAVE_DIR/file_name

    data = {"epsilon": e, "reactions": reduced_reactions,
            "species": species}
    
    #with open(file_path, "w") as f:
        #json.dump(data, f, indent = 2)