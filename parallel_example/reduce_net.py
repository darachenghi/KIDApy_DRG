import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import time

from parser import Network
from DRG_Par import DRG_p

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT/ "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
CLUSTER_DIR = Path("/work/10864/arjunveejay/mysharedirectory/clusters_params_only")

#LOAD NETWORK
net = Network(grains = True)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()
net._get_stoich(dropped) #adds stoichiometric coefficient dictionary to each reaction

#Define Source Species and Tolerance
clust = [f"{num:04}" for num in range(40,50)]
sources = ['CO', 'C+', 'O+', 'O', 'e-']
eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

for c in clust:
    CLUSTER_PATH = CLUSTER_DIR/f"cluster_{c}.npy"
    SAVE_DIR =  REPO_ROOT/ "reduced_networks"/c
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cluster = np.load(CLUSTER_PATH)
    print(f"\nCluster: {c}")

    drg = DRG_p()
    start = time.perf_counter()
    drg.reduce_net(net, cluster, sources, eps, dropped, savedir=SAVE_DIR) 
    end = time.perf_counter()
    print(f"Time {end -start} seconds")

