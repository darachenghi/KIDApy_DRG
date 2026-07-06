import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import time

from parser import Network
from DRG_Par import DRG_p

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAVE_DIR =  REPO_ROOT.parent/ "reduced_networks" 
SAVE_DIR.mkdir(parents=True, exist_ok=True)

NETWORK_PATH    = REPO_ROOT/ "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
CLUSTER_PATH = REPO_ROOT.parent/ "cluster_0000.npy"

#LOAD NETWORK
net = Network(grains = True)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()
net._get_stoich(dropped) #adds stoichiometric coefficient dictionary to each reaction

#Load Cluster
cluster = np.load(CLUSTER_PATH)
cluster = cluster[:100, :]

#Define Source Species and Tolerance
sources = ['CO', 'C+', 'O+', 'O', 'e-']
eps = [0.001]

#Reduce Network (reduced networks saved to SAVE_DIR under name "reduced_net_eps{eps}.json")
#if running cluster in batches set prevfolder to directory where previous batch of reduced networks are saved
drg = DRG_p()
start = time.perf_counter()
drg.reduce_net(net, cluster, sources, eps, dropped) 
end = time.perf_counter()
print(f"Time {end -start} seconds")

