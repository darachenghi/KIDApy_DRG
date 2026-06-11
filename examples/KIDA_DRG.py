
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from parser import Network, load_abundances
from solver import QuadraticSolver
from DRG_union import DRG 

# Paths and settings

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT / "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
ABUNDANCES_PATH = REPO_ROOT / "networks" / "kida.uva.2024" / "abundances.in"
SAVE_DIR = HERE / "data" / "kida_uva_2024_point2"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3

# Load network
net = Network(grains=True)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()

# Initial conditions

abund = load_abundances(str(ABUNDANCES_PATH))
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

t_eval = np.logspace(0, np.log10(1e6 * YEAR), 200)

solver = QuadraticSolver()
t, y = solver.solve(
    A, B,
    t_span=(t_eval[0], t_eval[-1]),
    x0=x0,
    atol=ATOL,
    rtol=RTOL,
    t_eval=t_eval,
)

#Getting Reaction rates (k)
reaction_rates = net.reaction_rates(env)

#Sources
sources = ['CO']
source_indices = [net.species_map[i] for i in sources]
eps = 0.5

#DRG 
drg = DRG()
drg.reduce_net(net.reactions, net.species_map, reaction_rates, y, source_indices, dropped, eps = eps)
print(f'Numbers of reactions in reduced networks: {len(drg.reduced_rxns)}')
print(f'Number of species in reduced network: {len(drg.reduced_species)}')