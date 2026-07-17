"""Solves reduced network for test tracers
    tracers are read using read_feautre_matrix.ipynb 
    and t, y, and params are saved in .npz files"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solver import QuadraticSolverTracer
from reduced_parser import reduced_network, load_abundances
from parser import Network

########################################################################PATHS###########################################################################

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

NETWORK_PATH = REPO_ROOT/ "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
ABUNDANCES_PATH = REPO_ROOT/ "networks" / "kida.uva.2024" / "abundances.in"
REDUCED_NETWORK_DIR = REPO_ROOT.parent/"DRG_DATA"/"global_reduced_networks"
TRACER_DIR = REPO_ROOT.parent/"test_tracers"

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3
MIN_SCALE = 1e-22
PARAMS_STRIDE = 20  #num rows env params are the same in tracer

################################################################LOADING UNREDUCED NET###########################################################################
net = Network(grains = True)
net.load_from_disk(str(NETWORK_PATH))
tracer_species_map = net.species_map #indexing of species in tracer, for plotting true solutions

#SET EPSILONS
n_species = 578
eps = [1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 3e-2, 5e-2, 7e-2,
    1e-1, 1.5e-1, 2e-1, 3e-1, 5e-1]


num_species = []
sources = ['CO', 'C+', 'O+', 'O', 'e-']
source_indices = [tracer_species_map[s]for s in sources]

tracer_positions = [i for i in range(50)]

for pos in tracer_positions:
    POSITION = pos #Tracer Position
    FEATURE_PATH = TRACER_DIR/f"tracer_{str(POSITION)}.npz"

    if not FEATURE_PATH.exists():
        print(f"Can't find npz file for tracer {pos}")
        continue

    SAVE_DIR = TRACER_DIR/"results"/f"{POSITION}"
    SAVE_DIR.mkdir(parents=True, exist_ok= True)
    SOL_DIR = SAVE_DIR/"solutions"
    SOL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR = SAVE_DIR/"plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    #################################################################READ TRACER############################################################################
    tracer = np.load(FEATURE_PATH)
    t, y, params = tracer["t"], tracer["y"], tracer["params"]

    params = params[::PARAMS_STRIDE]
    dt_hydro = (t[1] - t[0]) * PARAMS_STRIDE

    ##################################################################INTEGRATE REDUCED NET##################################################################
    for e in eps:
        #LOAD REDUCED NETWORK
        REDUCED_NETWORK_PATH = REDUCED_NETWORK_DIR/f"reduced_net_eps{e}.json"
        red_net = reduced_network(grains=True)
        red_net.load_from_disk(REDUCED_NETWORK_PATH)
        species = red_net.species
        num_species.append(len(species))
        species_map = red_net.species_map

        #GETS INITIAL CONDITIONS (using abundance files)
        abund = load_abundances(str(ABUNDANCES_PATH))
        abund["e-"] = sum(val for name,val in abund.items() if name.endswith("+"))
        x0 = np.zeros(len(species), dtype=np.float64)
        for name, val in abund.items():
            if name in species_map:
                x0[species_map[name]] = val

        #GETS INITIAL CONDITIONS (using first state in tracer)
        #init_state = y[:,0]
        #x0 = np.zeros(len(species), dtype=np.float64)

        #for s, sidx in tracer_species_map.items():  #first row of tracer gives initial conditions
        #   if s in species_map:
        #      x0[species_map[s]] = init_state[sidx]

        #INTEGRATES
        solver = QuadraticSolverTracer()
        t_red,y_red = solver.solve(dt_hydro = dt_hydro,
                        pt = params,
                        get_tensors= red_net.get_operators,
                        x0 = x0,
                        atol = ATOL,
                        rtol = RTOL,
                        min_scale=MIN_SCALE,
                        t_eval=t)

        filepath = str(SOL_DIR/f"eps_{str(e).replace('.','p')}")
        solver.save_data(t_red, y_red, params, species, dt_hydro = dt_hydro, save_path = filepath)


    ########################################################################ERRORS###########################################################################
    rel_errors = {s: {} for s in sources}
    max_errors = {s: {} for s in sources}

    for e in eps:
        filepath = str(SOL_DIR / f"eps_{str(e).replace('.', 'p')}")
        df = pd.read_csv(filepath + ".csv")
        red_t = df["t"].to_numpy()

        for s in sources:
            true_sol = y[tracer_species_map[s], :]
            red_sol = np.interp(t, red_t, df[s].to_numpy())
            res = np.abs(true_sol - red_sol)
            res_norm = np.divide(res, np.abs(true_sol),out=np.zeros_like(res, dtype=float),where=true_sol != 0,)
            rel_errors[s][e] = ((np.linalg.norm(res,2))/np.linalg.norm(true_sol, 2))
            max_errors[s][e] = (np.max(res_norm))

    l2_err_df = pd.DataFrame(rel_errors).T
    l2_err_df.index.name = "species"
    l2_err_df.columns.name = "eps"
    l2_err_df.to_csv(SAVE_DIR / "l2_error_vs_epsilon.csv")

    max_err_df = pd.DataFrame(max_errors).T
    max_err_df.index.name = "species"
    max_err_df.columns.name = "eps"
    max_err_df.to_csv(SAVE_DIR / "max_error_vs_epsilon.csv")

    #PLOTS ERRORS VS EPSILON
    for err_df, err_name in [(l2_err_df, "l2"), (max_err_df, "max")]:
        plt.figure(figsize=(6, 5))
        for s in sources:
            plt.plot(err_df.columns, err_df.loc[s], marker="o", label=s)
        plt.loglog()
        plt.xlabel("epsilon")
        plt.ylabel(f"{err_name} relative error")
        plt.legend()
        plt.title(f"{err_name} error vs epsilon")
        plt.savefig(PLOTS_DIR / f"{err_name}_error_vs_epsilon.png", dpi=400)
        plt.close()

    ######################################################################PLOTTING QOIS######################################################################
    #SET EPSILONS
    n_species = 578
    eps = [0.0001, 0.01, 0.1, 0.3, 0.5]

    sources = ['CO', 'C+', 'O+', 'O', 'e-']
    source_indices = [tracer_species_map[s]for s in sources]

    num_species = []
    for e in eps:
        filepath = str(SOL_DIR / f"eps_{str(e).replace('.', 'p')}")
        df = pd.read_csv(filepath + ".csv")
        num_species.append(len(df.columns) - 1)  # exclude "t" column

    label = ["True"] + [f"eps={e} (N={n})" for e, n in zip(eps, num_species)]

    for s in sources:
        plt.figure(figsize=(6, 5))
        plt.plot(t / YEAR, y[tracer_species_map[s], :])

        for e in eps:
            filepath = str(SOL_DIR / f"eps_{str(e).replace('.', 'p')}")
            df = pd.read_csv(filepath + ".csv")
            plt.plot(df["t"] / YEAR, df[s], linestyle="dashed")

        plt.loglog()
        plt.xlabel("Time (Years)")
        plt.ylabel("Abundance per H")
        plt.legend(label)
        plt.title(s)
        plt.savefig(PLOTS_DIR / f"{s}.png", dpi=400)
        plt.close()
