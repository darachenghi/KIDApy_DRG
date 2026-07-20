"""Solves reduced network for test tracers"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  #headless backend for batch/compute nodes (no display)

from concurrent.futures import ProcessPoolExecutor, as_completed

from solver import QuadraticSolverTracer
from reduced_parser import reduced_network
from parser import Network
from read_tracer import read_tracer_trajectory, remove_duplicates, number_tracers

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

########################################################################CONFIG SAME ACROSS ALL TRACERS########################################################

NETWORK_PATH = REPO_ROOT/ "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
ABUNDANCES_PATH = REPO_ROOT/ "networks" / "kida.uva.2024" / "abundances.in"
REDUCED_NETWORK_DIR = REPO_ROOT.parent/"DRG_DATA"/"global_reduced_networks"
FEAT_PATH =  "./feature_matrix.npy"
SAVE_DIR = HERE/"results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3
MIN_SCALE = 1e-22
PARAMS_STRIDE = 20  #num rows env params are the same in tracer

#SET EPSILONS
eps = [1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 3e-2, 5e-2, 7e-2,
    1e-1, 1.5e-1, 2e-1, 3e-1, 5e-1]

n_species = 578

PLOT_EPS = [0.0001, 0.01, 0.1, 0.3, 0.5]

sources = ['CO', 'C+', 'O+', 'O', 'e-']

mm = np.load(FEAT_PATH, mmap_mode="r") #Loads tracer

net = Network(grains = True)
net.load_from_disk(str(NETWORK_PATH))
tracer_species_map = net.species_map #indexing of species in tracer, for plotting true solutions

sample_size = 200
n_tracers = number_tracers(FEAT_PATH)
tracer_positions = [i for i in range(n_tracers)]

########################################################################POOL PROCESS FUNCTION###########################################################################

def _init_worker(tracer_species_map, mm, n_species, savedir):
    '''initializes constant parameters for process_tracer during pool processes'''
    global _worker_tracer_species_map, _worker_mm, _worker_n_species, _worker_savedir
    _worker_tracer_species_map = tracer_species_map
    _worker_mm = mm
    _worker_n_species = n_species
    _worker_savedir = savedir


def _solve_tracer(pos, tracer_species_map, mm, n_species, sol_dir):
    '''reads a tracer's true trajectory and integrates the reduced network for every
       epsilon, saving each solution to sol_dir. Returns the true (t, y) for downstream
       error/plot steps, or ok=False with a message if the tracer is empty or a solve fails'''

    t, y, params = read_tracer_trajectory(mm, pos, n_species, return_params=True)
    t, y, params = remove_duplicates(t, y, params)

    if y.shape[1] == 0:
        return False, f"Tracer {pos} is empty block", None, None

    params = params[::PARAMS_STRIDE]
    dt_hydro = (t[1] - t[0]) * PARAMS_STRIDE

    for e in eps:
        try:
            #LOAD REDUCED NETWORK
            REDUCED_NETWORK_PATH = REDUCED_NETWORK_DIR/f"reduced_net_eps{e}.json"
            red_net = reduced_network(grains=True)
            red_net.load_from_disk(REDUCED_NETWORK_PATH)
            species = red_net.species
            species_map = red_net.species_map

            #GETS INITIAL CONDITIONS
            x0 = np.zeros(len(species), dtype=np.float64)
            init_state = y[:, 0]

            for s, sidx in tracer_species_map.items():  #first row of cluster chunk gives initial conditions
                if s in species_map:
                    x0[species_map[s]] = init_state[sidx]

            #INTEGRATES
            solver = QuadraticSolverTracer()
            t_red, y_red = solver.solve(dt_hydro=dt_hydro,
                            pt=params,
                            get_tensors=red_net.get_operators,
                            x0=x0,
                            atol=ATOL,
                            rtol=RTOL,
                            min_scale=MIN_SCALE,
                            t_eval=t,
                            equilibrate=False)

            filepath = str(sol_dir/f"eps_{str(e).replace('.','p')}")
            solver.save_data(t_red, y_red, params, species, dt_hydro=dt_hydro, save_path=filepath)

        except RuntimeError as err:
            return False, f"Tracer {pos} failed to solve: {err}", None, None

    return True, None, t, y


def _compute_tracer_errors(t, y, tracer_species_map, sol_dir, tracer_dir):
    '''computes L2 relative and max pointwise errors vs the true solution for each source
       species across all epsilons, saves them to csv, and returns the two error DataFrames'''

    rel_errors = {s: {} for s in sources}
    max_errors = {s: {} for s in sources}

    for e in eps:
        filepath = str(sol_dir / f"eps_{str(e).replace('.', 'p')}")
        df = pd.read_csv(filepath + ".csv")
        red_t = df["t"].to_numpy()

        for s in sources:
            true_sol = y[tracer_species_map[s], :]
            red_sol = np.interp(t, red_t, df[s].to_numpy())
            res = np.abs(true_sol - red_sol)
            res_norm = np.divide(res, np.abs(true_sol), out=np.zeros_like(res, dtype=float), where=true_sol != 0)
            rel_errors[s][e] = (np.linalg.norm(res, 2)) / np.linalg.norm(true_sol, 2)
            max_errors[s][e] = np.max(res_norm)

    l2_err_df = pd.DataFrame(rel_errors).T
    l2_err_df.index.name = "species"
    l2_err_df.columns.name = "eps"
    l2_err_df.to_csv(tracer_dir / "l2_error_vs_epsilon.csv")

    max_err_df = pd.DataFrame(max_errors).T
    max_err_df.index.name = "species"
    max_err_df.columns.name = "eps"
    max_err_df.to_csv(tracer_dir / "max_error_vs_epsilon.csv")

    return l2_err_df, max_err_df


def _plot_tracer_errors(l2_err_df, max_err_df, plots_dir):
    '''plots L2 and max error vs epsilon, one line per source species'''

    for err_df, err_name in [(l2_err_df, "l2"), (max_err_df, "max")]:
        plt.figure(figsize=(6, 5))
        for s in sources:
            plt.plot(err_df.columns, err_df.loc[s], marker="o", label=s)
        plt.loglog()
        plt.xlabel("epsilon")
        plt.ylabel(f"{err_name} error")
        plt.legend()
        plt.title(f"{err_name} error vs epsilon")
        plt.savefig(plots_dir / f"{err_name}_error_vs_epsilon.png", dpi=400)
        plt.close()


def _plot_tracer_qois(t, y, tracer_species_map, sol_dir, plots_dir):
    '''plots each source species' true trajectory against the reduced-network solutions
       at PLOT_EPS, one figure per species'''

    num_species = []
    for e in PLOT_EPS:
        filepath = str(sol_dir / f"eps_{str(e).replace('.', 'p')}")
        df = pd.read_csv(filepath + ".csv")
        num_species.append(len(df.columns) - 1)  # exclude "t" column

    label = ["True"] + [f"eps={e} (N={n})" for e, n in zip(PLOT_EPS, num_species)]

    for s in sources:
        plt.figure(figsize=(6, 5))
        plt.plot(t / YEAR, y[tracer_species_map[s], :])

        for e in PLOT_EPS:
            filepath = str(sol_dir / f"eps_{str(e).replace('.', 'p')}")
            df = pd.read_csv(filepath + ".csv")
            plt.plot(df["t"] / YEAR, df[s], linestyle="dashed")

        plt.loglog()
        plt.xlabel("Time (Years)")
        plt.ylabel("Abundance per H")
        plt.legend(label)
        plt.title(s)
        plt.savefig(plots_dir / f"{s}.png", dpi=400)
        plt.close()


def process_tracer(pos):
    '''full pipeline for one tracer: solve (all eps) -> errors -> plot errors -> plot QOIs'''

    tracer_species_map = _worker_tracer_species_map
    mm = _worker_mm
    n_species = _worker_n_species
    savedir = _worker_savedir

    TRACER_DIR = savedir/str(pos)
    PLOTS_DIR = TRACER_DIR/"plots"
    SOL_DIR = TRACER_DIR/"solutions"

    for d in (TRACER_DIR, PLOTS_DIR, SOL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    ########################################### SOLVE REDUCED NETWORK ###########################################
    ok, msg, t, y = _solve_tracer(pos, tracer_species_map, mm, n_species, SOL_DIR)
    if not ok:
        return pos, False, msg

    ########################################### COMPUTE ERRORS ###########################################
    l2_err_df, max_err_df = _compute_tracer_errors(t, y, tracer_species_map, SOL_DIR, TRACER_DIR)

    ########################################### PLOTTING ERRORS ###########################################
    _plot_tracer_errors(l2_err_df, max_err_df, PLOTS_DIR)

    ########################################### PLOTTING QOIS ###########################################
    _plot_tracer_qois(t, y, tracer_species_map, SOL_DIR, PLOTS_DIR)

    return pos, True, None


#MAIN FUNCTION
if __name__ == "__main__":
    failed = []

    with ProcessPoolExecutor(initializer=_init_worker,
                              initargs=(tracer_species_map, mm, n_species, SAVE_DIR)) as executor:
        futures = {executor.submit(process_tracer, pos): pos for pos in tracer_positions}

        for future in as_completed(futures):
            pos, ok, msg = future.result()
            if ok:
                print(f"Tracer {pos}: done")
            else:
                failed.append(pos)
                print(msg)

    if failed:
        print(f"{len(failed)}/{len(tracer_positions)} tracers failed: {sorted(failed)}")
        np.savetxt(str(SAVE_DIR/"failed_solves"),failed)
