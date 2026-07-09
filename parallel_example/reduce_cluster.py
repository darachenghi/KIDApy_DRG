"""Reduces network for a single cluster, solves reduced network for all fixed environment intervals in the cluster, plot errors
   Results are all saved in folder {cluster_number}_results
      subfolders: reduced_networks, reduced_solutions, errors, error_plots"""

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
REPO_ROOT = HERE.parent

###########################################REDUCE NETWORK##########################################

NETWORK_PATH    = REPO_ROOT/ "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
CLUSTER_DIR = Path("/work/10864/arjunveejay/mysharedirectory/clusters_params_only")

#LOADS NETWORK
net = Network(grains = True)
net.load_from_disk(str(NETWORK_PATH))
cluster_species_map = net.species_map #indexing of species in original cluster results
dropped = net.drop_passive_species() 
net._get_stoich(dropped) #adds stoichiometric coefficient dictionary to each reaction

#Define Source Species and Tolerance
sources = ['CO', 'C+', 'O+', 'O', 'e-']
eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

cluster_name = "0000"

CLUSTER_PATH = CLUSTER_DIR/f"cluster_{cluster_name}.npy"

SAVE_DIR = HERE/f"{cluster_name}_results" #All results: reduced network, reduced solution, error, and error plots
SAVE_DIR.mkdir(parents=True, exist_ok= True)

RED_NET_DIR =  SAVE_DIR/ "reduced_networks" #Where the reduced networks for each eps would be saved filename: reduced_net_eps{eps}.json
RED_NET_DIR.mkdir(parents=True, exist_ok=True)

cluster = np.load(CLUSTER_PATH)
drg = DRG_p()
drg.reduce_net(net, cluster, sources, eps, dropped, savedir=RED_NET_DIR) 

###########################################SOLVES REDUCED NETWORK##########################################

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3
CHUNK_SIZE = 20 #20 rows for each interval

SOLVED_DIR = SAVE_DIR / "reduced_solutions" # reduced solutions are saved
SOLVED_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = SAVE_DIR / "errors" #where the errors are saved
ERROR_DIR.mkdir(parents=True, exist_ok=True)

def _init_worker(species, reactions, species_map): 
    '''initializes constant parameters for function solve_interval during pool processes'''
    global _worker_reactions, _worker_species, _worker_species_map
    _worker_reactions = reactions #reactions in cluster skeletal network (before temperature selected)
    _worker_species = species #species in cluster skeletal network 
    _worker_species_map = species_map


def solve_interval(idx, start, end):
    '''solves reduced network for a given interval'''
    chunk = cluster[start:end]
    init_reactions = _worker_reactions
    species = _worker_species
    species_map = _worker_species_map

    solve_failed = False
    t, y = None, None
    rel_errors = {s: np.nan for s in sources}
    max_errors = {s: np.nan for s in sources}

    init_state = chunk[0, :] 

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

    t_eval = chunk[:, 1]

    try:
        solver = QuadraticSolver()
        t, y = solver.solve(A, B,
                             x0=x0,
                             atol=ATOL,
                             rtol=RTOL,
                             method="BDF",
                             t_eval=t_eval,
                             t_span=(t_eval[0], t_eval[-1]))

        for s in sources:
            true_sol = chunk[:,cluster_species_map[s]+7]
            red_sol = y[species_map[s],:]
            res = np.abs(true_sol - red_sol)
            rel_errors[s] = ((np.linalg.norm(res,2))/np.linalg.norm(true_sol, 2))
            max_errors[s] = (np.linalg.norm(res,np.inf))

    except RuntimeError:
        solve_failed = True
        print(f"Interval {idx} (rows {start}:{end}) failed to solve")

    return idx, start, end, t, y, env, rel_errors, max_errors, solve_failed

if __name__ == '__main__':
    n_rows = len(cluster)
    starts = list(range(0, n_rows, CHUNK_SIZE)) #rows where each interval starts

    intervals = [(k, s, min(s + CHUNK_SIZE, n_rows)) for k, s in enumerate(starts)] #list of (idx, start, end) for each interval
    n_intervals = len(intervals)

    for e in eps:
        REDUCED_NET_FILE = RED_NET_DIR/f"reduced_net_eps{e}.json"

        with open(REDUCED_NET_FILE) as f:
            data = json.load(f)

        species = data["species"] 
        reactions = data["reactions"]
        species_map = {s: i for i, s in enumerate(species)}

        results = [None] * n_intervals
        rel_errors_by_interval = [None] * n_intervals
        max_errors_by_interval = [None] * n_intervals
        failed = []

        with ProcessPoolExecutor(initializer=_init_worker,
                                 initargs=(species, reactions, species_map)) as executor:

            futures = {
                executor.submit(solve_interval, idx, start, end): idx
                for idx, start, end in intervals
            }

            for future in as_completed(futures):
                idx, start, end, t, y, env, rel_errors, max_errors, solve_failed = future.result()
                rel_errors_by_interval[idx] = rel_errors
                max_errors_by_interval[idx] = max_errors
                if solve_failed:
                    failed.append(idx)
                    continue
                results[idx] = (t, y)

        if failed:
            failed_msg = f"eps={e}: {len(failed)}/{n_intervals} intervals failed: {sorted(failed)}"
            print(failed_msg)
            with open(SAVE_DIR / "failed_intervals.txt", "a") as f:
                f.write(failed_msg + "\n")

        error_rows = []
        for idx in range(n_intervals):
            rel_row = [rel_errors_by_interval[idx][s] for s in sources]
            max_row = [max_errors_by_interval[idx][s] for s in sources]
            error_rows.append([idx] + rel_row + max_row)
        error_out = np.array(error_rows, dtype=np.float64)
        error_header = ["interval"] + [f"rel_err_{s}" for s in sources] + [f"max_err_{s}" for s in sources]

        error_root = str(ERROR_DIR / f"errors_eps{str(e).replace('.', 'p')}")
        np.savetxt(error_root + ".csv", error_out, delimiter=",",
                   header=",".join(error_header), comments="")
        print(f"eps={e}: saved {error_root}.csv")

        solved = [r for r in results if r is not None]
        if not solved:
            continue

        t_full = np.concatenate([t for t, y in solved]) #rebuilds t, y for whole cluster
        y_full = np.hstack([y for t, y in solved])

        out_root = str(SOLVED_DIR / f"reduced_sol_eps{str(e).replace('.', 'p')}")
        solver = QuadraticSolver()
        solver.save(out_root, t_full, y_full, col_names=species_map.keys()) #save full solution for all species in reduced network
        print(f"eps={e}: saved {out_root}.csv")

###########################################Plotting Errors##########################################

PLOTS_DIR = SAVE_DIR / "error_plots" 
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

max_rel_errors = {s: [] for s in sources}
max_max_errors = {s: [] for s in sources}
avg_rel_errors = {s: [] for s in sources}
avg_max_errors = {s: [] for s in sources}

for e in eps:
    error_file = ERROR_DIR / f"errors_eps{str(e).replace('.', 'p')}.csv"
    df = pd.read_csv(error_file)

    for s in sources:
        rel_col = df[f"rel_err_{s}"]
        max_col = df[f"max_err_{s}"]
        max_rel_errors[s].append(np.nanmax(rel_col))
        max_max_errors[s].append(np.nanmax(max_col))
        avg_rel_errors[s].append(np.nanmean(rel_col))
        avg_max_errors[s].append(np.nanmean(max_col))

def _plot_errors(data, ylabel, title, filename):
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    fig.set_size_inches((6, 5))
    for s in sources:
        ax.plot(eps, data[s], marker='o', label=s)
    ax.set_yscale("log")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Cluster {cluster_name}: {title}")
    ax.legend()
    fig.savefig(PLOTS_DIR / filename)
    plt.close(fig)

_plot_errors(max_rel_errors, "Max Relative Error", "Max Relative Error", "max_rel_error")
_plot_errors(max_max_errors, "Max Max Error", "Max Max Error", "max_max_error")
_plot_errors(avg_rel_errors, "Average Relative Error", "Average Relative Error", "avg_rel_error")
_plot_errors(avg_max_errors, "Average Max Error", "Average Max Error", "avg_max_error")

print(f"Saved error plots to {PLOTS_DIR}")
