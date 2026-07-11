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

FULL_N_SPECIES = len(net.species)     #baseline (full network) sizes the DRG reduces from,
FULL_N_REACTIONS = len(net.reactions) #used to report reduction percentages per epsilon

#Define Source Species and Tolerance
sources = ['CO', 'C+', 'O+', 'O', 'e-']
#log-spaced grid (denser through the 0.005-0.1 collapse region) for a systematic error-vs-eps study
eps = [1e-4, 2e-4, 5e-4,
       1e-3, 2e-3, 5e-3,
       1e-2, 2e-2, 3e-2, 5e-2, 7e-2,
       1e-1, 1.5e-1, 2e-1, 3e-1, 5e-1]   # 16 points

cluster_name = "0025"

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
COMPUTE_FULL_REF = True #solve the full (unreduced) network as an eps=0 reference for the error plots

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
    #shift time so the interval starts at 0: absolute cluster times (~1e12 s) have
    #a float spacing coarser than the initial step the stiff system needs, which
    #makes the solver fail with "required step size less than spacing between numbers".
    #The ODE is autonomous (rates depend only on the fixed interval environment),
    #so shifting is exact; absolute times are restored after the solve.
    t0 = t_eval[0]
    t_eval_shifted = t_eval - t0

    try:
        solver = QuadraticSolver()
        t, y = solver.solve(A, B,
                             x0=x0,
                             atol=ATOL,
                             rtol=RTOL,
                             method="BDF",
                             t_eval=t_eval_shifted,
                             t_span=(t_eval_shifted[0], t_eval_shifted[-1]))
        t = t + t0  #restore absolute time

        for s in sources:
            true_sol = chunk[:,cluster_species_map[s]+7]
            red_sol = y[species_map[s],:]
            res = np.abs(true_sol - red_sol)
            res_norm = np.divide(res, np.abs(true_sol),out=np.zeros_like(res, dtype=float),where=true_sol != 0,)
            rel_errors[s] = ((np.linalg.norm(res,2))/np.linalg.norm(true_sol, 2))
            max_errors[s] = (np.max(res_norm))

    except RuntimeError:
        solve_failed = True
        print(f"Interval {idx} (rows {start}:{end}) failed to solve")

    return idx, start, end, t, y, env, rel_errors, max_errors, solve_failed

if __name__ == '__main__':
    n_rows = len(cluster)
    starts = list(range(0, n_rows, CHUNK_SIZE)) #rows where each interval starts

    intervals = [(k, s, min(s + CHUNK_SIZE, n_rows)) for k, s in enumerate(starts)] #list of (idx, start, end) for each interval
    n_intervals = len(intervals)

    FAILED_LOG = SAVE_DIR / "failed_intervals.txt"
    if FAILED_LOG.exists():
        FAILED_LOG.unlink() #start fresh each run so stale failures aren't accumulated

    #per-failure environment diagnostics (which physical conditions break the solver)
    FAILED_ENV_CSV = ERROR_DIR / "failed_intervals_env.csv"
    failed_env_rows = []  #(eps, interval, nH, T, Av, uv_flux) accumulated across all eps

    error_header = ["interval"] + [f"rel_err_{s}" for s in sources] + [f"max_err_{s}" for s in sources]

    #full-network (eps=0) reference pass: solve the UNREDUCED network over every interval so the
    #error plots show where "no reduction" sits. Reuses solve_interval with the full network as the
    #worker network. This is the most expensive single pass (largest network); gate with COMPUTE_FULL_REF.
    if COMPUTE_FULL_REF:
        print("solving FULL network (eps=0 reference) over all intervals ...")
        full_rel = [None] * n_intervals
        full_max = [None] * n_intervals
        with ProcessPoolExecutor(initializer=_init_worker,
                                 initargs=(net.species, net.reactions, net.species_map)) as executor:
            futures = {executor.submit(solve_interval, idx, start, end): idx
                       for idx, start, end in intervals}
            for future in as_completed(futures):
                idx, start, end, t, y, env, rel_errors, max_errors, solve_failed = future.result()
                full_rel[idx] = rel_errors
                full_max[idx] = max_errors
        full_rows = [[idx] + [full_rel[idx][s] for s in sources] + [full_max[idx][s] for s in sources]
                     for idx in range(n_intervals)]
        np.savetxt(str(ERROR_DIR / "errors_full.csv"), np.array(full_rows, dtype=np.float64),
                   delimiter=",", header=",".join(error_header), comments="")
        print(f"eps=0 (full network): saved {ERROR_DIR / 'errors_full.csv'}")

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
                    failed_env_rows.append([e, idx, env["nH"], env["T"], env["Av"], env["uv_flux"]])
                    continue
                results[idx] = (t, y)

        if failed:
            failed_msg = f"eps={e}: {len(failed)}/{n_intervals} intervals failed: {sorted(failed)}"
            with open(FAILED_LOG, "a") as f:
                f.write(failed_msg + "\n") #saves failed solve intervals to txt file in save directory

        error_rows = []
        for idx in range(n_intervals):
            rel_row = [rel_errors_by_interval[idx][s] for s in sources]
            max_row = [max_errors_by_interval[idx][s] for s in sources]
            error_rows.append([idx] + rel_row + max_row)
        error_out = np.array(error_rows, dtype=np.float64)

        error_root = str(ERROR_DIR / f"errors_eps{str(e).replace('.', 'p')}")
        np.savetxt(error_root + ".csv", error_out, delimiter=",",
                   header=",".join(error_header), comments="")
        print(f"eps={e}: saved {error_root}.csv")

        if all(r is None for r in results):
            continue

        #rebuilds t, y for whole cluster, preserving the full time axis:
        #failed intervals are filled with NaN columns rather than dropped, so
        #the saved solution has no silent time gaps
        n_species = len(species)
        t_parts = []
        y_parts = []
        for idx, start, end in intervals:
            r = results[idx]
            if r is not None:
                t_i, y_i = r
            else:
                t_i = cluster[start:end, 1]
                y_i = np.full((n_species, end - start), np.nan, dtype=np.float64)
            t_parts.append(t_i)
            y_parts.append(y_i)

        t_full = np.concatenate(t_parts)
        y_full = np.hstack(y_parts)

        out_root = str(SOLVED_DIR / f"reduced_sol_eps{str(e).replace('.', 'p')}")
        solver = QuadraticSolver()
        solver.save(out_root, t_full, y_full, col_names=species_map.keys()) #save full solution for all species in reduced network
        print(f"eps={e}: saved {out_root}.csv")

    #environment of every failed interval, across all eps, for diagnosing solver failures
    if failed_env_rows:
        np.savetxt(FAILED_ENV_CSV, np.array(failed_env_rows, dtype=np.float64), delimiter=",",
                   header="epsilon,interval,nH,T,Av,uv_flux", comments="")
        print(f"saved failed-interval environments to {FAILED_ENV_CSV}")

###########################################Plotting Errors##########################################

PLOTS_DIR = SAVE_DIR / "error_plots" 
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

max_rel_errors = {s: [] for s in sources}
max_max_errors = {s: [] for s in sources}
avg_rel_errors = {s: [] for s in sources}
avg_max_errors = {s: [] for s in sources}
n_failed_by_eps = []  #number of failed intervals per eps (all-source-NaN rows)

for e in eps:
    error_file = ERROR_DIR / f"errors_eps{str(e).replace('.', 'p')}.csv"
    df = pd.read_csv(error_file)

    rel_cols = [f"rel_err_{s}" for s in sources]
    n_failed_by_eps.append(int(df[rel_cols].isna().all(axis=1).sum()))

    for s in sources:
        rel_col = df[f"rel_err_{s}"]
        max_col = df[f"max_err_{s}"]
        max_rel_errors[s].append(np.nanmax(rel_col))
        max_max_errors[s].append(np.nanmax(max_col))
        avg_rel_errors[s].append(np.nanmean(rel_col))
        avg_max_errors[s].append(np.nanmean(max_col))

#full-network (eps=0) reference aggregates, drawn as horizontal lines on each plot for sanity-checking
full_ref_file = ERROR_DIR / "errors_full.csv"
full_max_rel = full_max_max = full_avg_rel = full_avg_max = None
if full_ref_file.exists():
    dff = pd.read_csv(full_ref_file)
    full_max_rel = {s: np.nanmax(dff[f"rel_err_{s}"]) for s in sources}
    full_max_max = {s: np.nanmax(dff[f"max_err_{s}"]) for s in sources}
    full_avg_rel = {s: np.nanmean(dff[f"rel_err_{s}"]) for s in sources}
    full_avg_max = {s: np.nanmean(dff[f"max_err_{s}"]) for s in sources}

def _plot_errors(data, ylabel, title, filename, ref=None):
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    fig.set_size_inches((6, 5))
    for s in sources:
        line, = ax.plot(eps, data[s], marker='o', label=s)
        if ref is not None:  #full-network (eps=0) floor for this species, same color, dotted
            ax.axhline(ref[s], color=line.get_color(), linestyle=':', linewidth=1, alpha=0.7)
    if ref is not None:
        ax.plot([], [], color="k", linestyle=":", linewidth=1, label="full network (eps=0)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Cluster {cluster_name}: {title}")
    ax.legend()
    fig.savefig(PLOTS_DIR / filename)
    plt.close(fig)

_plot_errors(max_rel_errors, "Max Relative Error", "Max Relative Error", "max_rel_error", ref=full_max_rel)
_plot_errors(max_max_errors, "Max Max Error", "Max Max Error", "max_max_error", ref=full_max_max)
_plot_errors(avg_rel_errors, "Average Relative Error", "Average Relative Error", "avg_rel_error", ref=full_avg_rel)
_plot_errors(avg_max_errors, "Average Max Error", "Average Max Error", "avg_max_error", ref=full_avg_max)

print(f"Saved error plots to {PLOTS_DIR}")

###########################################Reduced Network Size vs Epsilon##########################################

n_species_by_eps = []
n_reactions_by_eps = []
for e in eps:
    reduced_net_file = RED_NET_DIR / f"reduced_net_eps{e}.json"
    with open(reduced_net_file) as f:
        data = json.load(f)
    n_species_by_eps.append(len(data["species"]))
    n_reactions_by_eps.append(len(data["reactions"]))

#save the plotted data (epsilon, reduced species & reaction counts) so it's persisted, not only plotted
size_out = np.array([[e, ns, nr] for e, ns, nr in zip(eps, n_species_by_eps, n_reactions_by_eps)],
                    dtype=np.float64)
size_csv = ERROR_DIR / "network_size_vs_eps.csv"
np.savetxt(size_csv, size_out, delimiter=",",
           header="epsilon,n_species,n_reactions", comments="")
print(f"Saved reduced-network size data to {size_csv}")

def _plot_size(data, ylabel, title, filename):
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    fig.set_size_inches((6, 5))
    ax.plot(eps, data, marker='o')
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Cluster {cluster_name}: {title}")
    fig.savefig(PLOTS_DIR / filename)
    plt.close(fig)

_plot_size(n_species_by_eps, "Number of Species", "Number of Species vs Epsilon", "n_species_vs_eps")
_plot_size(n_reactions_by_eps, "Number of Reactions", "Number of Reactions vs Epsilon", "n_reactions_vs_eps")

print(f"Saved reduced-network size plots to {PLOTS_DIR}")

###########################################Master Summary##########################################

#one consolidated row per epsilon: reduced-network size, reduction vs the full network,
#failure count, and error aggregated (max over sources, then max/mean over intervals)
summary_rows = []
for i, e in enumerate(eps):
    ns = n_species_by_eps[i]
    nr = n_reactions_by_eps[i]
    summary_rows.append([
        e,
        ns,
        nr,
        100.0 * (1.0 - ns / FULL_N_SPECIES),    #species reduction %
        100.0 * (1.0 - nr / FULL_N_REACTIONS),  #reaction reduction %
        n_failed_by_eps[i],
        max(max_rel_errors[s][i] for s in sources),  #worst-source max relative error
        max(avg_rel_errors[s][i] for s in sources),  #worst-source avg relative error
        max(max_max_errors[s][i] for s in sources),  #worst-source max pointwise error
        max(avg_max_errors[s][i] for s in sources),  #worst-source avg pointwise error
    ])
summary_header = ("epsilon,n_species,n_reactions,species_reduction_pct,reaction_reduction_pct,"
                  "n_failed,max_rel_err,avg_rel_err,max_max_err,avg_max_err")
summary_csv = SAVE_DIR / "summary.csv"
np.savetxt(summary_csv, np.array(summary_rows, dtype=np.float64), delimiter=",",
           header=summary_header, comments="")
print(f"Saved master summary to {summary_csv}")
