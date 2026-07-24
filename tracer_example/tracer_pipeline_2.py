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
REDUCED_NETWORK_DIR = REPO_ROOT/"global_reduced_networks_2"
FEAT_PATH =  "/scratch/10864/arjunveejay/KIDApy_DRG/parallel_example/results/feature_matrix.npy"

SAVE_DIR = HERE/"results"
ERROR_DIR = SAVE_DIR/"errors" #where errors are stored (l2 and max)
SOLUTIONS_DIR = SAVE_DIR/"solutions" #where solutions are stored, subfolders for each epsilon where solutions for each tracer are stored as .npy files
for d in (SAVE_DIR, ERROR_DIR, SOLUTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

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

EPS_TAGS = {e: str(e).replace(".", "p") for e in eps} #Create the epsilon subfolder in solutions
for e in eps:
    (SOLUTIONS_DIR/f"eps_{EPS_TAGS[e]}").mkdir(parents=True, exist_ok=True)

n_species = 578

sources = ['CO', 'C+', 'O+', 'O', 'e-']

mm = np.load(FEAT_PATH, mmap_mode="r") #Loads tracer

net = Network(grains = True)
net.load_from_disk(str(NETWORK_PATH))
tracer_species_map = net.species_map #indexing of species in tracer, for plotting true solutions

n_tracers = number_tracers(FEAT_PATH)
tracer_positions = list(range(n_tracers))

########################################################################POOL PROCESS FUNCTION###########################################################################

def _init_worker(tracer_species_map, mm, n_species, solutions_dir):
    '''initializes constant parameters for process_tracer during pool processes'''
    global _worker_tracer_species_map, _worker_mm, _worker_n_species, _worker_solutions_dir
    _worker_tracer_species_map = tracer_species_map
    _worker_mm = mm
    _worker_n_species = n_species
    _worker_solutions_dir = solutions_dir


def _concat_chunks(all_t, all_y):
    '''removes duplicates after solve'''
    t_out = [all_t[0]]
    y_out = [all_y[0]]
    for i in range(1, len(all_t)):
        t_out.append(all_t[i][1:])
        y_out.append(all_y[i][:, 1:])
    return np.concatenate(t_out), np.hstack(y_out)


def _solve_tracer(pos, tracer_species_map, mm, n_species, solutions_dir):
    '''reads a tracer's true trajectory and integrates the reduced network for every
       epsilon, saving each solution, returns (t, y) or ok=False with a message if the tracer is empty or a solve fails'''

    t, y, params_raw = read_tracer_trajectory(mm, pos, n_species, return_params=True)

    if y.shape[1] == 0:
        return False, f"Tracer {pos} is empty block", None, None

    pt = params_raw[::PARAMS_STRIDE]
    dt_hydro = t[PARAMS_STRIDE] - t[0]
    params = np.vstack([pt, pt[-1]])

    # Deduplicate the fine trajectory used for initial conditions, t_eval, and the true-solution comparison.
    t, y, _ = remove_duplicates(t, y, params_raw)

    l2_errors = {}
    max_errors = {}

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
            all_t, all_y = solver.solve(dt_hydro=dt_hydro,
                            pt=params,
                            get_tensors=red_net.get_operators,
                            x0=x0,
                            atol=ATOL,
                            rtol=RTOL,
                            min_scale=MIN_SCALE,
                            t_eval=t,
                            equilibrate=False)

            t_red, y_red = _concat_chunks(all_t, all_y)

            #SAVE SOLUTION: columns = time, species (alphabetical order)
            sorted_species = sorted(species)
            eps_dir = solutions_dir/f"eps_{EPS_TAGS[e]}"
            species_path = eps_dir/"species.npy"
            if not species_path.exists():
                np.save(species_path, np.array(sorted_species))

            col_order = [species_map[s] for s in sorted_species]
            out = np.vstack((t_red, y_red[col_order, :])).T  # (Nt, 1 + n_species)
            np.save(eps_dir/f"tracer_{pos}.npy", out)

            #COMPUTE ERRORS FOR SOURCE SPECIES
            rel_e = {}
            max_e = {}
            for s in sources:
                true_sol = y[tracer_species_map[s], :]
                red_sol = np.interp(t, t_red, y_red[species_map[s], :])
                res = np.abs(true_sol - red_sol)
                res_norm = np.divide(res, np.abs(true_sol), out=np.zeros_like(res, dtype=float), where=true_sol != 0)
                rel_e[s] = (np.linalg.norm(res, 2)) / np.linalg.norm(true_sol, 2)
                max_e[s] = np.max(res_norm)
            l2_errors[e] = rel_e
            max_errors[e] = max_e

        except RuntimeError as err:
            return False, f"Tracer {pos} failed to solve: {err}", None, None

    return True, None, l2_errors, max_errors


def process_tracer(pos):
    '''full pipeline for one tracer: solve + save solution + compute errors for every epsilon'''

    tracer_species_map = _worker_tracer_species_map
    mm = _worker_mm
    n_species = _worker_n_species
    solutions_dir = _worker_solutions_dir

    ok, msg, l2_errors, max_errors = _solve_tracer(pos, tracer_species_map, mm, n_species, solutions_dir)
    if not ok:
        return pos, False, msg, None, None

    return pos, True, None, l2_errors, max_errors


#MAIN FUNCTION
if __name__ == "__main__":
    failed = []
    l2_agg = {e: {} for e in eps}
    max_agg = {e: {} for e in eps}

    with ProcessPoolExecutor(initializer=_init_worker,
                              initargs=(tracer_species_map, mm, n_species, SOLUTIONS_DIR)) as executor:
        futures = {executor.submit(process_tracer, pos): pos for pos in tracer_positions}

        for future in as_completed(futures):
            pos, ok, msg, l2_errors, max_errors = future.result()
            if ok:
                for e in eps:
                    l2_agg[e][pos] = l2_errors[e]
                    max_agg[e][pos] = max_errors[e]
                print(f"Tracer {pos}: done")
            else:
                failed.append(pos)
                print(msg)

    #WRITE AGGREGATED ERROR CSVS: rows = tracers, columns = sources, one file per (eps, metric)
    for e in eps:
        tag = EPS_TAGS[e]
        pd.DataFrame(l2_agg[e]).T.sort_index().to_csv(ERROR_DIR/f"l2_error_eps_{tag}.csv", index_label="tracer")
        pd.DataFrame(max_agg[e]).T.sort_index().to_csv(ERROR_DIR/f"max_error_eps_{tag}.csv", index_label="tracer")

    if failed:
        print(f"{len(failed)}/{len(tracer_positions)} tracers failed: {sorted(failed)}")
        np.savetxt(str(SAVE_DIR/"failed_solves"),failed)
