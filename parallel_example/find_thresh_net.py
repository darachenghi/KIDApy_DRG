import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

clusters = [f"{n:04d}" for n in range(128)]
epsilon = [1e-4, 2e-4, 5e-4,
           1e-3, 2e-3, 5e-3,
           1e-2, 2e-2, 3e-2, 5e-2, 7e-2,
           1e-1, 1.5e-1, 2e-1, 3e-1]
sources = ["CO", "C+", "O+", "O", "e-"]
full_n_species = 578

RESULT_DIR = HERE/"results"
FULL_NET_FILE = HERE/"full_net.json"


def find_thresh_net(threshold, epsilon, clusters, result_dir, full_net_file, save_dir):
    epsilon = sorted(epsilon, reverse=True)
    err_cols = [f"rel_err_{s}" for s in sources]

    net_out_dir = save_dir / f"threshold_{threshold}_networks"
    net_out_dir.mkdir(parents=True, exist_ok=True)

    chosen_eps = {}
    n_species = {}
    no_reduce = []

    for c in clusters:
        err_dir = result_dir / f"{c}_results" / "errors"
        net_dir = result_dir / f"{c}_results" / "reduced_networks"
        selected = False

        for e in epsilon:
            e_str = str(e).replace(".", "p")
            err_file = err_dir / f"errors_eps{e_str}.csv"

            # Read only the columns we need, take the global max in one vectorized op.
            max_l2 = pd.read_csv(err_file, usecols=err_cols).to_numpy().max()

            if max_l2 <= threshold:
                net_file = net_dir / f"reduced_net_eps{e}.json"
                with open(net_file) as f:
                    n_species[c] = len(json.load(f)["species"])
                chosen_eps[c] = e
                shutil.copy(net_file, net_out_dir / f"{c}.json")
                selected = True
                break

        if not selected:
            no_reduce.append(c)
            chosen_eps[c] = np.nan
            n_species[c] = full_n_species
            shutil.copy(full_net_file, net_out_dir / f"{c}.json")

    print(f"{len(no_reduce)} clusters not reduceable within threshold error:")
    print(*no_reduce, sep = ', ')
    cluster_df = pd.DataFrame({"epsilon": chosen_eps, "n_species": n_species})
    cluster_df.to_csv(save_dir / f"threshold_{threshold}.csv", index_label="Cluster")
    return chosen_eps


if __name__ == "__main__":
    find_thresh_net(0.01, epsilon, clusters, RESULT_DIR, FULL_NET_FILE, save_dir=HERE)