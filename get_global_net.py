import sys
from pathlib import Path
import json
from parser import Network
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERE = Path(__file__).resolve().parent
RESULTS_DIR = Path("/scratch/10864/arjunveejay/KIDApy_DRG/parallel_example/results")
FULL_NET_DIR = ""
SAVE_DIR = HERE/"global_reduced_networks"
SAVE_DIR.mkdir(parents = True, exist_ok= True)

net = Network(grains=True)
net.load_from_disk(FULL_NET_DIR)
net.drop_passive_species()

varients_by_id = {}
for rxn in net.reactions:
    varients_by_id.setdefault(rxn["id", []]).append(rxn)
    
cluster_names = [f"{num:04}" for num in range(128)]
eps = [1e-4, 2e-4, 5e-4,
       1e-3, 2e-3, 5e-3,
       1e-2, 2e-2, 3e-2, 5e-2, 7e-2,
       1e-1, 1.5e-1, 2e-1, 3e-1, 5e-1]

def global_reduce(eps,cluster_names,results_dir, save_dir):
    for e in eps:
        glob_species = set()
        glob_reactions = []
        found_ids = set()

        for c in cluster_names:
            filepath = results_dir/f"{c}_results"/"reduced_networks"/f"reduced_net_eps{e}.json"
            with open(filepath) as f:
                data = json.load(f)
                
                glob_species.update(data["species"])
                for rxn in data["reactions"]:
                    r_id = rxn["id"]
                    if r_id in found_ids:
                        continue
                    glob_reactions.extend(varients_by_id[r_id])
                    found_ids.add(r_id)
        
        glob_species = sorted(glob_species)
        glob_data = {"epsilon": e, "reactions": glob_reactions,"species": glob_species}
        filename = save_dir/f"reduced_net_eps{e}.json"

        with open(filename, "w") as nf:
            json.dump(glob_data, nf, indent = 2)

if __name__ == "__main__":
    global_reduce(eps, cluster_names, RESULTS_DIR, SAVE_DIR)