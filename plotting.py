import json
import matplotlib.pyplot as plt

files = ["kida_eps_results_union.json", "kida_eps_results_max.json"]
plt.figure()

for file in files:
    with open(file) as f:
        data = json.load(f)
    epsilons = data["epsilons"]
    n_species = [len(s) for s in data["species"]]

    plt.plot(epsilons, n_species, marker = 'o')

plt.xlabel("Epsilon")
plt.ylabel("Number of Species")
plt.legend(["Union", "Max"])
plt.title("DRG Reducetion of KIDA Network")
plt.savefig("kida_eps")