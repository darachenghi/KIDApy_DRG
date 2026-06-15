import json
import matplotlib.pyplot as plt

files = ["/oden/cheng/Downloads/code/DRG/KIDApy_DRG/nelson_drg/nelson_eps_results_union.json"]
plt.figure()

for file in files:
    with open(file) as f:
        data = json.load(f)
    epsilons = data["epsilon"]
    n_species = [len(s) for s in data["species"]]
    print(f'epsilons:{epsilons}')
    print(f'number of species: {n_species}')
    plt.plot(epsilons, n_species, marker = 'o')

plt.xlabel("Epsilon")
plt.ylabel("Number of Species")
plt.title("DRG Reduction of Nelson Network")
plt.savefig("/oden/cheng/Downloads/code/DRG/KIDApy_DRG/nelson_drg/nelson_eps")

