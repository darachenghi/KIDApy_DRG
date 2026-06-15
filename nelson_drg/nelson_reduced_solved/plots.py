import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

YEAR = 3600 * 24 * 365.25
folder_path = "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/nelson_drg/nelson_reduced_solved"
eps = ['0p1', '0p2', '0p6', '0p7']

species = ["CO", "e-"]

for s in species:
    plt.figure()
    df = pd.read_csv("/oden/cheng/Downloads/code/DRG/KIDApy_DRG/examples/data/nelson_point/nelson_point.csv")
    plt.plot(df["t"]/YEAR, df[s])

    for e in eps:
        f = f'{folder_path}/{e}eps.csv'
        df = pd.read_csv(f)
        plt.plot(df["t"]/YEAR, df[s], linestyle = 'dashed')

    plt.loglog()
    plt.xlabel("Time (Years)")
    plt.ylabel('Abundance per H')
    plt.legend(["full order", "eps = 0.1", "eps = 0.2", "eps = 0.6", "eps = 0.7"])
    plt.title(f'{s}')
    folder = Path("/oden/cheng/Downloads/code/DRG/KIDApy_DRG/nelson_drg")
    file = folder/f'{s} Concentration Comparison'
    plt.savefig(file)