import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

YEAR = 3600 * 24 * 365.25
folder_path = "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/kida_reduced_solved"
eps = ['0p01','0p1', '0p2', '0p5', '0p5']

species = ["CO", "C+", "O+", "O"]

for s in species:
    plt.figure()
    df = pd.read_csv("/oden/cheng/Downloads/code/DRG/KIDApy_DRG/examples/data/kida_uva_2024_point/kida_uva_2024_point.csv")
    plt.plot(df["t"]/YEAR, df[s])

    for e in eps:
        f = f'{folder_path}/{e}eps_sub.csv'
        df = pd.read_csv(f)
        plt.plot(df["t"]/YEAR, df[s], linestyle = 'dashed')

    plt.loglog()
    plt.xlabel("Time (Years)")
    plt.ylabel('Abundance per H')
    plt.legend(["full order", "eps = 0.01", "eps = 0.1", "eps = 0.2", "eps = 0.4", "eps = 0.5"])
    plt.title(f'{s}')
    folder = Path("/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/kida_reduced_solved")
    file = folder/f'{s} Concentration Comparison'
    plt.savefig(file)