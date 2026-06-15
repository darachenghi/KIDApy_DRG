import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

YEAR = 3600 * 24 * 365.25

files = ["/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/kida_reduced_solved/kida_reduced_eps0.01.csv", 
         "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/kida_reduced_solved/kida_reduced_eps0.1.csv",
         "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/kida_reduced_solved/kida_reduced_eps0.2.csv",
         "/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/kida_reduced_solved/kida_reduced_eps0.01_max.csv"]

species = ["C", "CO", "O", "O+"]

for s in species:
    plt.figure()
    df = pd.read_csv("/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/kida_reduced_solved/kida_uva_2024_point.csv")
    plt.plot(df["t"]/YEAR, df[s])

    for f in files:
        df = pd.read_csv(f)
        plt.plot(df["t"]/YEAR, df[s], linestyle = 'dashed')

    plt.loglog()
    plt.xlabel("Time (Years)")
    plt.ylabel('Abundance per H')
    plt.legend(["full order", "eps = 0.01", "eps = 0.1", "eps = 0.2", "eps = 0.01 Max"])
    plt.title(f'{s}')
    folder = Path("/oden/cheng/Downloads/code/DRG/KIDApy_DRG/kida_drg/kida_reduced_solved/plots")
    file = folder/f'{s} Concentration Comparison'
    plt.savefig(file)