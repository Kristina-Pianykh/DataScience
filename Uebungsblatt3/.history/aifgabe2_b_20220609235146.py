from pathlib import Path
from pickle import TRUE
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def read_values(input_file: Path) -> pd.DataFrame:
    data = pd.read_csv(
        input_file,
        sep=" ",
        names=["Number", "Length", "Left", "Right", "Bottom", "Top", "Diagonal"],
        dtype={"Number": int, "Length": float, "Left": float, "Right": float, "Bottom": float, "Top": float, "Diagonal": float},
        skipinitialspace=True
    )
    return data

    
def main():
    file = Path("banknote.dat")
    data = read_values(file)
    bottom = data["Bottom"]
    top = data["Top"]
    print(f"mean bottom: {bottom.mean()}")
    print(f"mean bottom: {top.mean()}")

    #boxplot
    sns.set_style("whitegrid")
    plot = sns.boxplot(data=data[["Bottom", "Top"]])
    plot.set_ylabel("Size")
    # plt.savefig("A2b_boxplot.png")
    # plt.show()

    # normality test
    bottom_k2, bottom_p = stats.normaltest(bottom)
    top_k2, top_p = stats.normaltest(top)
    print("Results of the normality test:")
    print(f"p-value for bottom: {bottom_p}")
    print(f"p-value for top: {top_p}")

main()
