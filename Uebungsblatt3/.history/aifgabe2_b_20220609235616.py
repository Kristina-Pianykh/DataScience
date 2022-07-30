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
    print(f"Outlier for the top: {min(top)}")
    # top has an outlier at 7.7

    # normality test
    _, bottom_p = stats.normaltest(bottom)
    _, top_p = stats.normaltest(top)
    print("Results of the normality test:")
    print(f"p-value for bottom: {bottom_p}") # normality violated
    print(f"p-value for top: {top_p}")

    # Wilcoxon rank-sum test for independent samples with continuous distribution
    t, p = stats.ranksums(bottom, top, alternative="greater")
    print("ilcoxon rank-sum test for independent samples with same dsitribution")
    print(f"t statistic : {round(t, 4)}")
    print(f"p-value : {round(p, 4)}")

main()
