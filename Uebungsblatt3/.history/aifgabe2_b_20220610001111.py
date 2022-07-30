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

def normality_tes(data: pd.DataFrame, class: str):
    _, bottom_p = stats.normaltest(data["Bottom"])
    _, top_p = stats.normaltest(data["Top"])
    print("Results of the normality test for the {class} bank notes:")
    print(f"p-value for bottom: {bottom_p}") # normality violated
    print(f"p-value for top: {top_p}\n")

    
def main():
    file = Path("banknote.dat")
    data = read_values(file)
    true_money = data[:100]
    fake_money = data[100:]

    print("True money:")
    print(f"mean bottom: {true_money['Bottom'].mean()}")
    print(f"mean top: {true_money['Top'].mean()}\n")

    print("Fake money:")
    print(f"mean bottom: {fake_money['Bottom'].mean()}")
    print(f"mean top: {fake_money['Top'].mean()}\n")

    #boxplots
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, sharex=True)
    fig.suptitle('Boxplot of the width for the bottom and the top of the true and forged bank notes')
    axes[0].set_title('True')
    axes[1].set_title('Forged')

    sns.boxplot(ax=axes[0], data=true_money[["Bottom", "Top"]])
    # fig.set_ylabel("Size")

    sns.boxplot(ax=axes[1], data=fake_money[["Bottom", "Top"]])
    # plt.set_ylabel("Size")
    plt.savefig("A2b_boxplot.png")
    plt.show()
    print(f"Outlier for the top: {min(top)}")
    # top has an outlier at 7.7

    # normality test
    _, bottom_p = stats.normaltest(bottom)
    _, top_p = stats.normaltest(top)
    print("Results of the normality test for the true bank notes:")
    print(f"p-value for bottom: {bottom_p}") # normality violated
    print(f"p-value for top: {top_p}\n")

    # Wilcoxon rank-sum test for independent samples with continuous distribution
    t, p = stats.ranksums(bottom, top, alternative="two-sided")
    print("ilcoxon rank-sum test for independent samples with same dsitribution")
    print(f"t statistic : {round(t, 4)}")
    print(f"p-value : {round(p, 4)}")

main()
