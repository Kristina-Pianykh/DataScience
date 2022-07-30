from pathlib import Path
from pickle import TRUE
from typing import Any, List, Tuple

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

def normality_test(data: pd.DataFrame, category: str):
    _, bottom_p = stats.normaltest(data["Bottom"])
    _, top_p = stats.normaltest(data["Top"])
    print(f"Results of the normality test for the {category} bank notes:")
    print(f"p-value for bottom: {bottom_p}") # normality violated
    print(f"p-value for top: {top_p}\n")

def wilc_test(data: pd.DataFrame, category: str) -> Tuple[float, float]:
    # Wilcoxon rank-sum test for independent samples with continuous distribution
    t, p = stats.ranksums(data["Bottom"], data["Top"], alternative="two-sided")
    print("Wilcoxon rank-sum test for independent samples with same dsitribution")
    print("============================================================================")
    print(f"{category} data")
    print(f"t statistic : {round(t, 4)}")
    print(f"p-value : {round(p, 4)}\n")
    return t, p
    
def main():
    file = Path("banknote.dat")
    data = read_values(file)
    auf2b_i(data)

def auf2b_i(data: pd.DataFrame):
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
    plt.savefig("A2b_boxplots.png")
    plt.show()
    # top has an outlier at 7.7

    # normality test
    normality_test(true_money, "true")
    normality_test(fake_money, "forged")

    # Wilcoxon rank-sum test for independent samples with continuous distribution
    true_t, true_p = wilc_test(true_money, "true")
    fake_t, fake_p = wilc_test(fake_money, "forged")

main()
