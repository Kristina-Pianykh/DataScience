from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def read_values(input_file: Path) -> pd.DataFrame:
    data = pd.read_csv(
        input_file,
        sep=" ",
        names=["Number", "Length", "Left", "Right", "Bottom", "Top", "Diagonal"],
        dtype={
            "Number": int,
            "Length": float,
            "Left": float,
            "Right": float,
            "Bottom": float,
            "Top": float,
            "Diagonal": float,
        },
        skipinitialspace=True,
    )
    return data


def var_equality(series1: pd.DataFrame, series2: pd.DataFrame):
    _, p = stats.levene(series1, series2, center="mean")
    return p


def normality_test(data: pd.DataFrame, category: str):
    _, bottom_p = stats.normaltest(data["Bottom"])
    _, top_p = stats.normaltest(data["Top"])
    print(f"Results of the normality test for the {category} bank notes:")
    print(f"p-value for bottom: {bottom_p}")  # normality violated
    print(f"p-value for top: {top_p}\n")


def ks_test(series1: pd.Series, series2: pd.Series):
    result = stats.kstest(series1, series2, mode="auto", alternative="two-sided")
    print("Results of the kolmogorov-smirnov test:")
    print(f"test statistic: {result.statistic}")
    print(f"p-value: {result.pvalue}")
    return result


def wilc_test(data: pd.DataFrame, category: str) -> Tuple[float, float]:
    # Wilcoxon rank-sum test for independent samples with continuous distribution
    t, p = stats.ranksums(data["Bottom"], data["Top"], alternative="two-sided")
    print("Wilcoxon rank-sum test for independent samples with same dsitribution")
    print(
        "============================================================================"
    )
    print(f"{category} data")
    print(f"t statistic : {round(t, 4)}")
    print(f"p-value : {round(p, 4)}\n")
    return t, p


def auf2b_ii(data: pd.DataFrame):
    print("Aufgabe 2b ii)")
    true_money = data[:100]
    fake_money = data[100:]

    print("True money:")
    print(f"mean bottom: {true_money['Bottom'].mean()}")
    print(f"mean top: {true_money['Top'].mean()}\n")

    print("Fake money:")
    print(f"mean bottom: {fake_money['Bottom'].mean()}")
    print(f"mean top: {fake_money['Top'].mean()}\n")

    # boxplots
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, sharex=True)
    fig.suptitle(
        "Boxplot of the width for the bottom and the top of the true and forged bank notes"
    )
    axes[0].set_title("True")
    axes[1].set_title("Forged")

    sns.boxplot(ax=axes[0], data=true_money[["Bottom", "Top"]])

    sns.boxplot(ax=axes[1], data=fake_money[["Bottom", "Top"]])
    # plt.savefig("A2b_boxplots.png")
    # plt.show()

    normality_test(true_money, "true")
    normality_test(fake_money, "forged")

    print(
        f"Samples for true money have variance: p={var_equality(true_money['Bottom'], true_money['Top'])}"
    )  # null hypothesis: both samples have equal variance
    print(
        f"Samples for fake money have variance: p={var_equality(fake_money['Bottom'], fake_money['Top'])}"
    )  # null hypothesis: both samples have equal variance

    result_true = ks_test(true_money["Bottom"], true_money["Top"])
    result_fake = ks_test(fake_money["Bottom"], fake_money["Top"])


def auf2b_i(data: pd.DataFrame):
    print("Aufgabe 2b i)")
    print(f"mean bottom: {data['Bottom'].mean()}")
    print(f"mean top: {data['Top'].mean()}\n")

    # boxplot
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 1, sharex=True)
    fig.suptitle("Boxplot of the width for the bottom and the top of the bank notes")
    sns.boxplot(ax=axes, data=data[["Bottom", "Top"]])
    # plt.savefig("A2b_boxplot.png")
    # plt.show()

    normality_test(data, "all")

    print(
        f"Samples have variance p={var_equality(data['Bottom'], data['Top'])}"
    )  # null hypothesis: both samples have equal variance

    # two-tailed Kolmogorov-Smirnov Test
    result = ks_test(data["Bottom"], data["Top"])


def main():
    file = Path("banknote.dat")
    data = read_values(file)
    auf2b_ii(data)
    # auf2b_i(data)


main()
