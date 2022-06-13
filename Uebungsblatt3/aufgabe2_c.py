from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

NULL_MEAN = 500
NULL_MEAN1 = 55


def read_values(input_file: Path) -> pd.DataFrame:
    data = pd.read_csv(
        input_file,
        sep=" ",
        names=["ID", "Clinic-ID", "Duration", "Jail", "Dose"],
        dtype={"ID": int, "Clinic-ID": int, "Duration": int, "Jail": int, "Dose": int},
        skipinitialspace=True,
    )
    return data


def expected_val(series: pd.Series):
    freq = series.value_counts()
    freq_dict = freq.to_dict()
    sum = 0
    for i in freq_dict:
        probability = freq_dict[i] / len(freq_dict)
        sum += probability * i
    return sum


def var_equality(series1: pd.Series, series2: pd.Series):
    _, p = stats.levene(series1, series2, center="mean")
    return p


def normality_test(data: pd.Series, category: str):
    _, p = stats.normaltest(data)
    print(f"Results of the normality test for {category}:")
    print(f"p-value: {p}\n")


def wilcoxon(series: pd.DataFrame, alternative: str):
    mean_vctorized = np.array(len(series) * [NULL_MEAN])
    diff = series.to_numpy() - mean_vctorized
    result = stats.wilcoxon(diff, alternative=alternative)
    return result


def t_test(series: pd.DataFrame, alternative: str):
    test = stats.ttest_1samp(series, popmean=NULL_MEAN1, alternative=alternative)
    return test


def boxplot(series1: pd.Series, series2: pd.Series, boxplot_name: str):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, sharex=True)
    fig.suptitle("Boxplot of the width for the duration of stay in clinic 1 and 2")
    axes[0].set_title("Clinic 1")
    axes[1].set_title("Clinic 2")

    sns.boxplot(ax=axes[0], data=series1.to_frame())
    sns.boxplot(ax=axes[1], data=series2.to_frame())
    # plt.savefig(boxplot_name)
    # plt.show()


def aufgabe(
    duration_1: pd.Series,
    duration_2: pd.Series,
    alternative: str,
    category: str,
    test: Callable,
    boxplot_name: str,
):

    print(f"Mean duration of stay in {category} 1: {duration_1.mean()}")
    print(f"Mean duration of stay in {category} 2: {duration_2.mean()}\n")

    normality_test(duration_1, f"{category} 1")  # non-normal distribution
    normality_test(duration_2, f"{category} 2")  # non-normal distribution

    boxplot(duration_1, duration_2, boxplot_name)

    # print(f"The outlier in the sample for Cinic 2: {max(duration_2)}\n")

    test1 = test(duration_1, alternative)
    print(
        f"Results of the non-parametric one-tailed Wilcoxon signed-rank test for {category} 1:"
    )
    print(f"statisic={test1.statistic}")
    print(f"p-value={test1.pvalue}\n")

    test2 = wilcoxon(duration_2, alternative)
    print(
        f"Results of the non-parametric one-tailed Wilcoxon signed-rank test for {category} 2:"
    )
    print(f"statisic={test2.statistic}")
    print(f"p-value={test2.pvalue}\n")


def aufgabe_ii(series: pd.Series, alternative: str, category: str, test: Callable):

    print(f"Mean duration of stay in {category}: {series.mean()}\n")

    normality_test(series, f"{category}")  # non-normal distribution

    sns.set_style("whitegrid")
    plot = sns.boxplot(data=series.to_frame())
    # plot.set_ylabel("Reaction time in seconds")
    # plt.savefig("A3cii_boxplot.png")
    # plt.show()

    # print(f"The outlier in the sample for Cinic 2: {max(duration_2)}\n")

    test = test(series, alternative)
    print(
        f"Results of the non-parametric one-tailed Wilcoxon signed-rank test for {category}:"
    )
    print(f"statisic={test.statistic}")
    print(f"p-value={test.pvalue}\n")

    # test2 = wilcoxon(duration_2, alternative)
    # print(
    #     f"Results of the non-parametric one-tailed Wilcoxon signed-rank test for {category} 2:"
    # )
    # print(f"statisic={test2.statistic}")
    # print(f"p-value={test2.pvalue}\n")


def main():
    file = Path("heroin.dat")
    data = read_values(file)

    duration_1 = data[data["Clinic-ID"] == 1]["Duration"]
    duration_2 = data[data["Clinic-ID"] == 2]["Duration"]

    dose = data["Dose"]

    # aufgabe i)
    # aufgabe(duration_1, duration_2, "less", "Clinic", wilcoxon, "A2ci_boxplots.png")

    # aufgabe ii)
    aufgabe_ii(dose, "two-sided", "Dose", t_test)


main()
