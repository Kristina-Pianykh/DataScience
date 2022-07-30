import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def main():
    B = np.array(
        [0.61, 0.79, 0.83, 0.66, 0.94, 0.78, 0.81, 0.60, 0.88, 0.90, 0.75, 0.86]
    )
    K = np.array(
        [0.70, 0.58, 0.64, 0.70, 0.69, 0.80, 0.71, 0.63, 0.82, 0.60, 0.91, 0.59]
    )

    _, B_p = stats.normaltest(B)
    _, K_p = stats.normaltest(K)
    print(f"p-value for B: {B_p}")
    print(f"p-value for K: {K_p}")

    alpha = 0.05
    # if p < alpha:  # null hypothesis: x comes from a normal distribution
    #     print("The null hypothesis can be rejected")
    # else:
    #     print("The null hypothesis cannot be rejected")

    """
    Wilcoxon signed-rank test for independent samples with continuous distribution
    """
    t, p = stats.wilcoxon(B, K, alternative="greater")
    print("Wilcoxon rank-sum test for independent samples with same dsitribution")
    print(f"t statistic : {round(t, 4)}")
    print(f"p-value : {round(p, 4)}")

    data = pd.DataFrame({"Group K": K, "Group B": B})
    sns.set_style("whitegrid")
    plot = sns.boxplot(data=data)
    plot.set_ylabel("Reaction time in seconds")
    plt.savefig("A2_boxplot.png")
    plt.show()


main()
