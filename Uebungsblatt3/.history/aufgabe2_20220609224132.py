from os import stat

import numpy as np
from scipy import stats

B = np.array([0.61, 0.79, 0.83, 0.66, 0.94, 0.78, 0.81, 0.60, 0.88, 0.90, 0.75, 0.86])
K = np.array([0.70, 0.58, 0.64, 0.70, 0.69, 0.80, 0.71, 0.63, 0.82, 0.60, 0.91, 0.59])


B_k2, B_p = stats.normaltest(B)
K_k2, K_p = stats.normaltest(K)
print(f"p-value for B: {B_p}")
print(f"p-value for K: {K_p}")

alpha = 0.05
# if p < alpha:  # null hypothesis: x comes from a normal distribution
#     print("The null hypothesis can be rejected")
# else:
#     print("The null hypothesis cannot be rejected")

t, p = stats.ttest_ind(B, K, equal_var=False, alternative="greater")
print(f"t statistic : {t}")
print(f"p statistic : {p}")
