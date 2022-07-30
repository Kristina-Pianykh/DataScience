import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
print(f"t statistic : {round(t, 4)}")
print(f"p-value : {round(p, 4)}")

# fig, axs = plt.subplots(1, 2)
# axs[0, 0].boxplot(B)

# plt.boxplot(B)
# fig = plt.figure()
# plt.boxplot(K, labels=["Control group K"])
# fig = plt.figure(figsize=(10.0, 7.0))
# plt.show()

data = pd.DataFrame({"Control group K": K, "Group B": B}, columns=["Reaction times", "Group"])
print(data)
sns.set_style("whitegrid")
sns.boxplot(data=data)
plt.show()
