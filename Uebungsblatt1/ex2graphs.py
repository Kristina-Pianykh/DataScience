import matplotlib.pyplot as plt
import seaborn as sns

from exercise_2 import *

df = read_data()
kinder = ["unter 1 Jahr", "1 bis unter 15 Jahre"]
lineplot_df = df.loc[df['Altersgruppe'].isin(kinder)].groupby(['Jahr','Geschlecht']).sum('Anzahl').reset_index()
#print(lineplot_df)
sns.lineplot(x="Jahr", y="Anzahl", hue="Geschlecht", data=lineplot_df, ci=None)
plt.savefig("c1.png")
# plt.figure().clear()
# boxplot_df = lineplot_df.groupby('Jahr').sum('Anzahl').reset_index()
# bp = sns.boxplot(x=boxplot_df["Anzahl"])
# bp.set_xticks(range(min(boxplot_df["Anzahl"]), max(boxplot_df["Anzahl"]), 3000))
# plt.savefig("c2.png")
