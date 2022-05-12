from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

input_file = Path("todesursachen.csv")
write_lineplot = Path("kids_deaths_per_year.png")
write_boxplot = Path("kids_deaths_per_year_boxplot.png")

# read data
data: pd.DataFrame = pd.read_csv(input_file, sep=";", header=0)
age_groups = ['unter 1 Jahr', '1 bis unter 15 Jahre']
df = data[data["Altersgruppe"].isin(age_groups)].groupby(["Jahr", "Geschlecht"]).sum().reset_index()

# plot number of deaths among girls and boys under 15 per year (line plot) 
sns.lineplot(x="Jahr", y="Anzahl", hue="Geschlecht", data=df)
plt.grid()
plt.savefig(write_lineplot)

plt.figure().clear()

# plot distribution of deaths of children under 15 per year (box plot) 
boxplot_df = df.groupby(["Jahr"]).sum().reset_index()
bp = sns.boxplot(x="Anzahl", data=boxplot_df)
plt.grid()
bp.set_xticks(range(min(boxplot_df["Anzahl"]), max(boxplot_df["Anzahl"]), 3000))
plt.savefig(write_boxplot)
