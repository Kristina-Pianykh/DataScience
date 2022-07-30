from typing import Dict, List, Tuple, Union

import seaborn as sns

from analyze_sunshine_hours import (get_average_sunshine_hours_by_month,
                                    get_average_sunshine_hours_per_year,
                                    read_entries)

"""
Compute the months with the most average sunshine and least sunshine
"""
data: List[Tuple] = read_entries(
    "/home/kristina/Desktop/DataScience/DataScience/Uebungsblatt0/sunshine_changi.csv"
)
months: List[str] = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dez",
]
month_average_sunshine: Dict[str, float] = {month: 0.0 for month in months}
for month in months:
    month_average_sunshine[month] = get_average_sunshine_hours_by_month(data, month)

all_sunshine_hours: List[float] = [
    month_average_sunshine[month] for month in month_average_sunshine
]
max_sun: float = max(all_sunshine_hours)
min_sun: float = min(all_sunshine_hours)
max_min_months: List[str] = [
    key
    for key in month_average_sunshine
    if (month_average_sunshine[key] == max_sun)
    or ((month_average_sunshine[key] == min_sun))
]
print(f"The month with max sunshine is {max(max_min_months)} with {max_sun} hours")
print(f"The month with min sunshine is {min(max_min_months)} with {min_sun} hours")

"""
plot average sunshine hours against each year with using a lineplot
"""
all_months_aver_sun: Dict[int, float] = get_average_sunshine_hours_per_year(data)
sns.set_theme(style="darkgrid")
plot_data: Dict[str, List[Union[int, float]]] = {"year": [], "hours": []}
plot_data["year"] = [key for key in all_months_aver_sun]
plot_data["hours"] = [all_months_aver_sun[key] for key in all_months_aver_sun]


ax = sns.lineplot(x="year", y="hours", data=plot_data)
fig = ax.get_figure()
fig.savefig("/home/kristina/Desktop/DataScience/DataScience/Uebungsblatt0/mein_diagramm.png")
