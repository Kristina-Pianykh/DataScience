from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import iqr

# the input files should be placed in the same folder as the script exercise_2
input_file = Path("todesursachen.csv")
data: pd.DataFrame = pd.read_csv(input_file, sep=";", header=0)

def aufgabe_a1() -> int:
    """
    Determines the number of different ("unique") causes of death
    that are recorded in the data set.

    :return: number of different causes of death in the data set (as int)
    """
    unique_deaths = data["Todesursache"].unique().size
    return unique_deaths


def aufgabe_a2() -> Dict[str, Union[int, float]]:
    """
    Gives the mean, the median, the standard deviation, the minimum value,
    the maximum value, the interquartile range, and the 90% percentile
    of deceased per year.

    The return of the ratios is done as a dictionary with the following structure:
    {
        "mean": <average (float)>,
        "med": <median (int)>,
        "stddev": <standard deviation (float)>,
        "min": <minimum value (float)>,
        "max": <maximum value (int)>,
        "iqr": <interquartile range (float)>,
        "90p": <90%percentile (int)>.
    }

    The values in the dictionary are to be specified either as float or as int.

    :return: Dictionary with the descriptive statistics
    """
    death_stats: Dict[str, Union[int, float]] = {}
    deaths = data.groupby(["Jahr"]).sum()["Anzahl"]

    death_stats["mean"] = round(float(np.mean(deaths)), 4)
    death_stats["median"] = int(np.median(deaths))
    death_stats["stddev"] = round(float(np.std(deaths, ddof=1)), 4)
    death_stats["min"] = float(deaths.min())
    death_stats["max"] = int(deaths.max())
    death_stats["iqr"] = round(float(iqr(deaths)), 4)
    death_stats["90p"] = int(np.percentile(deaths, 90))

    return death_stats


def aufgabe_a3() -> Dict[int, str]:
    """
    Determines the number of children or adolescents (< 15 years) who died from one cause,
    from which at least 10 other children/adolescents also died in that year.
    Determines this ratio for each year covered in the data set.

    The return is a dictionary that maps a year (int) to the number of children (int) who died.
  
    {
        1980: 101234,
        1981: 12456,
        1982: 9876,
        ....

    }
    (example shows exemplary values!)

    :return: dictionary with a year as the key (int)
    and the number of deceased children as the value (int).
    """
    kids_10 = data[
        ((data["Altersgruppe"] == "unter 1 Jahr") | (data["Altersgruppe"] == "1 bis unter 15 Jahre"))
        & (data["Anzahl"] >= 10)
    ]
    filtered = kids_10.groupby(["Jahr"]).sum().to_dict()["Anzahl"]
    dicti = {year: str(filtered[year]) for year in filtered}

    return dicti


def aufgabe_b1() -> int:
    """
    In which year did the most men die from falling?
    An int is expected as a return value, which specifies the respective year.

    :return: year (int) in which most men died from falling
    """
    final_data = data[
        (data["Geschlecht"] == "männlich") & (data["Todesursache"] == "Stürze")
    ].groupby(["Jahr"]).sum()

    year: int = final_data[
        final_data["Anzahl"] == final_data["Anzahl"].max()
    ].reset_index()["Jahr"][0]
 
    return year


def aufgabe_b2() -> Tuple[float, float]:
    """
    Do more children and adolescents (< 15 years) or more people of retirement age (>= 65 years)
    drown each year, on average?

    The expected return is a pair of float numbers that indicate
    the two means of the respective groups.

    :return: (<average-children (float)>, <average-pensioners (float)>)
    """
    kids_age_groups = [
        "unter 1 Jahr",
        "1 bis unter 15 Jahre"
    ]
    old_age_groups = [
        "65 bis unter 70 Jahre",
        "70 bis unter 75 Jahre",
        "75 bis unter 80 Jahre",
        "80 bis unter 85 Jahre",
        "85 Jahre und mehr"
    ]
    drowned_set = data[(data["Todesursache"] == "Ertrinken und Untergehen")]
    drowned_kids = drowned_set[drowned_set["Altersgruppe"].isin(kids_age_groups)].groupby(["Jahr"]).sum()["Anzahl"]
    drowned_old_fucks = drowned_set[drowned_set["Altersgruppe"].isin(old_age_groups)].groupby(["Jahr"]).sum()["Anzahl"]
    return (round(np.mean(drowned_kids), 4), round(np.mean(drowned_old_fucks), 4))


def aufgabe_b3() -> str:
    """
    Which age group has the largest median of deaths per year?

    :return: age group with the largest median number of deceased per year (str).
    """
    age_groups_medians = data.groupby(["Altersgruppe", "Jahr"]).sum("Anzahl").reset_index().groupby(["Altersgruppe"]).median()
    age_with_big_median = age_groups_medians[age_groups_medians["Anzahl"] == age_groups_medians["Anzahl"].max()].reset_index()["Altersgruppe"].values[0]

    return age_with_big_median


def aufgabe_b4() -> float:
    """
    How many more women died then men per year percentagewise?
    The expected return is a float between 0 and 1.

    :return: percentage of years in which more women than men died (as float).
    """
    men_women_grouped_by_year = data.groupby(["Jahr", "Geschlecht"]).sum()
    unique_years = data["Jahr"].unique()
    total_years = len(unique_years)
    years_more_women_died = 0

    for year in unique_years:
        men_died: int = men_women_grouped_by_year.loc[year].loc["männlich"].values[0]
        women_died: int = men_women_grouped_by_year.loc[year].loc["weiblich"].values[0]
        if women_died > men_died:
            years_more_women_died += 1

    return float(years_more_women_died / total_years)


def aufgabe_b5() -> List[Tuple[float, float]]:
    """
    Which causes of death show the greatest differences between men and women
    aged >= 20 years and < 30? Calculate the absolute difference of the average
    values per year and cause of death and give three causes of death with
    the largest average differences.

    The expected return is a three-item list showing the causes of death with
    the largest differences between men and women (in descending order).
    Every element of the list is a pair (2-tuple) consisting of the name
    of the cause of death and the average difference. Example:

    [
        ("BN of stomach," 245.45),
        ("Diabetes mellitus," 200.87),
        [ "Diseases of the kidney," 196.5]
    ]

    :return: three-element list of causes of death with the largest differences between men and women.
    """
    death_causes_with_largest_diffs: List[Tuple[float, float]] = []
    age_groups = ['20 bis unter 25 Jahre', '25 bis unter 30 Jahre']
    df = data[(data["Altersgruppe"].isin(age_groups))].groupby(["Todesursache", "Jahr", "Geschlecht"]).sum().groupby(["Todesursache", "Geschlecht"]).mean()
    pivoted_df = df["Anzahl"].unstack(level=-1)
    pivoted_df["diff"] = abs(pivoted_df["männlich"] - pivoted_df["weiblich"])
    pivoted_df = pivoted_df.sort_values(by="diff", ascending=False).head(3)

    for row in pivoted_df.itertuples():
        death_causes_with_largest_diffs.append((row[0], row[-1]))
    return death_causes_with_largest_diffs


def aufgabe_b6() -> List[Tuple[str, float]]:
    """
    Which causes of death show the smallest fluctuations in terms of
    the number of deaths per year? Calculate the standard deviations
    of the individual causes of death and relate them to their respective means.
    Give three causes of death with the largest (relative) standard deviations.

    The expected return is a three-element list containing the causes of death
    with the smallest variations (in descending order). Every element of the list
    is a pair (2-tuple) consisting of the name of the death cause and the (relative)
    standard deviation. Example:

    [
        ("BN of stomach," 0.123),
        ("Diabetes mellitus," 0.104),
        ("diseases of the kidney," 0.0965)
    ]

    :return: three-element list of death causes with the highest variation
    """
    deaths_least_std: List[Tuple[str, float]] = []
    mean = data[["Todesursache", "Jahr", "Anzahl"]].groupby(["Todesursache", "Jahr"]).sum().reset_index().groupby("Todesursache").mean()
    stds = data[["Todesursache", "Jahr", "Anzahl"]].groupby(["Todesursache", "Jahr"]).sum().reset_index().groupby("Todesursache").std()
    mean["relative_std"] = stds["Anzahl"] / mean["Anzahl"]
    df = mean.sort_values(by="relative_std", ascending=1).head(3).reset_index()

    for entry in df.sort_values(by="relative_std", ascending=1).head(3).reset_index().itertuples():
        deaths_least_std.append((entry[2], round(entry[-1], 4)))

    return deaths_least_std


if __name__ == "__main__":
    #
    # Hier nichts verändern!
    #
    print("Lösungen für a)")

    print("\t1: Anzahl an verschiedenen (“unique”) Todesursachen:")
    print(f"\t\t{aufgabe_a1()}\n")

    print("\t2: Kennzahlen der Verstorbenen pro Jahr:")
    kennzahlen = aufgabe_a2()
    for key in sorted(kennzahlen.keys()):
        print(f"\t\t{key}: {round(kennzahlen[key], 1)}")
    print()

    print("\t3: Anzahl der verstorbenen Kinder bzw. Heranwachsenden (< 15 Jahre) pro Jahr:")
    jahr_zu_todesursache = aufgabe_a3()
    for jahr in sorted(jahr_zu_todesursache.keys()):
        print(f"\t\t{jahr}: {jahr_zu_todesursache[jahr]}")
    print()

    # ----

    print("Lösungen für b)")

    print("\t1: In welchem Jahr sind die meisten Männer durch Stürze ums Leben gekommen?")
    print(f"\t\t{aufgabe_b1()}\n")

    print("\t2: Ertrinken im Durchschnitt mehr Kinder bzw. Heranwachsende (< 15 Jahre) oder mehr "
          "Menschen im Rentenalter (>= 65 Jahre) pro Jahr?")
    ergebnis_b2 = aufgabe_b2()
    print(f"\t\tDurchschnitt Kinder: {round(ergebnis_b2[0], 2)}")
    print(f"\t\tDurchschnitt Rentner:innen: {round(ergebnis_b2[1], 2)}\n")

    print("\t3: Welche Altersgruppe weist den größten Median hinsichtlich der Anzahl an Verstorbenen "
          "pro Jahr aus?")
    print(f"\t\t{aufgabe_b3()}\n")

    print("\t4: In wieviel Prozent der erfassten Jahre sind mehr Frauen als Männer in einem Jahr gestorben?")
    print(f"\t\t{round(aufgabe_b4(), 4) * 100}%\n")

    print("\t5:Welche Todesursachen weisen die größten Unterschiede zwischen Männern und Frauen auf?")
    todesursachen_b5 = aufgabe_b5()
    for i, (todesursache, differenz) in enumerate(todesursachen_b5):
        print(f"\t\t{i}: {todesursache} ({round(differenz, 2)})")
    print()

    print("\t6:Welche Todesursachen weisen die kleinsten Schwankungen hinsichtlich der Anzahl an "
          "Verstorbenen pro Jahr auf?")
    todesursachen_b6 = aufgabe_b6()
    for i, (todesursache, rel_std_abw) in enumerate(todesursachen_b6):
        print(f"\t\t{i}: {todesursache} ({rel_std_abw})")
    print()
