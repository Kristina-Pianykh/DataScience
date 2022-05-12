# ---------------------------------------------------------------------------------------
# Abgabegruppe: 24
# Personen:
# - Kristina Pianykh, pianykhk, 617331
# - Miguel Nuno Carqueijeiro Athouguia, carqueim, 618203
# - Winston Winston, winstonw, 602307
# -------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import iqr

# Die Eingabedatei muss im gleichen Verzeichnis liegen wie die Skriptdatei exercise_2
input_file = Path("todesursachen.csv")
data: pd.DataFrame = pd.read_csv(input_file, sep=";", header=0)

def aufgabe_a1() -> int:
    """
    Ermitteln Sie die Anzahl an verschiedenen (“unique”) Todesursachen, die im Datensatz erfasst werden.

    :return: Anzahl der verschiedenen Todesursachen im Datensatz (als int)
    """
    unique_deaths = data["Todesursache"].unique().size
    return unique_deaths
    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_a2() -> Dict[str, Union[int, float]]:
    """
    Geben Sie den Durchschnitt (mean), den Median (med), die Standardabweichung (stddev), den minimalen Wert (min),
    den maximalen Wert (max), den Interquartilsabstand (iqr) und das 90%-Perzentil (90p) der Verstorbenen pro Jahr
    an.

    Die Rückgabe der Kennzahlen erfolgt als Dictionary mit folgendem Aufbau:
    {
        "mean": <Durchschnitt (float)>,
        "med": <Median (int)>,
        "stddev": <Standardabweichung (float)>,
        "min": <Minimaler-Wert (float)>,
        "max": <Maximaler-Wert (int)>,
        "iqr": <Interquartilsabstand (float)>,
        "90p": <90%-Perzentil (int)>
    }

    Die Kennzahlen sind entweder als float oder als int anzugeben (siehe Aufbaubeschreibung)

    :return: Dictionary mit den Kennzahlen
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

    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_a3() -> Dict[int, str]:
    """
    Ermitteln Sie die Anzahl der Kinder bzw. Heranwachsenden (< 15 Jahre), welche an einer Ursache verstorben sind,
    an welcher in dem jeweiligen Jahr auch mindestens 10 andere Kinder / Heranwachsende verstorben sind.
    Ermitteln Sie diese Kennzahl für jedes im Datensatz erfasste Jahr.

    Die Rückgabe erfolgt als Dictionary, welches ein Jahr (int) auf die Anzahl an verstorbenen Kindern (int)
    abbildet.

    {
        1980: 101234,
        1981: 12456,
        1982: 9876,
        ....

    }
    (Beispiel zeigt exemplarische, nicht-korrekte Werte!)

    :return: Dictionary, welches ein Jahr (int) auf die Anzahl an verstorbenen Kindern (int) abbildet.
    """
    kids_10 = data[
        ((data["Altersgruppe"] == "unter 1 Jahr") | (data["Altersgruppe"] == "1 bis unter 15 Jahre"))
        & (data["Anzahl"] >= 10)
    ]
    filtered = kids_10.groupby(["Jahr"]).sum().to_dict()["Anzahl"]
    dicti = {year: str(filtered[year]) for year in filtered}

    return dicti
    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_b1() -> int:
    """
    In welchem Jahr sind die meisten Männer durch Stürze ums Leben gekommen?

    Als Rückgabe wird ein int erwartet, welcher das jeweilige Jahr angibt.

    :return: Jahr (int) in dem am meisten Männer durch Stürze ums Leben gekommen sind
    """
    final_data = data[
        (data["Geschlecht"] == "männlich") & (data["Todesursache"] == "Stürze")
    ].groupby(["Jahr"]).sum()

    year: int = final_data[
        final_data["Anzahl"] == final_data["Anzahl"].max()
    ].reset_index()["Jahr"][0]
    
    return year
    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_b2() -> Tuple[float, float]:
    """
    Ertrinken im Durchschnitt mehr Kinder bzw. Heranwachsende (< 15 Jahre) oder mehr Menschen im
    Rentenalter (>= 65 Jahre) pro Jahr?

    Als Rückgabe wird ein Pair von float-Zahlen erwartet, welche die beiden Durchschnittswerte der jeweiligen
    Personengruppen angeben:

    (<Durchschnitt-Kinder (float)>, <Durchschnitt-Renter*innen (float)>)

    :return: (<Durchschnitt-Kinder (float)>, <Durchschnitt-Renter*innen (float)>)
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

    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_b3() -> str:
    """
    Welche Altersgruppe weist den größten Median hinsichtlich der Anzahl an Verstorbenen pro Jahr aus?

    Als Rückgabe wird die Beschreibung der Altersgruppe als string erwartet (bspw. "15 bis unter 20 Jahre")

    :return: Altersgruppe mit dem größten Median hinsichtlich der Anzahl an Verstorbenen pro Jahr
    """
    # age_groups_medians: pd.DataFrame = data.groupby(["Altersgruppe", "Jahr"]).median().reset_index()
    age_groups_medians = data.groupby(["Altersgruppe", "Jahr"]).sum("Anzahl").reset_index().groupby(["Altersgruppe"]).median()
    age_with_big_median = age_groups_medians[age_groups_medians["Anzahl"] == age_groups_medians["Anzahl"].max()].reset_index()["Altersgruppe"].values[0]

    return age_with_big_median
    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_b4() -> float:
    """
    In wieviel Prozent der erfassten Jahre sind mehr Frauen als Männer in einem Jahr gestorben?

    Als Rückgabe wird ein float zwischen 0 und 1 erwartet.

    :return: Anteil der Jahre in denen mehr Frauen als Männer verstorben sind (als float)
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

    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_b5() -> List[Tuple[float, float]]:
    """
    Welche Todesursachen weisen die größten Unterschiede zwischen Männern und Frauen im Alter von >= 20 Jahren und
    < 30 auf? Berechnen Sie hierzu die absoluten Differenz der Durchschnittswerte pro Jahr und Todesursache und
    geben Sie die drei Todesursachen mit den größten durchschnittlichen Differenzen an.

    Als Rückgabe wird eine drei-elementige Liste erwartet, welche die Todesursachen mit den größten Differenzen
    zwischen Männern und Frauen (absteigend geordnet) enthält. Jedes Element der Liste ist dabei ein Pair (2-Tupel)
    bestehend aus dem Namen der Todesursache und der durchschnittlichen Differenz. Beispiel:

    [
        ("BN des Magens", 245.45),
        ("Diabetes mellitus", 200.87),
        ("Krankheiten der Niere", 196.5)
    ]

    :return: Drei-elementige Liste mit den Todesursachen mit den größten Unterschieden zwischen Männern und Frauen.
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

    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_b6() -> List[Tuple[str, float]]:
    """
    Welche Todesursachen weisen die kleinsten Schwankungen hinsichtlich der Anzahl an Verstorbenen pro Jahr auf?
    Berechnen Sie hierzu die Standardabweichungen der einzelnen Todesursachen und setzen Sie diese in Relation
    zu deren jeweiligen Durchschnittswert. Geben Sie die drei Todesursachen mit den größten (relativen)
    Standardabweichungen an.

    Als Rückgabe wird eine drei-elementige Liste erwartet, welche die Todesursachen mit den kleinsten Schwankungen
    (absteigend geordnet) enthält. Jedes Element der Liste ist dabei ein Pair (2-Tupel) bestehend aus dem Namen
    der Todesursache und der (relativen) Standardabweichung. Beispiel:

    [
        ("BN des Magens", 0.123),
        ("Diabetes mellitus", 0.104),
        ("Krankheiten der Niere", 0.0965)
    ]

    :return: Drei-elementige Liste mit den Todesursachen mit den höchsten Schwankungen.
    """
    deaths_least_std: List[Tuple[str, float]] = []
    mean = data[["Todesursache", "Jahr", "Anzahl"]].groupby(["Todesursache", "Jahr"]).sum().reset_index().groupby("Todesursache").mean()
    stds = data[["Todesursache", "Jahr", "Anzahl"]].groupby(["Todesursache", "Jahr"]).sum().reset_index().groupby("Todesursache").std()
    mean["relative_std"] = stds["Anzahl"] / mean["Anzahl"]
    df = mean.sort_values(by="relative_std", ascending=1).head(3).reset_index()

    for entry in df.sort_values(by="relative_std", ascending=1).head(3).reset_index().itertuples():
        deaths_least_std.append((entry[2], round(entry[-1], 4)))

    return deaths_least_std

    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


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
