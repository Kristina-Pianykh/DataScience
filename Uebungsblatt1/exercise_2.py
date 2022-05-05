# ---------------------------------------------------------------------------------------
# Abgabegruppe: 24
# Personen:
#
# -------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import iqr

# Die Eingabedatei muss im gleichen Verzeichnis liegen wie die Skriptdatei exercise_2
input_file = Path("todesursachen.tsv")

# TODO: Hier das Einlesen der Daten implementieren! Auf die Daten kann dann in jeder Funktion zugegriffen werden.


def read_data(input_file: Path) -> pd.DataFrame:
    """
    Liest die gemessenen Sonnenscheindaten aus der Datei file in eine von Ihnen zu wählende Tuple-Datenstruktur ein.
    Das gewählte Tuple-Format definiert den Eingabedatentyp der folgenden Analysefunktionen.

    :param input_file: Pfad zur Eingabedatei
    :return: Alle Einträge aus der Eingabedatei als Liste von Tupeln
    """
    data = pd.read_csv(input_file, sep=";", header=0)
    return data

def aufgabe_a1() -> int:
    """
    Ermitteln Sie die Anzahl an verschiedenen (“unique”) Todesursachen, die im Datensatz erfasst werden.

    :return: Anzahl der verschiedenen Todesursachen im Datensatz (als int)
    """
    data: pd.DataFrame = read_data("todesursachen.csv")
    unique_deaths = data["Todesursache"].unique().size
    return unique_deaths
    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_a2() -> Dict[str, Dict[str, Union[int, float]]]:
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
    data: pd.DataFrame = read_data("todesursachen.csv")
    stats_to_each_year: Dict[str, Dict[str, Union[int, float]]] = {}

    deaths_per_year: Dict[int, NDArray] = {} # because nan is of type float in numpy
    unique_years: np.ndarray = data["Jahr"].unique()
    for year in unique_years:
        deaths_per_year[year] = data[(data["Jahr"] == year) & (data["Anzahl"] != 0)]["Anzahl"]

    for year in deaths_per_year:
        deaths: NDArray = deaths_per_year[year]
        str_year = str(year)
        stats_to_each_year[str_year] = {}

        stats_to_each_year[str_year]["mean"] = round(float(deaths.mean()), 3)
        stats_to_each_year[str_year]["median"] = int(deaths.median())
        stats_to_each_year[str_year]["stddev"] = round(deaths.std(ddof=1), 3)
        stats_to_each_year[str_year]["min"] = float(deaths.min())
        stats_to_each_year[str_year]["max"] = int(deaths.max())
        stats_to_each_year[str_year]["iqr"] = float(iqr(deaths))
        stats_to_each_year[str_year]["90p"] = int(np.percentile(deaths, 90))
    
    return stats_to_each_year

    

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
    data: pd.DataFrame = read_data("todesursachen.csv")
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
    data: pd.DataFrame = read_data("todesursachen.csv")
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
    data: pd.DataFrame = read_data("todesursachen.csv")
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
    drowned_kids = drowned_set[drowned_set["Altersgruppe"].isin(kids_age_groups)]["Anzahl"].sum()
    drowned_old_fucks = drowned_set[drowned_set["Altersgruppe"].isin(old_age_groups)]["Anzahl"].sum()
    return (float(drowned_kids), float(drowned_old_fucks))

    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_b3() -> str:
    """
    Welche Altersgruppe weist den größten Median hinsichtlich der Anzahl an Verstorbenen pro Jahr aus?

    Als Rückgabe wird die Beschreibung der Altersgruppe als string erwartet (bspw. "15 bis unter 20 Jahre")

    :return: Altersgruppe mit dem größten Median hinsichtlich der Anzahl an Verstorbenen pro Jahr
    """
    data: pd.DataFrame = read_data("todesursachen.csv")
    age_groups_medians: pd.DataFrame = data[["Altersgruppe", "Anzahl"]][(data["Anzahl"] != 0)].groupby(["Altersgruppe"]).median()
    age_groups_medians: pd.DataFrame = age_groups_medians.reset_index()
    age_with_big_median: str = age_groups_medians[age_groups_medians["Anzahl"] == age_groups_medians["Anzahl"].max()]["Altersgruppe"].values[0]

    return age_with_big_median
    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def aufgabe_b4() -> float:
    """
    In wieviel Prozent der erfassten Jahre sind mehr Frauen als Männer in einem Jahr gestorben?

    Als Rückgabe wird ein float zwischen 0 und 1 erwartet.

    :return: Anteil der Jahre in denen mehr Frauen als Männer verstorben sind (als float)
    """
    data: pd.DataFrame = read_data("todesursachen.csv")
    men_women_grouped_by_year = data.groupby(["Jahr", "Geschlecht"]).sum()
    total_years = men_women_grouped_by_year.size
    unique_years = data["Jahr"].unique()
    years_more_women_died = 0

    for year in unique_years:
        men_died: int = men_women_grouped_by_year.loc[1980].loc["männlich"].values[0]
        women_died: int = men_women_grouped_by_year.loc[1980].loc["weiblich"].values[0]
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
    data: pd.DataFrame = read_data("todesursachen.csv")
    age_groups = ['20 bis unter 25 Jahre', '25 bis unter 30 Jahre']
    av_sex_per_year_and_deathcause = data[(data["Altersgruppe"].isin(age_groups))].groupby(["Todesursache", "Jahr", "Geschlecht"]).mean()
    pivoted_df = av_sex_per_year_and_deathcause["Anzahl"].unstack(level=-1)
    pivoted_df["diff"] = abs(pivoted_df["männlich"] - pivoted_df["weiblich"])
    biggest_diffs: pd.Series = pivoted_df["diff"].sort_values(ascending=False)[:3]
    biggest_diffs_dict = biggest_diffs.to_dict()
    death_causes_with_largest_diffs = [(key[0], biggest_diffs_dict[key]) for key in biggest_diffs_dict]

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
    # for key in sorted(kennzahlen.keys()):
    #     print(f"\t\t{key}: {round(kennzahlen[key], 1)}")
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
