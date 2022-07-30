# ---------------------------------------------------------------------------------------
# Abgabegruppe: 26
# Personen: Wei Jin, Pavlo Myronov, Anna Savchenko
#
# -------------------------------------------------------------------------------------
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns

# Die Eingabedatei muss im gleichen Verzeichnis liegen wie die Skriptdatei exercise_2
input_file = Path("todesursachen.csv")


# TODO: Hier das Einlesen der Daten implementieren! Auf die Daten kann dann in jeder Funktion zugegriffen werden.

def read_data() -> pd.DataFrame:
    global input_file
    return pd.read_csv(input_file, sep=";")

def aufgabe_a1() -> int:
    """
    Ermitteln Sie die Anzahl an verschiedenen (“unique”) Todesursachen, die im Datensatz erfasst werden.

    :return: Anzahl der verschiedenen Todesursachen im Datensatz (als int)
    """
    df = read_data()
    return len(set(df['Todesursache']))


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
    df = read_data()
    todesursache = df.groupby("Jahr").sum("Anzahl")['Anzahl']
    ret_dict = {}
    ret_dict['mean'] = np.mean(todesursache)
    ret_dict['median'] = round(np.median(todesursache))
    ret_dict['stddev'] = np.std(todesursache, ddof=1)
    ret_dict['min'] = round(np.min(todesursache))
    ret_dict['max'] = np.max(todesursache)
    ret_dict['iqr'] = np.percentile(todesursache, 75) - np.percentile(todesursache, 25)
    ret_dict['90p'] = round(np.percentile(todesursache, 90))
    return ret_dict


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
    altersgruppe = ["unter 1 Jahr", "1 bis unter 15 Jahre"]
    df = read_data()
    df = df.loc[np.isin(df['Altersgruppe'], altersgruppe)]
    df = df.loc[df['Anzahl'] >= 10]
    df = df.groupby('Jahr').sum("Anzahl")
    ret_dict = {year: value for (year, value) in df.itertuples()}
    return ret_dict


def aufgabe_b1() -> int:
    """
    In welchem Jahr sind die meisten Männer durch Stürze ums Leben gekommen?

    Als Rückgabe wird ein int erwartet, welcher das jeweilige Jahr angibt.

    :return: Jahr (int) in dem am meisten Männer durch Stürze ums Leben gekommen sind
    """
    df = read_data()
    df = df.loc[(df['Geschlecht'] == "männlich") & (df['Todesursache'] == "Stürze")]
    df = df.sort_values(by='Anzahl', ascending=0)
    return df['Jahr'].values[0]


def aufgabe_b2() -> Tuple[float, float]:
    """
    Ertrinken im Durchschnitt mehr Kinder bzw. Heranwachsende (< 15 Jahre) oder mehr Menschen im
    Rentenalter (>= 65 Jahre) pro Jahr?

    Als Rückgabe wird ein Pair von float-Zahlen erwartet, welche die beiden Durchschnittswerte der jeweiligen
    Personengruppen angeben:

    (<Durchschnitt-Kinder (float)>, <Durchschnitt-Renter*innen (float)>)

    :return: (<Durchschnitt-Kinder (float)>, <Durchschnitt-Renter*innen (float)>)
    """
    df = read_data()
    kinder = ["unter 1 Jahr", "1 bis unter 15 Jahre"]
    renter = ["65 bis unter 70 Jahre", "70 bis unter 75 Jahre",
              "75 bis unter 80 Jahre", "80 bis unter 85 Jahre",
              "85 Jahre und mehr"]
    df = df.loc[df['Todesursache'] == 'Ertrinken und Untergehen']
    kinder_df = df.loc[np.isin(df['Altersgruppe'], kinder)].groupby(['Jahr']).sum('Anzahl')
    renter_df = df.loc[np.isin(df['Altersgruppe'], renter)].groupby(['Jahr']).sum('Anzahl')
    return (np.mean(kinder_df['Anzahl']), np.mean(renter_df['Anzahl']))


def aufgabe_b3() -> str:
    """
    Welche Altersgruppe weist den größten Median hinsichtlich der Anzahl an Verstorbenen pro Jahr aus?

    Als Rückgabe wird die Beschreibung der Altersgruppe als string erwartet (bspw. "15 bis unter 20 Jahre")

    :return: Altersgruppe mit dem größten Median hinsichtlich der Anzahl an Verstorbenen pro Jahr
    """
    df = read_data()
    df = df.groupby(['Altersgruppe', 'Jahr']).sum('Anzahl') \
        .groupby('Altersgruppe')[['Anzahl']].apply(np.median).sort_values(ascending=False)
    print(df)
    return df.index[0]


def aufgabe_b4() -> float:
    """
    In wieviel Prozent der erfassten Jahre sind mehr Frauen als Männer in einem Jahr gestorben?

    Als Rückgabe wird ein float zwischen 0 und 1 erwartet.

    :return: Anteil der Jahre in denen mehr Frauen als Männer verstorben sind (als float)
    """
    df = read_data()
    df = df.groupby(['Jahr', 'Geschlecht']).sum('Anzahl').reset_index()
    df = df.pivot(index="Jahr", columns='Geschlecht', values='Anzahl').reset_index()
    df['mehr_frauen'] = df['männlich'] < df['weiblich']
    return np.mean(df['mehr_frauen'])


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
    df = read_data()
    df = df.groupby(['Jahr', 'Geschlecht',  'Todesursache']).sum('Anzahl').reset_index()
    df = df.filter(['Geschlecht',  'Todesursache', 'Anzahl'])\
        .groupby(['Geschlecht',  'Todesursache']).apply( np.mean).reset_index()
    df = df.pivot(index='Todesursache', columns='Geschlecht', values='Anzahl').reset_index()
    df['difference'] = df['männlich'] - df['weiblich']
    df = df.sort_values(by="difference", ascending=False).filter(['Todesursache', 'difference']).head(3)
    tuples_list = [(s, round(t, 2)) for (f, s, t) in df.itertuples()]
    return tuples_list


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
    df = read_data()
    df = df.groupby(['Jahr', 'Todesursache']).sum('Anzahl').reset_index()
    df = df.filter(['Todesursache', 'Anzahl']).groupby(['Todesursache'])
    df = df[['Anzahl']].apply(lambda x: np.std(x, ddof=1)/(np.mean(x))).reset_index()
    df = df.sort_values(by="Anzahl", ascending=1).head(3)
    tuples_list = [(s, t) for (f, s, t) in df.itertuples()]
    return tuples_list


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
