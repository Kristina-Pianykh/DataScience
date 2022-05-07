# ---------------------------------------------------------------------------------------
# Abgabegruppe: 24
# Personen:
#
# -------------------------------------------------------------------------------------
import random
import sys
from math import ceil, floor, sqrt
from pathlib import Path
from typing import Any, Dict, List, Tuple

USAGE_TEXT = """
Das Skript wurde mit der falschen Anzahl an Parametern aufgerufen.
Die korrekte Aufrufsyntax ist:

    python bootstrapping.py <datei> <n> <m> <p>
        <datei> - Eingabedatei mit den Werten des Samples
        <n>     - Anzahl an Bootstrapping-Stichproben
        <m>     - Anzahl an Ziehungen pro Bootstrapping-Stichproben
        <p>     - Zu berechnendes Konfidenzintervall (im Bereich (0,1))

Beispiel:
  
    python bootstrapping.py einkommen.txt 1000 80 0.95                
"""


def read_samples(sample_file: Path) -> List[float]:
    """
    Liest die Sample-Werte aus der übergebenen Datei ein.

    :param sample_file: Einzulesende Datei mit den Sample-Werten
    :return: Liste der Sample-Werte (als float)
    """
    with sample_file.open("r") as in_stream:
        data = [float(line.strip()) for line in in_stream.readlines()]

    return data


def get_median(sample: List[float]) -> float:
    sorted_sample = sorted(sample)
    median_idx = int(len(sorted_sample) / 2)
    
    return sorted_sample[median_idx]


def get_mean(lst: List[float]) -> float:
    accumulator = 0.0
    for item in lst:
        accumulator += item
    return accumulator / float(len(lst))


# degrees of freedom: 1
def get_stddev(lst: List[float]) -> float:
    accumulator = 0.0
    mean = get_mean(lst)

    for item in lst:
        accumulator += (item - mean)**2

    variance = accumulator / float(len(lst) - 1)
    return round(sqrt(variance), 4)


def compute_bootstrapping_medians(samples: List[float], n: int, m: int) -> List[float]:
    """
    Bildet basierend auf dem übergebenen Werten eines Samples n Bootstrapping-Stichproben mit jeweils
    m Werten und berechnet jeweils das mittlere Element dieser Bootstrapping-Stichprobe. Die Funktion
    gibt die Liste der berechneten mittleren Elemente zurück.

    :param samples: Liste der Werte des ursprünglichen Examples
    :param n: Anzahl der zu bildenden Bootstrapping-Stichproben
    :param m: Anzahl der Elemente je Bootstrapping-Stichprobe

    :return: Liste mit den mittleren Elementen der n Bootstrapping-Stichproben
    """
    medians: List[float] = []

    for sample in range(n):
        new_sample = random.choices(samples, k=m)
        medians.append(get_median(new_sample))
    
    return medians
    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def get_standard_error(bootstrap_medians: List[float]) -> float:
    """
    Gibt den Stichprobenfehler / Standardfehler der übergebenen Liste von mittleren Elemente
    des Bootstrappings zurück.

    :param bootstrap_medians: Liste der berechneten Mediane aus dem Bootstrapping
    :return: Standardfehler der Bootstrapping-Mediane
    """
    return get_stddev(bootstrap_medians)
    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


def get_confidence_interval(bootstrap_medians: List[float], p: float) -> Tuple[float, float]:
    """
    Berechnet das p-% Konfidenzintervall der übergebenen Liste an Bootstrapping-Mediane.

    :param bootstrap_medians: Liste der berechneten Mediane aus dem Bootstrapping
    :param p: Zu ermittelndes Konfidenzintervall im Bereich (0,1)

    :return: Start- und Endwerte des Konfidenzintervalls (jeweils als float)
    """
    sorted_medians = sorted(bootstrap_medians)
    within_interval_fl: float = float(len(bootstrap_medians)) * p
    
    # round down to the next smaller integer if the decimal part >= 05
    # round down to the next bigger integer, otherwise
    big_decimal_part: bool = (within_interval_fl - floor(within_interval_fl)) >= 0.5
    if big_decimal_part:
        within_interval: int = ceil(within_interval_fl)
    else:
        within_interval: int = floor(within_interval_fl)

    # round down the number of items to remove from each side to the next smaller integer
    to_remove: int = floor((len(bootstrap_medians) - within_interval) / 2)
    lower_idx: int = 0 + to_remove
    upper_idx: int = (len(sorted_medians) - to_remove) - 1
    return (sorted_medians[lower_idx], sorted_medians[upper_idx])

    raise NotImplementedError("ToDo: Funktion muss noch implementiert werden!")


if __name__ == "__main__":
    #
    # Hier nichts ändern!
    #

    if len(sys.argv) != 5:
        print(USAGE_TEXT)
        exit(42)

    # Auslesen der Kommandozeilenparameter
    input_file = Path(sys.argv[1])
    n = int(sys.argv[2])
    m = int(sys.argv[3])
    p = float(sys.argv[4])

    # Prüfe, ob die Eingabedatei existiert und kein Verzeichnis ist
    if not input_file.exists() or input_file.is_dir():
        print(f"Die Eingabedatei {input_file} existiert nicht")

    # Einlesen der Werte des Samples
    sample = read_samples(input_file)

    # Berechne die Mediane der Bootstrapping Samples
    computed_medians = compute_bootstrapping_medians(sample, n, m)

    # Berechne den Stichprobenfehler
    print(f"Stichprobenfehler: {get_standard_error(computed_medians)}")

    # Berechne das Konfidenzintervall
    start, end = get_confidence_interval(computed_medians, p)
    print(f"Konfidenzintervall: {start} - {end}")
