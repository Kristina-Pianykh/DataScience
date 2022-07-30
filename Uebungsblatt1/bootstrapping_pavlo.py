# ---------------------------------------------------------------------------------------
# Abgabegruppe: 26
# Personen: Wei Jin, Pavlo Myronov, Anna Savchenko
#
# -------------------------------------------------------------------------------------
import sys

from pathlib import Path
from typing import Tuple, List
import random
from math import sqrt, floor

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
    medians = []
    for i in range(n):
        stichprobe = [None] * m
        for j in range(m):
            idx = random.randint(0, m-1)
            stichprobe[j] = samples[idx]
        stichprobe.sort()
        medians.append(stichprobe[floor(m/2)])
    return medians


def get_standard_error(bootstrap_medians: List[float]) -> float:
    """
    Gibt den Stichprobenfehler / Standardfehler der übergebenen Liste von mittleren Elemente
    des Bootstrappings zurück.

    :param bootstrap_medians: Liste der berechneten Mediane aus dem Bootstrapping
    :return: Standardfehler der Bootstrapping-Mediane
    """
    mean = sum(bootstrap_medians) / len(bootstrap_medians)
    mean_differences = [(med - mean) * (med - mean) for med in bootstrap_medians]
    variance = sum(mean_differences) / (len(bootstrap_medians) - 1)
    return sqrt(variance)


def get_confidence_interval(bootstrap_medians: List[float], p: float) -> Tuple[float, float]:
    """
    Berechnet das p-% Konfidenzintervall der übergebenen Liste an Bootstrapping-Mediane.

    :param bootstrap_medians: Liste der berechneten Mediane aus dem Bootstrapping
    :param p: Zu ermittelndes Konfidenzintervall im Bereich (0,1)

    :return: Start- und Endwerte des Konfidenzintervalls (jeweils als float)
    """
    bootstrap_medians.sort()
    to_delete = round(((1-p) / 2) * len(bootstrap_medians))
    return tuple([bootstrap_medians[to_delete], bootstrap_medians[-(to_delete+1)]])


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
