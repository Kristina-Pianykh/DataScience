import random
import sys
from math import sqrt
from pathlib import Path
from typing import List, Tuple


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
    medians: List[float] = []

    for sample in range(n):
        new_sample = random.choices(samples, k=m)
        medians.append(get_median(new_sample))
    
    return medians

def get_median(sample: List[float]) -> float:
    sorted_sample = sorted(sample)
    median_idx = int(len(sorted_sample) / 2)
    
    return sorted_sample[median_idx]

def get_mean(lst: List[float]) -> float:
    accumulator = 0.0
    for item in lst:
        accumulator += item
    return accumulator / float(len(lst))

def get_stddev(lst: List[float]) -> float:
    accumulator = 0.0
    mean = get_mean(lst)

    for item in lst:
        accumulator += (item - mean)**2

    variance = accumulator / float(len(lst) - 1)
    return round(sqrt(variance), 4)

def get_standard_error(bootstrap_medians: List[float]) -> float:
    """
    Gibt den Stichprobenfehler / Standardfehler der übergebenen Liste von mittleren Elemente
    des Bootstrappings zurück.

    :param bootstrap_medians: Liste der berechneten Mediane aus dem Bootstrapping
    :return: Standardfehler der Bootstrapping-Mediane
    """
    return get_stddev(bootstrap_medians)

if __name__ == "__main__":
    #
    # Hier nichts ändern!
    #

    # Auslesen der Kommandozeilenparameter
    input_file = Path(sys.argv[1])
    n_samples = int(sys.argv[2])
    per_sample = int(sys.argv[3])
    data = read_samples(input_file)
    # print(data)

    medians = compute_bootstrapping_medians(data, n_samples, per_sample)
    print(get_standard_error(medians))
