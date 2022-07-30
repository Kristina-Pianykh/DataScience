# ---------------------------------------------------------------------------------------
# Abgabegruppe: 24
# Personen:
# - Kristina Pianykh, pianykhk, 617331
# - Miguel Nuno Carqueijeiro Athouguia, carqueim, 618203
# - Winston Winston, winstonw, 602307
# -------------------------------------------------------------------------------------
import random
import sys
import typing
from pathlib import Path
from typing import List, Set, Tuple

USAGE_TEXT = """
Das Skript wurde mit der falschen Anzahl an Parametern aufgerufen.
Die korrekte Aufrufsyntax ist:

    python permutationtest.py <sample-datei1> <sample-datei2> <alpha> <modus>
        <sample-datei1> - Eingabedatei mit den Werten der ersten Stichprobe
        <sample-datei2> - Eingabedatei mit den Werten der zweiten Stichprobe
        <alpha>         - Signifikanzniveau (im Bereich (0,1))
        <modus>         - Modus der Berechnung, exakt (exact) oder approximativ (approx)

Beispiel:

    python permutationtest.py sample1.txt sample2.txt 0.05 exact
        
        oder
    
    python permutationtest.py sample1.txt sample2.txt 0.05 approx 10000
                        
"""

N = 80
SAMPLE_SIZE = 80
ALL_IDXS = range(2 * N)

SampleGroup = Tuple[Set[int], Set[int]]

TValue = typing.TypeVar("TValue", bound=int)


class HashableSet(set, typing.Generic[TValue]):
    def __hash__(self) -> int:
        """Calculate the hash by adding the weighted by idx sum for values
        as values are sorted in a set."""
        return sum(i + v for i, v in enumerate(self))


def generate_sample(all_idxs: range, sample_size: int) -> HashableSet[int]:
    sample = random.sample(all_idxs, sample_size)
    return HashableSet(sample)


def generate_sample_groups(num_samples: int, all_idx_set: Set[int]) -> List[SampleGroup]:
    samples: set[HashableSet[int]] = set()
    sample_size = int(len(all_idx_set) / 2)
    while len(samples) < num_samples:
        sample = generate_sample()
        samples.add(sample)
    sample_groups: list[SampleGroup] = []
    for sample in samples:
        other_sample = all_idx_set.difference(sample)
        sample_groups.append((set(sample), other_sample))
    return sample_groups


def test_sample_group(sample_group: SampleGroup, sample_size: int):
    sample, other_sample = sample_group
    all_elements = {*sample, *other_sample}
    assert len(all_elements) == 2 * sample_size


def get_mean(lst: List[float]) -> float:
    return float(sum(lst) / len(lst))



def read_values(input_file: Path) -> List[float]:
    with input_file.open("r") as in_stream:
        values = [float(line.strip()) for line in in_stream.readlines() if line.strip()]

    return values


def n_choose_k(lst: List[int], k: int) -> List[List[int]]:
     
    if k == 0:
        return [[]]
     
    combinations: List[List[int]] = []
    for i in range(0, len(lst)):
        m = lst[i]
        remaining = lst[0:i] + lst[i + 1:]
         
        remainlst_combo = n_choose_k(remaining, k-1)
        for pick in remainlst_combo:
            new_combination = sorted([m, *pick])
            if new_combination not in combinations:
                combinations.append(new_combination)
           
    return combinations


def get_elements_by_idx(idx_set: Set[int], lookup_lst: List[float]) -> List[float]:
    return [lookup_lst[idx] for idx in idx_set]



def run_exact_permutationtest(samples1: List[float], samples2: List[float]) -> Tuple[float, float]:
    """
    Diese Funktion testet, ob sich die Erwartungswerte der beiden Stichproben samples1 und samples2
    signifikant unterscheiden durch Prüfung aller möglichen Permutationen. Die (alternative) Hypothese
    nimmt an, dass sich die Erwartungswerte der beiden Populationen unterscheiden ((E(samples1) != E(samples2)).
    Die Null-Hypothese nimmt hingegen an, dass die Samples aus Populationen mit dem gleichen Erwartungswert stammen
    (E(samples1) = E(samples2)).

    Die Funktion gibt die Differenz der Erwartungswerte der Stichproben und den p-Wert des Tests zurück. Der p-Wert
    quantifiziert die Wahrscheinlichkeit, den gleichen oder einen noch größeren (absoluten) Unterschied der
    Erwartungswerte zu beobachten, wenn die Nullhypothese, dass die Stichproben aus Populationen mit demselben
    Erwartungswert gezogen wurden, wahr ist.

    :param samples1: Liste der beobachteten Werte aus der ersten Stichprobe
    :param samples2: Liste der beobachteten Werte aus der zweiten Stichprobe
    :return: Tuple bestehend aus der Differenz der Mittelwerte und dem p-Wert des Tests: (mean-diff,p)
    """
    sample_size = len(samples1)
    observed_mean_diff = get_mean(samples2) - get_mean(samples1)
    mean_diffs: List[float] = []
    both_samples = samples1 + samples2
    samples_ranks = [idx for idx in range(len(both_samples))] # needed in case of duplicate values

    ranked_combinations: List[List[int]] = n_choose_k(samples_ranks, int(len(both_samples) / 2))
    for sample1_indices in ranked_combinations:
        sample2_indices = set(samples_ranks).difference(sample1_indices)
        sample_group: SampleGroup = (set(sample1_indices), sample2_indices)
        test_sample_group(sample_group, sample_size)

        sample1: List[float] = get_elements_by_idx(set(sample1_indices), both_samples)
        sample2: List[float] = get_elements_by_idx(set(sample2_indices), both_samples)
        mean_diffs.append(get_mean(sample2) - get_mean(sample1))
    
    relevant_mean_diffs = [value for value in mean_diffs if abs(value) >= abs(observed_mean_diff)]
    p_value = float(len(relevant_mean_diffs) / len(mean_diffs))

    return (round(observed_mean_diff, 4), round(p_value, 4))
    raise NotImplementedError("ToDo: Funktsudo apt install fonts-firacodeion muss implementiert werden.")


def run_approx_permutationtest(samples1: List[float], samples2: List[float], n: int) -> Tuple[float, float]:
    """
    Diese Funktion testet, ob sich die Erwartungswerte der beiden Stichproben samples1 und samples2
    signifikant unterscheiden durch Prüfung von n zufälligen, duplikatfreien Permutationen. Die (alternative)
    Hypothese nimmt an, dass sich die Erwartungswerte der beiden Populationen unterscheiden
    ((E(samples1) != E(samples2)). Die Null-Hypothese nimmt hingegen an, dass die Samples aus Populationen
    mit dem gleichen Erwartungswert stammen (E(samples1) = E(samples2)).

    Die Funktion gibt die Differenz der Erwartungswerte der Stichproben und den p-Wert des Tests zurück. Der p-Wert
    quantifiziert die Wahrscheinlichkeit, den gleichen oder einen noch größeren (absoluten) Unterschied der
    Erwartungswerte zu beobachten, wenn die Nullhypothese, dass die Stichproben aus Populationen mit demselben
    Erwartungswert gezogen wurden, wahr ist.

    :param samples1: Liste der beobachteten Werte aus der ersten Stichprobe
    :param samples2: Liste der beobachteten Werte aus der zweiten Stichprobe
    :param n: Anzahl der zu bildenden zufälligen, duplikatfreien Permutationen
    :return: Tuple bestehend aus der Differenz der Mittelwerte und dem p-Wert des Tests: (mean-diff,p)
    """
    sample_size = len(samples1)
    both_samples: List[float] = samples1 + samples2
    all_idx_set: Set[int] = set(idx for idx in range(len(both_samples)))
    observed_mean_diff = float(get_mean(samples2) - get_mean(samples1))
    sample_indices: List[SampleGroup] = generate_sample_groups(n, all_idx_set)

    for sample_group in sample_indices:
        test_sample_group(sample_group, sample_size)

    mean_diffs: List[float] = []
    for sample1_indices, sample2_indices in sample_indices:
        sample1: List[float] = get_elements_by_idx(sample1_indices, both_samples)
        sample2: List[float] = get_elements_by_idx(sample2_indices, both_samples)
        mean_diffs.append(get_mean(sample2) - get_mean(sample1))

    relevant_mean_diffs = [value for value in mean_diffs if abs(value) >= abs(observed_mean_diff)]
    p_value = float(len(relevant_mean_diffs) / len(mean_diffs))
    # p_value = float(len([val for val in mean_diffs if abs(val) >= abs(observed_mean_diff)]) / len(mean_diffs))
    return (round(observed_mean_diff, 4), round(p_value, 4))
    raise NotImplementedError("ToDo: Funktion muss implementiert werden.")


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print(USAGE_TEXT)
        exit(-1)

    input_file1 = Path(sys.argv[1])
    if not input_file1.exists() or input_file1.is_dir():
        print(f"Eingabedatei {input_file1} ist nicht vorhanden")
        exit(-1)

    input_file2 = Path(sys.argv[2])
    if not input_file2.exists() or input_file2.is_dir():
        print(f"Eingabedatei {input_file2} ist nicht vorhanden")
        exit(-1)

    alpha = float(sys.argv[3])
    if not alpha > 0 and alpha < 1.0:
        print("Der Parameter alpha muss im Bereich (0, 1) liegen!")
        exit(-1)

    modus = sys.argv[4].lower()
    if modus not in ["exact", "approx"]:
        print("Der Parameter muss entweder 'exact' oder 'approx' sein")
        exit(-1)

    samples1 = read_values(input_file1)
    samples2 = read_values(input_file2)

    if modus == "exact":
        mean_diff, p_value = run_exact_permutationtest(samples1, samples2)
    elif modus == "approx":
        num_permutations = int(sys.argv[5])
        if num_permutations <= 0:
            print("Anzahl der zu generierenden Permutationen muss größer als 0 sein")
            exit(0)

        mean_diff, p_value = run_approx_permutationtest(samples1, samples2, num_permutations)
    else:
        raise AssertionError()

    print(f"Differenz zwischen den Mittelwerten: {mean_diff}")
    print(f"P-Wert (p={p_value})")
