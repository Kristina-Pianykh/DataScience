import random
import sys
from pathlib import Path
from typing import Generic, List, Set, Tuple, TypeVar

USAGE_TEXT = """
The script was called with the wrong number of parameters.
The correct call syntax is:

    python permutationtest.py <sample-file1> <sample-file2> <alpha> <mode>
        <sample-file1> - input file with the values of the first sample
        <sample-file2> - input file with the values of the second sample
        <alpha> - significance level (in the range (0,1))
        <mode> - mode of calculation, exact (exact) or approximate (approx)

Example:

    python permutationtest.py sample1.txt sample2.txt 0.05 exact

        or

    python permutationtest.py sample1.txt sample2.txt 0.05 approx 10000

"""


TValue = TypeVar("TValue", bound=int)

class HashableSet(set, Generic[TValue]):
    def __hash__(self) -> int:
        """
        Calculate the hash by adding the weighted by idx sum for values
        as values are sorted in a set.
        """
        return sum(i + v for i, v in enumerate(self))


IdxSampleGroups = Tuple[Set[int], Set[int]]

def generate_idx_sample_groups(num_idxs: int, first_group_size: int, num_samples: int) -> List[IdxSampleGroups]:
    samples: Set[HashableSet[int]] = set()
    idxs = range(num_idxs)
    while len(samples) < num_samples:
        sample = random.sample(idxs, first_group_size)
        samples.add(HashableSet(sample))
    sample_groups: List[IdxSampleGroups] = []
    idxs_set = set(idxs)
    for sample in samples:
        other_sample = idxs_set.difference(sample)
        sample_groups.append((set(sample), other_sample))
    return sample_groups


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
    This function tests whether the expected values of the two samples
    (samples1 and samples2) differ significantly by testing all possible permutations.
    The (alternative) hypothesis assumes that the expected values of the two populations
    differ ((E(samples1) != E(samples2)).
    The null hypothesis, on the other hand, assumes that the samples come
    from populations with the same expected value (E(samples1) = E(samples2)).

    The function returns the difference in the expected values of the samples
    and the p-value of the test. The p-value quantifies the probability of observing
    the same or an even larger (absolute) difference in expected if the null hypothesis
    that the samples were drawn from populations with the same expected value is true.

    :param samples1: list of observed values from the 1st sample
    :param samples2: list of observed values from the 2nd sample
    :return: tuple consisting of the difference in the means and
    the p-value of the test: (mean-diff,p)
    """
    observed_mean_diff = get_mean(samples2) - get_mean(samples1)
    mean_diffs: List[float] = []
    both_samples = samples1 + samples2
    samples_ranks = [idx for idx in range(len(both_samples))] # needed in case of duplicate values

    ranked_combinations: List[List[int]] = n_choose_k(samples_ranks, int(len(both_samples) / 2))
    for sample1_indices in ranked_combinations:
        sample2_indices = set(samples_ranks).difference(sample1_indices)
        sample1: List[float] = get_elements_by_idx(set(sample1_indices), both_samples)
        sample2: List[float] = get_elements_by_idx(set(sample2_indices), both_samples)
        mean_diffs.append(get_mean(sample2) - get_mean(sample1))

    relevant_mean_diffs = [value for value in mean_diffs if abs(value) >= abs(observed_mean_diff)]
    p_value = float(len(relevant_mean_diffs) / len(mean_diffs))

    return (round(observed_mean_diff, 4), round(p_value, 4))


def run_approx_permutationtest(samples1: List[float], samples2: List[float], n: int) -> Tuple[float, float]:
    """
    This function tests whether the expected values of the two samples
    (samples1 and samples2) differ significantly by testing n random,
    duplicate-free permutations. The (alternative) hypothesis assumes
    that the expected values of the two populations are different
    ((E(samples1) != E(samples2)).
    The null hypothesis, on the other hand, assumes that the samples
    come from populations with the same expected value (E(samples1) = E(samples2)).

    The function returns the difference in the expected values of the
    samples and the p-value of the test. The p-value quantifies the probability
    of observing the same or an even larger (absolute) difference in expected
    if the null hypothesis that the samples were drawn from populations with
    the same expected value is true.

    :param samples1: list of observed values from the 1st sample
    :param samples2: list of observed values from the 2nd sample
    :param n: number of random, duplicate-free permutations to be formed
    :return: tuple consisting of the difference of the means and the p-value
    of the test: (mean-diff,p)
    """
    sample_size = len(samples1)
    both_samples: List[float] = samples1 + samples2
    observed_mean_diff = float(get_mean(samples2) - get_mean(samples1))
    sample_indices: List[IdxSampleGroups] = generate_idx_sample_groups(len(samples1) + len(samples2), sample_size, n)

    mean_diffs: List[float] = []
    for sample1_indices, sample2_indices in sample_indices:
        sample1: List[float] = get_elements_by_idx(sample1_indices, both_samples)
        sample2: List[float] = get_elements_by_idx(sample2_indices, both_samples)
        mean_diffs.append(get_mean(sample2) - get_mean(sample1))

    relevant_mean_diffs = [value for value in mean_diffs if abs(value) >= abs(observed_mean_diff)]
    p_value = float(len(relevant_mean_diffs) / len(mean_diffs))
    return (round(observed_mean_diff, 4), round(p_value, 4))


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
