"""
This script bootstraps 1000 samples with 80 items per
sample from einkommen.txt 10 times.

Each time it calculates the standard error and the
75%-confidence interval of medians.

In the end it prints the mean values as well as standard deviation for
lower and upper bounds of the computed 75%-confidence intervals
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from numpy import array, mean, std

from bootstrapping import (compute_bootstrapping_medians,
                           get_confidence_interval, get_standard_error,
                           read_samples)


def main():
    input_file = Path("einkommen.txt")
    sample = read_samples(input_file)
    num_samples = 1000
    per_sample = 80
    confidence_int = 0.95

    bootstrap_10_analyses: Dict[int, Dict[str, Any]] = {}
    for i in range(10):
        bootstrap_10_analyses[i] = {}
        computed_medians = compute_bootstrapping_medians(sample, num_samples, per_sample)
        bootstrap_10_analyses[i]["standard error"] = get_standard_error(computed_medians)
        bootstrap_10_analyses[i]["confidence interval"] = get_confidence_interval(computed_medians, confidence_int)


    confidence_intervals: List[Tuple[float, float]] = [bootstrap_10_analyses[sample_num]["confidence interval"] for sample_num in bootstrap_10_analyses]
    start_points = array([interval[0] for interval in confidence_intervals])
    end_points = array([interval[1] for interval in confidence_intervals])
    mean_start = mean(start_points)
    mean_end = mean(end_points)
    stddev_start = round(std(start_points, ddof=1), 4)
    stddev_end = round(std(end_points, ddof=1), 4)

    print(f"""
    Means:
        lower bounds of the 75%-confidence intervals: {mean_start}
        upper bounds of the 75%-confidence intervals: {mean_end}

    Standard deviation:
        lower bounds of the 75%-confidence intervals: {stddev_start}
        upper bounds of the 75%-confidence intervals: {stddev_end}

    """)
   

main()
