from pathlib import Path
from random import sample

import numpy as np
from scipy import stats

from permutationtest import read_values


def main():
    input_file1 = Path("variante_a.txt")
    input_file2 = Path("variante_b.txt")
    gmbh_file = Path("gmbh_produkt.txt")
    konkurrent_file = Path("konkurrent.txt")
    samples1 = read_values(input_file1)
    samples2 = read_values(input_file2)
    gmbh = read_values(gmbh_file)
    konkurrent = read_values(konkurrent_file)

    both_samples = gmbh + konkurrent
    #print(len(np.random.permutation(both_samples)))
    rand1 = sample(both_samples, 80)
    #print(len(rand1))
    #print(len(set(rand1)))

    print(stats.ttest_ind(gmbh, konkurrent, permutations=20000))
    # while True:
    #     random_sample1 = set(np.random.permutation(both_samples))
    #     random_sample2 = set(np.random.permutation(both_samples))
    #     if len(random_sample1) == 80 and len(random_sample2) == 80:
    #         break
    
        
    #print(set(np.random.permutation(samples1+samples1)))
    #print(len(set(samples1 + samples2)))
    #print(factorial(160)/(factorial(80)*2))


main()
