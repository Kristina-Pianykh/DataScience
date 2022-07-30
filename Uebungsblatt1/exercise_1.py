from copy import deepcopy
from math import sqrt
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import scipy.stats as sp_stat
from scipy.stats import iqr

items = np.array([88, 92, 92, 86, 92, 85, 91, 90, 92, 90, 92])
print(mean(items))
print(median(items))
# print(sp_stat.norm.cdf(items))
print(sp_stat.norm.cdf(69, items.mean(), items.std(ddof=1)))
print(sp_stat.norm.ppf(0.95, sample.mean(), sample.std(ddof=1)))
