from sys import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def read_values(input_file: Path) -> List[float]:
    with input_file.open("r") as in_stream:
        values = [float(line.strip()) for line in in_stream.readlines() if line.strip()]

    return values
