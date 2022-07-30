from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# def read_values(input_file: Path) -> pd.DataFrame:
#     data = pd.read_csv(input_file, sep=" ")
#     return data

def main():
    file = Path("banknote.dat")
    data = pd.read_csv(input_file, sep=" ")
