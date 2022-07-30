from pathlib import Path
from pickle import TRUE
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def read_values(input_file: Path) -> pd.DataFrame:
    data = pd.read_csv(
        input_file,
        sep=" ",
        names=["Number", "Length", "Left", "Right", "Bottom", "Top", "Diagonal"],
        dtype={"Number": int, "Length": float, "Left": float, "Right": float, "Bottom": float, "Top": float, "Diagonal": float},
        skipinitialspace=True
    )
    return data

    
def main():
    file = Path("banknote.dat")
    data = read_values(file)
    bottom = data["Bottom"]
    top = data["Top"]

main()
