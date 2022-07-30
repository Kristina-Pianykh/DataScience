from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# def read_values(input_file: Path) -> pd.DataFrame:
#     data = pd.read_csv(input_file, sep=" ")
#     return data

def read_values(input_file: Path) -> List[Any]:
    with input_file.open("r") as in_stream:
        values = [float(line.strip()) for line in in_stream.readlines() if line.strip()]

    return values
    
def main():
    file = Path("banknote.dat")
    # data = read_values(file)
    # print(data)
    data = pd.read_csv(file, sep=" ")
    # data.columns = ["Number", "Length", "Left", "Right", "Bottom", "Top", "Diagonal"]
    print(data.columns)
    print(data.shape)

main()
