from math import isclose, pow
from typing import Callable, Union

import numpy as np
from scipy.misc import derivative

"""
This program applies Newton's algorithm to compute
zeros of the functions provided in A2.pdf


Output of the program:
Task a)

x=0.7757038518335664, f(x)=-0.5974227409938351
x=2.113381879603041, f(x)=-0.1498845586601809
x=2.7601754557021967, f(x)=-0.023659195572019343
x=2.9035336830002274, f(x)=-0.0006820833588314645
x=2.90790908441493, f(x)=-5.344782929572744e-07
x=2.907912518339108, f(x)=-3.2729374765949615e-13
x=2.907912518341211, f(x)=0.0

Finished in 7 iterations
y=0.0 for x=2.907912518341211

Task b)

x=2.421052631579649, f(x)=-0.027555037180278408
x=2.414262619005194, f(x)=-0.00019623673865210023
x=2.4142135649254266, f(x)=-1.0209325829180216e-08
x=2.4142135623730954, f(x)=-1.3322676295501878e-15
x=2.414213562373095, f(x)=-8.881784197001252e-16
x=2.414213562373095, f(x)=-8.881784197001252e-16

Finished in 6 iterations
y=-8.881784197001252e-16 for x=2.414213562373095
"""


def derivative_(func: Callable, x: float) -> float:
    return derivative(func, x, dx=1e-5)


def approximation(x: float, func: Callable):
    return x - func(x) / derivative_(func, x)


def func1(x: float) -> float:
    return np.sqrt(x + 1) + np.sin(x) / 10 - 2


def func2(x: float) -> float:
    return -pow(x, 3) + 3*pow(x, 2) - x - 1


def newton(start: Union[int, float], func: Callable):
    x = float(start)
    y = func(x)
    y_prev = -9  # random number not equal 0
    iteration = 0

    while (y != 0):
        if (y == y_prev):
            break
        x_prev = x
        x = approximation(x_prev, func)
        y_prev = y
        y = func(x)
        print(f"x={x}, f(x)={y}")
        iteration += 1


    print(f"\nFinished in {iteration} iterations")
    print(f"y={y} for x={x}")
    return x
    

def main():
    print("Task a)\n")
    start = 8
    func = func1
    nullstelle_x = newton(start, func)
    assert func(nullstelle_x) == 0

    print("\nTask b)\n")
    start = 2.5
    func = func2
    nullstelle_x = newton(start, func)
    assert isclose(func(nullstelle_x), 0, abs_tol=1e-15)


if __name__ == "__main__":
    main()
