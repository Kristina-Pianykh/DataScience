import numpy as np

"""
This program computes the closed-form solution
to the given linear regression equation.


output of the program:
X.T * X =
[[ 5.  0.]
 [ 0. 10.]]

X.T * y =
[10. 10.]

(X.T * X)^(-1) =
[[0.2 0. ]
 [0.  0.1]]

W = (X.T * X)^(-1) * (X.T * y) =
[2. 1.]

a = 2.0, b = 1.0
"""

def compute_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.stack([np.ones(x.shape[0]), x], axis=1)
    a = np.matmul(X.T, X)
    multiplicand_1 = np.linalg.pinv(a)
    multiplicand_2 = np.matmul(X.transpose(), y)
    w = np.matmul(multiplicand_1, multiplicand_2) 
    print(f"X.T * X =\n{np.matmul(X.T, X)}\n")
    print(f"X.T * y =\n{multiplicand_2}\n")
    print(f"(X.T * X)^(-1) =\n{multiplicand_1}\n")
    print(f"W = (X.T * X)^(-1) * (X.T * y) =\n{w}\n")
    return w


def main():
    x = np.asarray([-2, -1, 0, 1, 2])
    y = np.asarray([2, 0, 0, 2, 6])
    w = compute_weights(x, y)
    print(f"a = {w[0]}, b = {w[1]}")


if __name__ == "__main__":
    main()
