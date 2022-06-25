from sklearn.metrics import accuracy_score
from scipy.special import softmax
import numpy as np

labels = {"sport": 1, "politics": 2, "society": 3}

y_pred = [2, 3, 1, 2, 2, 3, 1, 3, 1, 2, 1, 3, 2, 2, 2, 2]
y_true = [2, 2, 1, 2, 3, 2, 1, 3, 3, 2, 1, 1, 2, 2, 1, 2]

print(f"Accuracy score: {accuracy_score(y_true, y_pred)}")


W = np.matrix("1 2 0.5 4 2; 2 4 1 2 1")
print(f"gewichtematrix:\n{W}")
print(f"or:\n{W.T}\n")
x1 = np.array([2, -2, 4, 0, 1])
x2 = np.array([-3, 2, 2, -1, 3])
b = np.array([1, 0.5])
y1 = np.dot(W, x1) + b
y2 = np.dot(W, x2) + b
print(f"Eingabe x1: {y1}")
print(f"Eingabe x1: {y2}\n")


z1 = softmax(y1)
z2 = softmax(y2)
print(f"Aktivierungszustände für y1 gegeben x1: {z1}")
print(f"Aktivierungszustände für y2 gegeben x2: {z2}")


"""
The input x1 is classified to belong to class c1 with the probability 0.817574 and to c2 with the probability 0.182426
==> the input x1 is classified as c1

The input x2 is classified to belong to class c1 with the probability 0.377540 and to c2 with the probability 0.622459
==> the input xs is classified as c2
"""
