from sklearn.metrics import accuracy_score
from scipy.special import softmax
import numpy as np


@np.vectorize
def predict(num):
    if num >= 0.5:
        return 1
    return 0


def cross_enthropy(y_class_true, y_scores):
    y_class_arr = np.asarray(y_class_true).reshape(-1)
    y_scores_arr = np.asarray(y_scores).reshape(-1)
    result = -1 * np.dot(y_class_arr, np.log(y_scores_arr))
    return round(result, 6)


def accuracy():
    labels = {"sport": 1, "politics": 2, "society": 3}

    y_pred = [2, 3, 1, 2, 2, 3, 1, 3, 1, 2, 1, 3, 2, 2, 2, 2]
    y_true = [2, 2, 1, 2, 3, 2, 1, 3, 3, 2, 1, 1, 2, 2, 1, 2]

    print(f"Accuracy score: {accuracy_score(y_true, y_pred)}")


def neurons():
    W = np.matrix("1 2 0.5 4 2; 2 4 1 2 1")
    print(f"gewichtematrix:\n{W}")
    print(f"or:\n{W.T}\n")
    x1 = np.array([2, -2, 4, 0, 1])
    x2 = np.array([-3, 2, 2, -1, 3])
    b = np.array([1, -0.5])
    y1 = np.dot(W, x1) + b
    y2 = np.dot(W, x2) + b
    print(f"Eingabe x1: {y1}")
    print(f"Eingabe x1: {y2}\n")

    z1 = softmax(y1)
    z2 = softmax(y2)

    class_pred_y1 = predict(z1)
    class_pred_y2 = predict(z2)
    true_class_y1 = np.matrix("1; 0")
    true_class_y2 = np.matrix("0; 1")

    print(
        f"Aktivierungszustände für y1 gegeben x1: {z1}. "
        f"The predicted class: {class_pred_y1}. "
        f"The cross entrope error: {cross_enthropy(true_class_y1, z1)}"
    )
    """
    Aktivierungszustände für y1 gegeben x1: [[0.92414182 0.07585818]].
    The predicted class: [[1 0]].
    The cross entrope error: 0.07889
    """
    print(
        f"Aktivierungszustände für y2 gegeben x2: {z2}. "
        f"The predicted class: {class_pred_y2}. "
        f"The cross entrope error: {cross_enthropy(true_class_y2, z2)}"
    )
    """
    Aktivierungszustände für y2 gegeben x2: [[0.62245933 0.37754067]].
    The predicted class: [[1 0]].
    The cross entrope error: 0.974077
    """


def main():
    accuracy()
    neurons()


if __name__ == "__main__":
    main()
