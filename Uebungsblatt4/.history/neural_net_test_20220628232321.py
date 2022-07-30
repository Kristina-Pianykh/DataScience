import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import classification_report

USAGE_DESCRIPTION = """
Das Skript wurde mit der falschen Anzahl an Parametern aufgerufen.
Die korrekte Aufrufsyntax ist:

    python neural_net.py <eingabe_datei> <trainingsgroesse> <epochen> <batchgroesse>
        <eingabe_datei>    - Pfad zur Datei mit den Trainings- und Testdaten mnist.npz
        <trainingsgroesse> - Anzahl der zu verwendenden Trainingsbeispiele (0 für alle)
        <epochen>          - Anzahl der Trainingsepochen
        <batchgroesse>     - Anzahl der Trainingsinstanzen pro Batch


Beispiel:
    Training eines Netzwerk mit 10.000 Trainingsbeispiel für 35 Epoche mit je 256 Instanzen pro Batch:

    python neural_net.py mnist.npz 10000 35 256
"""

# Festlegung des Seeds für die Generierung der Zufallszahlen
np.random.seed(42)


def create_batches(input: np.ndarray, n: int) -> List[np.ndarray]:
    """
    Teilt das übergebene Numpy-Array in Teilarrays der Größe n ein.

    :param input: Eingabe Numpy-Array
    :param n: Größe eines Batches
    :return: Liste mit allen Teilarrays
    """
    l = len(input)
    batches = []
    for ndx in range(0, l, n):
        batches.append(input[ndx : min(ndx + n, l)])

    return batches


def load_data(
    input_file: Path,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Lädt die Daten aus der Eingabedatei.

    :param input_file: Pfad zur Eingabedatei
    :return: Tupel mit zwei Tupeln welches die Trainingsdaten und -klassen sowie die Testdaten und -klassen
             als Numpy-Array erfasst.
    """
    with np.load(str(input_file), allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)


def to_one_hot_vectors(y: np.ndarray) -> np.ndarray:
    """
    Diese Funktion wandelt die Goldstandard-Labels in One-Hot-Vektoren um.

    :param y: Goldstandard-Klassen als Array (n)
    :return: Goldstandard-Klasse als One-Hot-Vektoren (Dim (n, 10))
    """
    n = y.shape[0]
    categorical = np.zeros((n, 10), dtype="float32")
    categorical[np.arange(n), y] = 1
    output_shape = y.shape + (10,)
    categorical = np.reshape(categorical, output_shape)

    return categorical


def squash(input_vector: np.ndarray) -> np.ndarray:
    return np.exp(input_vector) / sum(np.exp(input_vector))


def main():
    train_size = 10000
    data_file = Path("mnist.npz")
    (x_train, y_train), (x_test, y_test) = load_data(data_file)
    # Reduziere die Trainingsdaten auf die eingegebene Menge
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    print(f"#Trainingsinstanzen: {x_train.shape[0]}")
    print(f"#Testinstanzen     : {x_test.shape[0]}\n")
    data = create_batches(x_train, 256)
    # Anpassung und Normalisierung der Trainingsdaten;
    # Jedes Bild wird als Matrix der Größe (1, 28*28=784) dargestellt
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_train = x_train.astype("float32")

    # Normalisiere die Eingabefeatures, sodass diese im Bereich [0,1] liegen
    x_train /= 255

    # Umwandlung der Goldstandard-Labels in ein One-Hot-Vektor;
    # (bspw. die Zahl 3 wird zu [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    y_train_enc = to_one_hot_vectors(y_train)

    # Führe die gleichen Umwandlungen auch für die Testdaten durch
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_test = x_test.astype("float32")
    x_test /= 255
    y_test_enc = to_one_hot_vectors(y_test)


main()
