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


class FeedforwardNetwork:
    """
    Diese Klasse implementiert ein einfaches, einschichtiges neuronales Netzwerk.
    """

    def __init__(self, input_size: int, output_size: int):
        # Membervariablen zur Erfassung der Gewichte und Bias-Werte der linearen
        # Transformation. Die Gewichte und Bias-Werte werden mit zufälligen Werten
        # initialisiert.
        self.weights = np.random.rand(
            input_size, output_size
        )  # Dimensionalität (<Anzahl-Features>, <Anzahl-Klassen>)
        self.bias = np.random.rand(output_size)  # Dimensionalität (<Anzahl-Klassen>)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diese Methode berechnet den Forward-Pass, d.h. die lineare Transformation der Eingabedaten und die Anwendung
        der Aktivierungsfunktion, für die in x gegebenen n Trainingsbeispiele. Als Aktivierungsfunktion soll die
        **Softmax-Funktion** genutzt werden.

        Die Eingabe der n Trainingsbeispiele erfolgt als Numpy-Array der Dimensionalität (n, 768). Als Rückgabe wird
        ein Tupel erwartet, welches die Netzwerkausgabe, d.h. die lineare Transformation ohne die Aktivierung, sowie
        die Aktivierungszustände des Netzwerks (nach Anwendung der Softmax-Funktion) repräsentiert. Beide Rückgaben
        sollen als Numpy-Float-Array der Dimensionalität (n, 10) erfolgen.

        :param x: Feature-Werte der n Eingabebeispiele (Dim. (n, 768))
        :return: Tupel mit der Netzwerkausgabe (ohne Aktivierung) und den Aktivierungszuständen als
                 Numpy-Arrays (Dim (n, 10))
        """
        not_activated = np.dot(x, self.weights) + self.bias
        activated = np.apply_along_axis(softmax, 1, not_activated)

        return not_activated, activated

    def predict_labels(self, p: np.ndarray) -> np.ndarray:
        """
        Diese Methode berechnet die Vorhersage, d.h. die vom Netzwerk geschätzte Klasse bzw. Ziffer, basierend auf den
        in p gegebenen (Softmax-) Aktivierungen von n Trainingsbeispielen.

        Die Eingabe von p erfolgt als Numpy-Float-Array der Dimensionalität (n, 10). Die Rückgabe der Labels soll
        als Numpy-Int-Array der Dimensionalität (n, 1) erfolgen.

        :param p: Aktivierungszustände von n Beispielen (Dim (n, 10))
        :return: Vorhersage des Netzwerks (Dim (n, 1))
        """
        total = p.shape[0]
        predictions = np.zeros(total, dtype=np.float64)
        for idx in range(total):
            predictions[idx] = np.argmax(p[idx])

        return predictions

    def loss(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Diese Methode berechnet den Wert der **Cross-Entropy-Fehlerfunktion** für n Trainingsbeispiele.

        Der Eingabeparameter p der Methode erfasst die Aktivierungen für die n Trainingsbeispiele und ist als
        Numpy-Float-Array der Dimensionalität (n, 10) gegeben. Die Übergabe der Goldstandard-Labels erfolgt mittels
        One-Hot-Vektoren über den Parameter y. y wird als Numpy-Int-Array der Dimensionalität (n, 10) übergeben.

        Die Rückgabe soll als Numpy-Float-Array der Dimensionalität (n, 1), welches den Fehlerwert der jeweiligen
        Trainingsbeispiele enthält, erfolgen.

        :param y: Goldstandard-Klassen für n Beispiele repräsentiert als One-Hot-Vektoren (Dim (n, 10))
        :param p: Aktivierungszustände von n Beispielen (Dim (n, 10))
        :return: Fehlerwerte der n Trainingsbeispiele (Dim (n, 1))
        """

        total = p.shape[0]
        loss = np.zeros((total, 10), dtype="float32")
        log_transformation = np.apply_along_axis(np.log, 1, p)
        # loss = -1 * npsumdot(log_transformation, y)
        for row in range(total):
            loss[row] = -1 * np.dot(log_transformation[row], y[row])
        return loss[:, 0]

    def backward(self, x: np.ndarray, p: np.ndarray, y: np.ndarray):
        """
        Diese Methode berechnet mittels Backpropagation die Gradienten der Gewichte und Bias-Werte für
        n Trainingsbeispiele.

        Die Eingabeparameter x und p repräsentieren hierbei die Features der n Trainingsbeispiele sowie die
        entsprechenden (Softmax) Aktivierungen des Netzwerks. Die Übergabe der Goldstandard-Klassen erfolgt als
        One-Hot-Vektoren über den Parameter y. y wird als Numpy-Int-Array der Dimensionalität (n, 10) übergeben.

        Als Rückgabe wird ein Tupel erwartet, welches die Gradienten der Gewichte (Numpy-Float-Array der
        Dimensionalität (768, 10)) und der Bias-Werte (Numpy-Float-Array der Dimensionalität (10)) erfasst.

        :param x: Feature-Werte der n Trainingsbeispiele (Dim (n, 768))
        :param p: (Softmax) Aktivierungszustände des Netzwerks der n Trainingsbeispiele (Dim (n, 10))
        :param y: Goldstandard-Klassen für n Beispiele repräsentiert als One-Hot-Vektoren (Dim (n, 10))

        :return: Tupel mit den Gradienten der Gewichte (Dim (768, 10)) und der Bias-Werte (Dim (10))
        """
        grad_weights = -1 * (np.dot(x.T, y + (-p))).T
        grad_bias = -1 * np.sum(y + (-p), axis=0)
        return grad_weights.T, grad_bias

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        # Erzeuge Batches mit jeweils batch_size Trainingsbeispielen
        x_batches = create_batches(x, batch_size)
        y_batches = create_batches(y, batch_size)
        num_batches = len(x_batches)

        # Trainingsschleife
        for i in range(epochs):
            sum_loss = 0

            # In jeder Epoche wird jeder Batch einmal betrachtet
            for b_x, b_y in zip(x_batches, y_batches):
                # Berechne den Forward-Pass
                net_output, p = self.forward(b_x)

                # Berechne die Gradienten
                grad_weights, grad_bias = self.backward(b_x, p, b_y)

                # Aktualisiere die Gewichte und Bias-Werte anhand des Gradienten
                self.weights -= learning_rate * grad_weights / b_x.shape[0]
                self.bias -= learning_rate * grad_bias / b_x.shape[0]

                # Berechne den aktuellen Fehlerwert (zum Debugging des Lernprozesses)
                loss = self.loss(b_y, p).mean()
                sum_loss += loss

            # Evaluiere das aktuelle Modell auf den Trainingsdaten (nur zum Debugging)
            y_pred = self.predict(x, batch_size)
            accuracy = (y_pred == y.argmax(axis=-1)).sum() / y.shape[0]
            accuracy = round(accuracy, 6)

            # Berechne den durchschnittlichen Fehlerwert
            loss = sum_loss / num_batches
            loss = round(loss, 6)

            print(
                f" Epoche: {i + 1}/{epochs}   Fehler={loss}   Trainingsgenauigkeit={accuracy}"
            )

        print("Training ist abgeschlossen\n")

    def predict(self, x: np.ndarray, batch_size: int):
        # Führe die Vorhersage auch in Batches durch
        x_batches = create_batches(x, batch_size)

        predictions = []
        for b_x in x_batches:
            # Berechne den Forward-Pass
            _, p_x = self.forward(b_x)

            # Bilde die Vorhersage anhand der Aktivierungen
            y_pred = self.predict_labels(p_x)
            predictions.append(y_pred)

        # Fasse die Vorhersage der einzelnen Batches in einem Array zusammen
        y_pred = np.concatenate(predictions)

        return y_pred

    def evaluate(self, x: np.ndarray, y_gold: np.ndarray, batch_size: int):
        y_pred = self.predict(x, batch_size)

        print(classification_report(y_gold, y_pred, digits=3))


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


def softmax(input_vector: np.ndarray) -> np.ndarray:
    return np.exp(input_vector) / np.sum(np.exp(input_vector))


def hypothesis(W: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = np.zeros(b.shape, dtype=np.float64)
    g = softmax(np.dot(W, x) + b)
    return g


if __name__ == "__main__":
    # if len(sys.argv) != 5:
    #     print(USAGE_DESCRIPTION)
    #     exit(-1)

    # data_file = Path(sys.argv[1])
    # if not data_file.exists() or data_file.is_dir():
    #     print(f"Die Datei {data_file} mit den Trainings- und Testdaten existiert nicht")
    #     exit(-1)

    # train_size = int(sys.argv[2])
    # if train_size <= 0:
    #     print(f"Die Anzahl der Trainingsinstanzen muss >= 1 sein")
    #     exit(-1)

    # num_epochs = int(sys.argv[3])
    # if num_epochs <= 0:
    #     print(f"Die Anzahl der Trainingsepochen muss > 0 sein")
    #     exit(-1)

    # batch_size = int(sys.argv[4])
    # if batch_size <= 0:
    #     print(f"Die Größe der Batches muss > 0 sein")
    #     exit(-1)

    data_file = Path("mnist.npz")
    train_size = 10000
    num_epochs = 35
    batch_size = 256
    # Laden der Daten aus Eingabedatei
    print(f"Lese die Trainings- und Testdaten von {data_file} ein")
    (x_train, y_train), (x_test, y_test) = load_data(data_file)

    # Reduziere die Trainingsdaten auf die eingegebene Menge
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    print(f"#Trainingsinstanzen: {x_train.shape[0]}")
    print(f"#Testinstanzen     : {x_test.shape[0]}\n")

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

    # Baue das Netzwerk auf
    ff_network = FeedforwardNetwork(28 * 28, 10)

    # Trainiere das Modell anhand der Trainingsdaten
    print(f"Starte das Training des Modells mit {len(x_train)} Beispielen")
    ff_network.fit(
        x_train,
        y_train_enc,
        batch_size=batch_size,
        epochs=num_epochs,
        learning_rate=0.1,
    )

    # Evaluiere das Modell
    print("Starte die Evaluation des Modells")
    ff_network.evaluate(x_test, y_test, batch_size=batch_size)
