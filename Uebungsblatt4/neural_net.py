import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import classification_report

USAGE_DESCRIPTION = """
The script was called with the wrong number of parameters.
The correct calling syntax is:
     python neural_net.py <input_file> <training size> <epochs> <batch size>
         <input_file> - path to the file containing the training and testing data mnist.npz
         <training size> - number of training samples to use (0 for all)
         <epochs> - number of training epochs
         <batchsize> - number of training instances per batch
Example:
     Training a network with 10,000 training example for 35 epochs with 256 instances per batch:
     python neural_net.py mnist.npz 10000 35 256
"""

# fix the seed for the generation of random numbers
np.random.seed(42)


def create_batches(input: np.ndarray, n: int) -> List[np.ndarray]:
    """
    Divides the given numpy array into subarrays of size n.

     :param input: input numpy array
     :param n: size of a batch
     :return: list with all subarrays
    """
    l = len(input)
    batches = []
    for ndx in range(0, l, n):
        batches.append(input[ndx : min(ndx + n, l)])

    return batches


class FeedforwardNetwork:
    """
    This class implements a simple, one-layer neural network
    """

    def __init__(self, input_size: int, output_size: int):
        # class attributes for the weights and bias values of the linear
        # transformation (initialized to random values)
        self.weights = np.random.rand(
            input_size, output_size
        )  # dimensionality (<number-of-features>, <number-of-classes>)
        self.bias = np.random.rand(output_size)  # dimensionality (<number-of-classes>)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method calculates the forward pass, i.e. the linear transformation
        of the input data and provides the output of the activation function
        for the n training examples given in x. As an activation function, the
        **Softmax function** is used.
        The n training examples are passed as a numpy array of dimensionality (n, 768).
        The return results are the network output, i.e. the results of the linear
        transformation without the activation, as well as the activation states of
        the network (after applying the softmax function). The return values are
        expected as a numpy float array of dimensionality (n, 10).

        :param x: feature values of the n input samples (Dim. (n, 768))
        :return: tuple with the network output (without activation) and
        the activation states as Numpy Arrays (Dim(n,10))
        """
        not_activated = np.dot(x, self.weights) + self.bias
        activated = np.apply_along_axis(softmax, 1, not_activated)

        return not_activated, activated

    def predict_labels(self, p: np.ndarray) -> np.ndarray:
        """
        This method calculates the prediction, i.e. the class or the digit,
        based on the given (softmax) activations p of n training examples.
        Input p is a numpy float array of dimensionality (n, 10).
        The return of the labels is a numpy int array of dimensionality (n, 1).

        :param p: activation states of n examples (Dim (n, 10))
        :return: prediction of the network (Dim(n,1))
        """
        total = p.shape[0]
        predictions = np.zeros(total, dtype=np.float32)
        for idx in range(total):
            predictions[idx] = np.argmax(p[idx])

        return predictions

    def loss(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        This method calculates the values of the **cross-entropy error function**
        for n training samples. The input parameter p of the method captures
        the activations for the n training samples and it is passed as a numpy float
        array of dimensionality (n, 10). The gold standard labels are handed over
        by means of one-hot vectors via the y parameter. y is passed as a numpy int
        array of dimensionality (n, 10). The return should be a numpy float array of
        dimensionality (n, 1) containing the error value of the respective includes
        training examples.

        :param y: gold standard labels for n examples represented as one-hot
        vectors (Dim(n, 10))
        :param p: activation states of n examples (Dim (n, 10))
        :return: error values of the n training examples (Dim (n, 1))
        """

        def npsumdot(x, y):
            return np.sum(x * y, axis=1)

        log_transformation = np.apply_along_axis(np.log, 1, p)
        vectorized_loss = -1 * npsumdot(log_transformation, y)

        return vectorized_loss

    def backward(self, x: np.ndarray, p: np.ndarray, y: np.ndarray):
        """
        This method calculates the gradients of the weights and bias values
        ​​for using backpropagation for n training examples.
        The input parameters x and p represent the features of the n training examples
        and the corresponding (softmax) activations of the network. The gold standard
        classes are handed over as one-hot vectors via the y parameter. y is passed as
        a numpy int array of dimensionality (n, 10).
        The return value isa tuple_ which contains the gradients of the weights
        (numpy float array with the dimensionality(768,10)) and the bias values ​
        (numpy float array with dimensionality(10)).

        :param x: feature values ​​of the n training examples (Dim (n, 768))
        :param p: (softmax) activation states of the network of the n training
        examples (Dim(n,10))
        :param y: gold standard labels for n examples represented as one-hot vectors
        (Dim(n, 10))
        :return: tuple with the gradients of the weights (Dim(768,10)) and bias
        values ​​(Dim(10))
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
    Loads the data from the input file.
    :param input_file: path to the input file
    :return: Tuple with two tuples containing the training data with lables
    and the test data with labels captured as a numpy array.
    """
    with np.load(str(input_file), allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)


def to_one_hot_vectors(y: np.ndarray) -> np.ndarray:
    """
    This function converts the gold standard labels into one-hot vectors.
    :param y: gold standard labels as array(s)
    :return: gold standard labels as one-hot vectors (Dim(n,10))
    """
    n = y.shape[0]
    categorical = np.zeros((n, 10), dtype="float32")
    categorical[np.arange(n), y] = 1
    output_shape = y.shape + (10,)
    categorical = np.reshape(categorical, output_shape)

    return categorical


def softmax(input_vector: np.ndarray) -> np.ndarray:
    return np.exp(input_vector) / np.sum(np.exp(input_vector))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(USAGE_DESCRIPTION)
        exit(-1)

    data_file = Path(sys.argv[1])
    if not data_file.exists() or data_file.is_dir():
        print(f"Die Datei {data_file} mit den Trainings- und Testdaten existiert nicht")
        exit(-1)

    train_size = int(sys.argv[2])
    if train_size <= 0:
        print(f"Die Anzahl der Trainingsinstanzen muss >= 1 sein")
        exit(-1)

    num_epochs = int(sys.argv[3])
    if num_epochs <= 0:
        print(f"Die Anzahl der Trainingsepochen muss > 0 sein")
        exit(-1)

    batch_size = int(sys.argv[4])
    if batch_size <= 0:
        print(f"Die Größe der Batches muss > 0 sein")
        exit(-1)

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
