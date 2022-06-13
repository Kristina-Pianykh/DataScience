# ---------------------------------------------------------------------------------------
# Abgabegruppe: 24
# Personen:
# - Kristina Pianykh, pianykhk, 617331
# - Miguel Nuno Carqueijeiro Athouguia, carqueim, 618203
# - Winston Winston, winstonw, 602307
# -------------------------------------------------------------------------------------
import sys
from math import log
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

USAGE_TEXT = """
Das Skript wurde mit der falschen Anzahl an Parametern aufgerufen.
Die korrekte Aufrufsyntax ist:

    python naive_bayes.py <trainingsdatei> <testdatei>
        <trainingsdatei> - Pfad zur Datei mit den Trainingsdaten
        <testdatei>      - Pfad zur Datei mit den Testdaten

Beispiel:

    python naive_bayes.py imdb_train.tsv imdb_test_1.tsv
"""


def read_reviews(input_file: Path) -> List[Tuple[str, str]]:
    """
        Liest die Filmbewertungen aus der Datei input_file und gibt diese
        als Liste von Tupeln (<bewertungstext>, <label>) zurück.

    :param input_file: Pfad zur Eingabedatei
    :return: Filmbewertungen als Liste von Tupeln (<text>, <label>)
    """
    reviews = []

    with input_file.open("r") as in_stream:
        for i, line in enumerate(in_stream.readlines()):
            if i == 0:
                continue

            text, label = line.split("\t")
            text = text.strip()
            label = label.strip()

            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]

            reviews.append((text, label))

    return reviews


class NaiveBayesClassifier:
    def __init__(self):
        self.label_probabilities: Dict[str, float] = {"pos": 0.0, "neg": 0.0}
        self.word_probabilities: Dict[str, Dict[str, float]] = {"pos": {}, "neg": {}}
        self.training_dataset_size = 0

    def get_split_words(self, reviews: Any) -> Tuple[Any, Any]:
        def get_word_list(sentiment: Literal["pos", "neg"]):
            reviews_ = reviews[np.where(reviews[:, 1] == sentiment)]
            words_ = reviews_[:, 0]
            return np.concatenate(np.char.split(words_))

        return get_word_list("pos"), get_word_list("neg")

    def train(self, reviews: List[Tuple[str, str]]):
        """
            Trainiert ein Naive Bayes Klassifikationsmodell basierend auf den in reviews gegebenen
            Trainingsbeispielen. Die Trainingsbeispiele werden als Liste von Tupeln übergeben. Jedes
            Tupel besteht aus dem Bewertungstext (1. Komponente) und dem Goldstandard-Klasse (2. Komponente).

            Die Funktion schätzt die Wahrscheinlichkeit p(c), dass ein Review zur Klasse c € {pos, neg} gehört,
            sowie die bedingte Wahrscheinlichkeit p(w|c), dass das Wort t in einem Review der Klasse c vorkommt,
            für alle Klassen und Wörter und speichert diese in geeigneten Datenstrukturen (in den Attributen
            der Klasse) ab.

            Zur Aufteilung des Bewertungstext in einzelne Wörter wird das Leerzeichen verwendet.

        :param reviews: List von Filmbewertungen im Format (<bewertungstext>, <klasse>)
        """
        reviews_total = len(reviews)
        self.training_dataset_size = reviews_total
        reviews_arr = np.array([np.array(review) for review in reviews])
        labels = reviews_arr[:, 1]
        unique_labels, count_labels = np.unique(labels, return_counts=True)
        label_probabilities = count_labels / reviews_total
        self.label_probabilities = dict(zip(unique_labels, label_probabilities))

        pos_words, neg_words = self.get_split_words(reviews_arr)

        def get_word_probabilities(words):
            words_total = len(words)
            unique_words, count_words = np.unique(words, return_counts=True)
            words_probabilities = count_words / words_total
            return dict(zip(unique_words, words_probabilities))

        self.word_probabilities["pos"] = get_word_probabilities(pos_words)
        self.word_probabilities["neg"] = get_word_probabilities(neg_words)

    def predict(self, reviews: List[str]) -> List[Tuple[str, float]]:
        """
            Berechnet die Vorhersage für die in reviews gegebenen Filmbewertungen basierend auf dem
            gelernten Modell. Die Filmbewertungen werden als Liste von Strings an die Funktion übergeben.

            Die Funktion gibt die berechneten Klassen als Liste von Tupeln (<klasse>, <score)) zurück. Die erste
            Komponente des Tupel repräsentiert dabei die vorhergesagte Klasse und die zweite Komponente den
            berechneten Score der Klasse. Die Reihenfolge der Ergebnisliste entspricht dabei der Reihenfolge der
            Filmkritiken in der Eingabeliste.

        :param reviews: Liste von Bewertungstexten als String
        :return: Liste von Tupeln (<klasse>, <score>), die die Vorhersage des Modells festhalten
        """
        log_label_probabilities = {
            sentiment: log(self.label_probabilities[sentiment])
            for sentiment in self.label_probabilities
        }
        predictions: List[Tuple[str, float]] = []

        def get_probability(word: str, sentiment: Literal["pos", "neg"]):
            return log(
                self.word_probabilities[sentiment].get(
                    word, 1 / self.training_dataset_size
                )
            )

        get_probability_vec = np.vectorize(get_probability)

        reviews_arr = np.array(reviews)
        split_reviews = np.char.split(reviews_arr)

        def get_label_probability(review, sentiment: Literal["pos", "neg"]) -> float:
            review_arr = np.array(review)
            probabilities = get_probability_vec(review_arr, sentiment)
            return np.sum(probabilities) + log_label_probabilities[sentiment]

        for split_review in split_reviews:
            prob_pos = get_label_probability(split_review, "pos")
            prob_neg = get_label_probability(split_review, "neg")

            if prob_pos > prob_neg:
                predictions.append(("pos", prob_pos))
            else:
                predictions.append(("neg", prob_neg))

        return predictions

    def evaluate(
        self, reviews: List[Tuple[str, str]], predictions: List[Tuple[str, float]]
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Diese Funktion nimmt eine Liste reviews von Filmbewertungen und eine dazugehörige Liste von Vorhersagen
        predictions entgegen und berechnet darauf basierend verschiedene Evaluationsmaße. Die Trainingsbeispiele
        werden als Liste von Tupeln (<text>, <klasse>) übergeben. Die Vorhersagen werden als Liste von Tupeln
        (<klasse>, <score>) übergeben.

        Die Funktion gibt die erzielte Genauigkeit (Accuracy), Precision, Recall und den F1-Wert für beide Klassen
        sowie den erzielten Micro- und Macro-F1-Wert als 8-Tupel zurück:

        (<Pos-Precision>, <Pos-Recall>, <Pos-F1>, <Neg-Precision>, <Neg-Recall>, <Neg-F1>,  <F1-Micro>, <F1-Macro>)

        Runden Sie alle Ergebnisse auf drei Nachkommastellen.

        :param reviews: Liste von Filmbewertungen als Tupel (<Text>, <Klasse>)
        :param predictions: Liste von dazugehörigen Vorhersagen (<Klasse>, <Score>)
        :return: 8-Tupel  mit den erzielten Ergebnissen
        """
        confusion_table = {"pos": {}, "neg": {}}
        results = {"pos": {}, "neg": {}}

        def compare_results_per_sentiment(
            reviews: List[Tuple[str, str]],
            predictions: List[Tuple[str, float]],
        ):
            labels_rev = np.array([np.array(entry) for entry in reviews])[:, 1]
            labels_pred = np.array([np.array(entry) for entry in predictions])[:, 0]
            return labels_rev, labels_pred

        review_sentiments, predicted_sentiments = compare_results_per_sentiment(
            reviews, predictions
        )
        all_sentiments = np.stack((review_sentiments, predicted_sentiments), axis=1)
        accuracy = np.count_nonzero(review_sentiments == predicted_sentiments) / total = len(reviews)
        print(f"Accuracy: {accuracy}")

        for sentiment in ["pos", "neg"]:
            sentiments_by_review = all_sentiments[
                np.where(all_sentiments[:, 0] == sentiment)
            ]
            sentiments_by_prediction = all_sentiments[
                np.where(all_sentiments[:, 1] == sentiment)
            ]
            other_sentiments_review = all_sentiments[
                np.where(all_sentiments[:, 0] != sentiment)
            ]
            true_positives = np.count_nonzero(
                sentiments_by_review[:, 0] == sentiments_by_review[:, 1]
            )
            false_positives = np.count_nonzero(
                sentiments_by_prediction[:, 0] != sentiments_by_prediction[:, 1]
            )
            false_negatives = np.count_nonzero(
                sentiments_by_review[:, 0] != sentiments_by_review[:, 1]
            )
            true_negatives = np.count_nonzero(
                other_sentiments_review[:, 0] == other_sentiments_review[:, 1]
            )
            confusion_table[sentiment]["true positives"] = true_positives
            confusion_table[sentiment]["false positives"] = false_positives
            confusion_table[sentiment]["false negatives"] = false_negatives
            confusion_table[sentiment]["true negatives"] = true_negatives

            results[sentiment]["precision"] = true_positives / (
                true_positives + false_positives
            )
            results[sentiment]["recall"] = true_positives / (
                true_positives + false_negatives
            )
            results[sentiment]["F1 score"] = (
                2 * results[sentiment]["precision"] * results[sentiment]["recall"]
            ) / (results[sentiment]["precision"] + results[sentiment]["recall"])

        results["F1 micro"] = (
            confusion_table["pos"]["true positives"]
            + confusion_table["neg"]["true positives"]
        ) / (
            confusion_table["pos"]["true positives"]
            + confusion_table["neg"]["true positives"]
            + confusion_table["pos"]["false positives"]
            + confusion_table["neg"]["false positives"]
        )
        results["F1 macro"] = (
            results["pos"]["precision"] + results["neg"]["precision"]
        ) / 2
        return tuple(
            [
                round(metric, 3)
                for metric in [
                    results["pos"]["precision"],
                    results["pos"]["recall"],
                    results["pos"]["F1 score"],
                    results["neg"]["precision"],
                    results["neg"]["recall"],
                    results["neg"]["F1 score"],
                    results["F1 micro"],
                    results["F1 macro"],
                ]
            ]
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(USAGE_TEXT)
        exit(-1)

    train_file = Path(sys.argv[1])
    if not train_file.exists() or train_file.is_dir():
        print(f"Die Datei {train_file} mit den Trainingsdaten existiert nicht")
        exit(-1)

    test_file = Path(sys.argv[2])
    if not test_file.exists() or test_file.is_dir():
        print(f"Die Datei {train_file} mit den Testdaten existiert nicht")
        exit(-1)

    # Einlesen der Trainingsdaten
    train_data = read_reviews(train_file)

    # Erstelle und trainiere das Naive Bayes Modell
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(train_data)

    # Einlesen der Testdaten
    test_data = read_reviews(test_file)

    # Berechne die Vorhersage für die Testbewertungen
    test_reviews = [review[0] for review in test_data]
    predictions = nb_classifier.predict(test_reviews)

    # Gebe den Text und die Vorhersage der ersten 10 Testbewertungen aus
    for review, prediction in zip(test_data[:10], predictions[:10]):
        print(review[0])
        print(f"\tGoldstandard={review[1]}")
        print(f"\tVorhersage={prediction[0]} ({prediction[1]})\n")

    # Evaluiere das gelernte Modell
    results = nb_classifier.evaluate(test_data, predictions)

    print(f"Pos - Precision: {results[0]}")
    print(f"Pos - Recall: {results[1]}")
    print(f"Pos - F1: {results[2]}")
    print("----------------------------------")

    print(f"Neg - Precision: {results[3]}")
    print(f"Neg - Recall: {results[4]}")
    print(f"Neg - F1: {results[5]}")
    print("----------------------------------")

    print(f"F1 (Micro): {results[6]}")
    print(f"F1 (Macro): {results[7]}")
