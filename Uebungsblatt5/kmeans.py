# ---------------------------------------------------------------------------------------
# Abgabegruppe: 24
# Personen:
# - Kristina Pianykh, pianykhk, 617331
# - Miguel Nuno Carqueijeiro Athouguia, carqueim, 618203
# - Winston Winston, winstonw, 602307
# -------------------------------------------------------------------------------------
import matplotlib
import numpy as np
import random
import sys
from copy import deepcopy

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.manifold import TSNE
from typing import Tuple

USAGE_DESCRIPTION = """
Das Skript wurde mit der falschen Anzahl an Parametern aufgerufen.
Die korrekte Aufrufsyntax ist:

    python kmeans.py <eingabe_datei> <anzahl_datenpunkte> <k> <max_iterationen>
        <eingabe_datei>      - Pfad zur Datei mit den Trainings- und Testdaten mnist.npz
        <anzahl_datenpunkte> - Anzahl der zu verwendenden Datenpunkte
        <k>                  - Anzahl der zu ermittelnden Cluster
        <max_iterationen>    - Maximale Anzahl an Iterationen
    
Beispiel:
    python kmeans.py mnist.npz 2000 10 50
"""


class KMeans:
    def init_clusters(self, points: np.ndarray, k: int) -> np.ndarray:
        """
        Diese Methode berechnet die initialen zufälligen Clusterzentren. Hierzu werden aus den in points gegebenen
        Datenpunkten zufällig k Punkte als Clusterzentren ausgewählt.

        Die Übergabe der Datenpunkte erfolgt NumPy-Array der Dimensionalität (n, m), wobei n die Anzahl der Datenpunkte
        und m die Anzahl der Merkmale pro Datenpunkt repräsentiert. Die Rückgabe der gewählten Clustermittelpunkte
        erfolgt als NumPy-Array der Dimensionalität (k, m).

        :param points: Datenpunkte als NumPy-Array (Dim. (n, m))
        :param k: Anzahl an zu identifizierenden Clustern
        :return: Zufällig gewählte Clusterzentren als NumPy-Array (Dim. (k, m))
        """
        random_indices = random.sample(range(points.shape[0]), k)
        return points[random_indices, :]

    def compute_distances(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Diese Methode berechnet den euklidischen Abstand zwischen jedem der in points gegebenen Datenpunkte und jedem
        der in centers gegebenen Clusterzentren.

        Die Übergabe der Parameter points und centers erfolgt als NumPy-Array der Dimensionalität (n, m) bzw. (k, m),
        wobei n die Anzahl der Datenpunkte, k die Anzahl der Clusterzentren und m die Anzahl der Features je
        Datenpunkt bzw. Clusterzentrum repräsentiert.

        Die Rückgabe erfolgt als NumPy-Array der Dimensionalität (n, k). Hierbei hält die i-te Zeile die Abstände
        des i-ten Datenpunktes zu jedem der k Clusterzentren fest. Die i-te Zeile und j-te Spalte erfasst
        dementsprechend den Abstand zwischen dem i-ten Datenpunkt und dem j-ten Clusterzentrum aus der Eingabe.

        :param points: Datenpunkte als NumPy-Array (Dim. (n, m))
        :param centers: Clusterzentren als NumPy-Array (Dim. (k, m))
        :return: Abstandsmatrix als NumPy-Array (Dim (n, k))
        """
        distances = np.zeros((points.shape[0], centers.shape[0]), dtype="float32")

        for idx in range(distances.shape[0]):
            for cluster in range(centers.shape[0]):
                distances[idx][cluster] = np.linalg.norm(
                    points[idx, :] - centers[cluster, :]
                )
        return distances

    def compute_cluster_assignments(self, distances: np.ndarray) -> np.ndarray:
        """
        Diese Methode berechnet die Zuordnung der Datenpunkte zu einem Cluster basierend auf den in distances gegebenen
        Abständen zwischen den Datenpunkten und den Clusterzentren. Ein Datenpunkt wird hierbei dem Cluster zugeordnet,
        zu dem es den geringsten Abstand besitzt.

        Die Übergabe der Abstände erfolgt als NumPy-Array der Dimensionalität (n, k), wobei n die Anzahl der Datenpunkte
        und k die Anzahl der Clusterzentren repräsentiert. Die i-te Zeile und j-te Spalte von distances erfasst den
        Abstand zwischen dem i-ten Datenpunkt und dem j-ten Clusterzentrum  Der Eintrag distances[0][1] gibt bspw. den
        Abstand zwischen dem ersten Datenpunkt und dem zweiten Clusterzentrum wieder.

        Die Rückgabe der Zuordnung erfolgt als NumPy-Array der Länge n. Der i-te Eintrag erfasst dabei das zugeordnete
        Cluster des i-ten Datenpunkts. Das zugeordnete Cluster wird über dessen Index (von 0 bis k-1) repräsentiert.
        Wird der i-te Datenpunkt bspw. dem dritten Cluster zugeordnet, so ist der i-te Eintrag der Rückgabe 2 (als int).

        :param distances: Abstandsmatrix als NumPy-Array (Dim. (n, k))
        :return: Clusterzuordnung als NumPy-Array (Dim (n))
        """
        cluster_assignments = np.zeros(distances.shape[0], dtype=int)
        min_distances = np.min(distances, axis=1)

        for idx in range(distances.shape[0]):
            cluster_assignments[idx] = np.where(
                distances[idx, :] == min_distances[idx]
            )[0]

        return cluster_assignments

    def compute_cluster_centers(
        self, points: np.ndarray, assignments: np.ndarray, k: int
    ) -> np.ndarray:
        """
         Diese Methode berechnet die neuen Zentren der k Cluster basierend auf den in points gegebenen Datenpunkten und
         den in assignments festgehalten Clusterzuordnungen. Das neue Zentrum eines Clusters wird über den Mittelwert
         (pro Merkmal) der einem Cluster zugeordneten Datenpunkte gebildet.

         Die Übergabe der Datenpunkte erfolgt als NumPy der Dimensionalität (n, m). Die Clusterzuordnung assignments
         ist als NumPy-Array der Länge n gegeben. Die Cluster sind hierbei als Index von 0 bis k-1 erfasst.

         Die Rückgabe der neuen Clusterzentren erfolgt als NumPy-Array der Dimensionalität (k, m), wobei m die Anzahl
         der Merkmale je Datenpunkte bzw. Clusterzentrum sei. Die i-te Zeile erfasst das (neue) Clusterzentrum des
         Clusters mit dem Index i.

        :param points: Datenpunkte als NumPy-Array (Dim. (n, m))
        :param assignments: Zuordnung der Datenpunkte zu Clusterzentren als NumPy-Array (Dim (n))
        :param k: Anzahl Cluster
        :return: Neue Clusterzentren als NumPy-Array (Dim. (k, m))
        """
        new_cluster_centers = np.zeros((k, points.shape[1]), dtype="float32")
        for idx in range(k):
            points_subset = points[np.where(assignments == idx), :]
            new_cluster_centers[idx] = np.mean(points_subset, axis=1)

        return new_cluster_centers

    def cluster(self, points: np.ndarray, k: int, max_iterations: int) -> np.ndarray:
        """
        Diese Methode berechnet eine Aufteilung der in points gegebenen Datenpunkte in k Cluster mittels des
        k-Means-Algorithmus. Der Algorithmus wird solange durchgeführt bis kein Datenpunkt einem neuen Cluster
        zugeordnet wird (max_iterations=0) oder max_iterations Iterationen durchgeführt wurden.

        Zur Implementierung können Sie die zuvor implementierten Methoden des Templates nutzen. Die Übergabe der
        Datenpunkte erfolgt als NumPy-Array der Dimensionalität (n, m), wobei n die Anzahl der Datenpunkte und m die
        Anzahl der Merkmale je Datenpunkte repräsentiert. Die Parameter k und max_iterations werden als int übergeben.

        Die Methode gibt die Zuordnung der Datenpunkte zu den jeweiligen Clustern als NumPy-Array der Länge n zurück.

        :param points: Datenpunkte als NumPy-Array (Dim. (n, m))
        :param k: Anzahl Cluster
        :param max_iterations: Maximale Anzahl an Iterationen oder 0 für unbegrenzte Durchführung
        :return: Clusterzuordnung als NumPy-Array (Dim (n))
        """
        cluster_assignments = np.zeros(points.shape[0], dtype=int)
        old_cluster_assignments = np.ones(
            points.shape[0], dtype=int
        )  # to keep track of the clustering in the prev iteration
        iteration_count = 0
        clusters = self.init_clusters(points, k)

        while True:
            max_iteration_reached = k and (iteration_count >= k)
            no_change_in_cluster_assignments = (
                not k and (old_cluster_assignments == cluster_assignments).all()
            )

            if max_iteration_reached or no_change_in_cluster_assignments:
                break

            distances = self.compute_distances(points, clusters)

            old_cluster_assignments = deepcopy(cluster_assignments)

            cluster_assignments = self.compute_cluster_assignments(distances)
            clusters = kmeans.compute_cluster_centers(points, cluster_assignments, k)
            iteration_count += 1

        return cluster_assignments


def load_data(input_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lädt die Daten aus der Eingabedatei.

    :param input_file: Pfad zur Eingabedatei
    :return: Tupel mit zwei Tupeln welches die Trainingsdaten und -klassen sowie die Testdaten und -klassen
             als Numpy-Array erfasst.
    """
    with np.load(str(input_file), allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        x = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        return x, y


def generate_cluster_goldstandard_comparison(
    points: np.ndarray, gold_labels: np.ndarray, assignments: np.ndarray
):
    """
    Erzeugt einen Vergleich der Goldstandard-Klassifikation der Datenpunkte und der identifizierten Cluster
    des Algorithmus. Hierzu werden die Datenpunkte zunächst mittels t-SNE auf zwei Dimensionen reduziert
    und dann mittels Scatterplots dargestellt.

    :param points: Datenpunkte als NumPy-Array (Dim. (n, m))
    :param gold_labels: Goldstandard-Klassifikation als NumPy-Array (Dim. (n))
    :param assignments: Clusterzuordnung der Datenpunkte als NumPy-Array (Dim (n))
    """
    # Reduziere die Datenpunkte auf 2D mit TSNE
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
    projected_points = tsne.fit_transform(points)

    # Speichere die Abbildung in einem DataFrame
    tsne_df = pd.DataFrame(data=projected_points, columns=["tsne_1", "tsne_2"])

    # Erstelle eine Grafik, welche zwei Subgrafik enthält
    fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(20, 8))

    # Erstelle ein Diagramm, welches die Goldstandard-Klassen für jeden Punkt anzeigt
    plt = sns.scatterplot(
        ax=axes[0],
        data=tsne_df,
        x="tsne_1",
        y="tsne_2",
        hue=gold_labels,
        palette="dark",
    )
    plt.set(title="Goldstandard Klassifikation")

    # Wandle Cluster-Ids in Buchstaben um, um Irritationen zu vermeiden!
    cluster_chars = [f"{chr(cluster_id + 65)}" for cluster_id in assignments]

    # Erstelle ein Diagramm, welches die Clusterzuordnung für jeden Punkt anzeigt
    plt = sns.scatterplot(
        ax=axes[1],
        data=tsne_df,
        x="tsne_1",
        y="tsne_2",
        hue=cluster_chars,
        palette="muted",
    )
    plt.set(title="Ergebnis K-Means Clustering")

    fig.savefig("vergleich.pdf")


def generate_cluster_examples(
    points: np.ndarray, assignments: np.ndarray, k: int, num_examples: int
):
    """
    Erstellt eine Übersicht mit num_examples zufällig gewählten Beispielen pro Cluster.

    :param points: Datenpunkte als NumPy-Array (Dim. (n, m))
    :param assignments: Clusterzuordnung der Datenpunkte als NumPy-Array (Dim (n))
    :param k: Anzahl der Cluster
    :param num_examples: Anzahl an Datenpunkten pro Cluster
    """
    fig, axes = matplotlib.pyplot.subplots(k, num_examples, figsize=(20, 30))
    for cluster_id in range(k):
        cluster_points = points[assignments == cluster_id]

        random_idx = random.sample(range(0, len(cluster_points)), num_examples)
        for i, point_idx in enumerate(random_idx):
            pixels = cluster_points[point_idx].reshape((28, 28))
            axes[cluster_id, i].imshow(pixels)
            if i == 0:
                axes[cluster_id, i].set_title(
                    f"Beispiele für Cluster {chr(cluster_id+65)}",
                    fontsize=16,
                    weight="bold",
                )

    fig.subplots_adjust(hspace=0)
    fig.savefig("beispiele.pdf")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(USAGE_DESCRIPTION)
        exit(-1)

    data_file = Path(sys.argv[1])
    if not data_file.exists() or data_file.is_dir():
        print(f"Die Datei {data_file} mit den Eingabedaten existiert nicht")
        exit(-1)

    num_points = int(sys.argv[2])
    if num_points <= 1:
        print(f"Die Anzahl der zu verwendenden Datenpunkte muss > 1 sein")
        exit(-1)

    k = int(sys.argv[3])
    if k <= 1:
        print(f"Die Anzahl der zu ermittelnden Cluster k muss > 1 sein")
        exit(-1)

    max_iterations = int(sys.argv[4])
    if max_iterations < 0:
        print(f"Die maximale Anzahl an Iterationen muss >= 0 sein")
        exit(-1)

    # Laden der Daten aus Eingabedatei
    print(f"Lese MNIST-Daten von {data_file} ein")
    points, labels = load_data(data_file)

    # Reduzierung der Datenpunkte auf die gewünschte Anzahl
    points = points[:num_points]
    labels = labels[:num_points]

    # Anpassung und Normalisierung der Datenpunkte;
    # Jedes Bild wird als Matrix der Größe (1, 28*28=784) dargestellt
    points = points.reshape(points.shape[0], 28 * 28)
    points = points.astype("float32")

    # Normalisiere die Pixcelwerte der Bilder, sodass diese im Bereich [0,1] liegen
    points /= 255

    # Durchführung des Clusterings
    kmeans = KMeans()
    cluster = kmeans.cluster(points, k, max_iterations)

    # Ausgabe eines Vergleichs von Goldstandard-Klassen und den Clusterzuordnungen
    generate_cluster_goldstandard_comparison(points, labels, cluster)

    # Ausgabe von 12 zufälligen Datenpunkten pro Cluster
    generate_cluster_examples(points, cluster, k, 12)
