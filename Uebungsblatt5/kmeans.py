import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

USAGE_DESCRIPTION = """
The script was called with the wrong number of parameters.
The correct calling syntax is:

     python kmeans.py <input_file> <number_of_datapoints> <k> <max_iterations>
        <input_file> - path to the file containing the training and testing data mnist.npz
        <number_datapoints> - number of datapoints
        <k> - number of clusters to determine
        <max_iterations> - maximum number of iterations

Example:
     python kmeans.py mnist.npz 2000 10 50
"""


class KMeans:
    def init_clusters(self, points: np.ndarray, k: int) -> np.ndarray:
        """
        This method calculates the initial random cluster centers.
        For this purpose, from the given data points select k random
        points to be the initial cluster centers.

        The data points are passed as a NumPy array of dimensionality (n, m)
        where n is the number of data points and m represents the number of
        features per data point. Returning the chosen cluster centers is done
        in a NumPy array of dimensionality (k, m).

        :param points: data points as NumPy array (Dim. (n, m))
        :param k: number of clusters to compute
        :return: randomly chosen cluster centers as NumPy array (Dim. (k, m))
        """
        random_indices = random.sample(range(points.shape[0]), k)
        return points[random_indices, :]

    def compute_distances(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        This method calculates the Euclidean distance between each of the data
        points and each of the cluster centers.

        The points and centers parameters are passed as a NumPy array with the
        dimensionality (n, m) or (k, m) where n is the number of data points,
        k is the number of cluster centers, and m is the number of features that
        represent a data point or cluster center.

        The return value is a NumPy array of dimensionality (n, k). Here,
        the i-th line holds the distances of the i-th data point to each of
        the k cluster centers. The i th row and j th column capture the distance
        between the i-th data point and the j-th cluster center, respectively.

        :param points: data points as NumPy array (Dim. (n, m))
        :param centers: cluster centers as NumPy array (Dim. (k, m))
        :return: distance matrix as NumPy array (Dim(n,k))
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
        This method calculates the assignment of the data points to a cluster
        based on those given in distances between the data points and the cluster
        centers. A data point is assigned to the cluster, to which it has
        the smallest distance.

        The distances are passed as a NumPy array of dimensionality (n, k)
        where n is the number of data points and k represents the number of cluster
        centers. The i-th row and j-th column of distances captures the distance
        between the i-th data point and the j-th cluster center. For example,
        the entry distances[0][1] reflects the distance between the 1st data point
        and the 2nd cluster center.

        The return value is a NumPy array of cluster assignments of length n.
        The i-th entry reflects the cluster assignment of the i-th data point.
        The assigned cluster is represented by its index (from 0 to k-1).
        For example, if the i-th data point is assigned to the 3rd cluster,
        the i-th entry is returned as 2 (as an int).

        :param distances: distance matrix as NumPy array (Dim.(n,k))
        :return: cluster mapping as NumPy array (Dim(n))
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
        This method calculates the new centers of the k clusters
        based on the data points and the cluster assignments passed as input.
        The new cluster is calculated as the mean (per feature) of the data
        points assigned to one cluster.

        The data points are passed as NumPy of dimensionality (n, m).
        The cluster map assignments are passed as a NumPy array of length n.
        The clusters are numbered based on their array index from 0 to k-1.

        The new cluster centers are returned as a NumPy array of dimensionality (k, m)
        where m is the count of the features per data point or cluster center.
        The i-th line captures the (new) cluster center of the clusters with index i.

        :param points: data points as NumPy array (Dim. (n, m))
        :param assignments: assignment of data points to cluster centers as
        NumPy array (Dim (n))
        :param k: number of clusters
        :return: new cluster centers as NumPy array (Dim. (k, m))
        """
        new_cluster_centers = np.zeros((k, points.shape[1]), dtype="float32")
        for idx in range(k):
            points_subset = points[np.where(assignments == idx), :]
            new_cluster_centers[idx] = np.mean(points_subset, axis=1)

        return new_cluster_centers

    def cluster(self, points: np.ndarray, k: int, max_iterations: int) -> np.ndarray:
        """
        This method calculates a division of the data points into k clusters
        using the k-means algorithm. The algorithm is carried out until no
        data point has been reaasigned to a new cluster (max_iterations=0)
        or the number of max iterations has been reached.

        The data points are passed as a NumPy array of dimensionality (n,m)
        where n is the number of datapoints and m is the number of features
        per data point. The k and max_iterations parameters are passed as an int.

        The method returns the assignment of the data points to the respective
        clusters as a NumPy array of length n.

        :param points: data points as NumPy array (Dim. (n, m))
        :param k: number of clusters
        :param max_iterations: maximum number of iterations or 0 for unlimited execution
        :return: cluster mapping as NumPy array (Dim(n))
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
    Loads the data from the input file.

    :param input_file: path to the input file
    :return: tuple with the training data and lables and the test data and labels,
    each as a numpy array.
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
    Produces a comparison of the gold standard classification of the data points
    and the identified clusters of the algorithm. For this purpose,
    the data points are first reduced to two dimensions using t-SNE and
    then represented by scatterplots.

    :param points: data points as NumPy array (Dim. (n, m))
    :param gold_labels: gold standard classification as NumPy array (Dim.(n))
    :param assignments: cluster assignment of the data points as NumPy array (Dim (n))
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
    Generates an overview with num_examples randomly chosen examples per cluster.

    :param points: data points as NumPy array (Dim. (n, m))
    :param assignments: cluster assignment of the data points as NumPy array (Dim (n))
    :param k: number of clusters
    :param num_examples: number of data points per cluster
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
