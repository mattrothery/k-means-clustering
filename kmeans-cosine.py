# This Python script is an implementation of the K-Means clustering algorithm. In this file, Cosine Similarity is used
# to update the cluster means. Unnormalized / normalized data is used as the input, a graph is produced comparing the
# Precision, Recall and F-Score for K values from 1-10, for both the unnormalized and normalized inputs.

# Written by Matthew Rothery - 201028655

# This file works in the same way as kmeans-euclidean. The only differences are the distance method and there is no
# normalization performed on the input data.

import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt


def import_data():
    print("Importing data...")
    word_types = ('animals', 'countries', 'fruits', 'veggies')
    word_type_counts = np.zeros(len(word_types))

    feature_data = []
    for type in word_types:
        with open(type) as file:
            for line in file:
                instance = line.strip().split(' ')
                instance = [float(x) for x in instance[1:]]
                word_type_counts[word_types.index(type)] += 1
                feature_data.append([word_types.index(type), instance])
    feature_data = np.array(feature_data)

    total_trues = 0
    for n in word_type_counts:
        total_trues += comb(n, 2)
    return feature_data, total_trues


def init_centroids(k, feature_data):
    num_instances = len(feature_data)
    num_features = len(feature_data[0][1])
    centroids = np.zeros((k, num_features))

    for k in range(k):
        for i in range(num_features):
            centroids[k][i] = feature_data[np.random.randint(0, num_instances)][1][np.random.randint(0, num_features)]
    return centroids


# Define method to calculate distance between word instance and centroid. Takes centroids and feature data as params.
def calc_dist(centroids, feature_data):
    distances = np.zeros((len(feature_data), len(centroids)))

    # For each word instance, compute distance between word and each cluster centroid, using Cosine Similarity metric
    for i in range(len(feature_data)):
        for k in range(len(centroids)):
            distances[i][k] = cosine(feature_data[i][1], centroids[k])
    return distances


def assign_cluster(feature_data, distance_data):
    assigned_data = []
    for i in range(len(distance_data)):
        cluster_id = closest_cluster(distance_data[i])
        assigned_data.append([cluster_id, feature_data[i][1], feature_data[i][0]])
    assigned_data = np.array(assigned_data)
    return assigned_data


def update_centroids(old_centroids, assigned_data):
    new_centroids = np.zeros(old_centroids.shape)

    for k in range(len(new_centroids)):
        means = np.zeros(len(new_centroids[0]))
        for instance in range(len(assigned_data)):
            if assigned_data[instance][0] == k:
                means = np.mean(assigned_data[instance][1], axis=0)
        new_centroids[k] = means

    for k in range(len(new_centroids)):
        for f in range(len(new_centroids[0])):
            if new_centroids[k][f] == 0:
                new_centroids[k][f] = assigned_data[np.random.randint(0, len(new_centroids))][1][
                    np.random.randint(0, len(new_centroids[0]))]
    return new_centroids


def evaluate(num_clusters, assigned_data, total_trues):
    true_positives = 0
    total_positives = 0

    for k in range(num_clusters):
        total_positives_cluster = 0
        true_positives_array = [0, 0, 0, 0]
        for i in range(len(assigned_data)):
            if assigned_data[i][0] == k:
                total_positives_cluster += 1
            for j in (range(len(true_positives_array))):
                if assigned_data[i][0] == k and assigned_data[i][2] == j:
                    true_positives_array[j] += 1

        true_positive_array = [round(x) for x in true_positives_array]
        true_positives += comb(max(true_positive_array), 2)

        total_positives_cluster = round(total_positives_cluster)
        total_positives += comb(total_positives_cluster, 2)

    precision = true_positives / total_positives
    recall = true_positives / total_trues
    f_score = (2 * precision * recall) / (precision + recall)

    print("For", num_clusters, "Clusters:", "\tPrecision:", precision, "Recall:", recall, "F-Score:", f_score, "\n")
    return precision, recall, f_score


def fit(k, features, total_trues):
    centroids = init_centroids(k, features)
    fit = False
    iterations = 0
    while not fit:
        iterations += 1
        distances = calc_dist(centroids, features)
        assigned = assign_cluster(features, distances)
        new_centroids = update_centroids(centroids, assigned)
        if iterations % 50 == 0:
            print("Fitting...")

        if np.array_equal(centroids, new_centroids):
            fit = True
            print("Model converged after", iterations, "iterations")

        if iterations == 200:
            fit = True
            print("Fitting stopped after", iterations, "iterations")
        centroids = new_centroids

    precision, recall, f_score = evaluate(k, assigned, total_trues)

    return precision, recall, f_score


def normalize(x, vector):
    return x / np.linalg.norm(vector)


def closest_cluster(dist_array):
    return np.argmin(dist_array)


# Define method to compute cosine similarity of two vectors a and b, using numpy dot product and linalg norm
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == '__main__':
    features, total_trues = import_data()

    precisions = []
    recalls = []
    f_scores = []

    print("Fitting unnormalized feature data...")
    for k in range(1, 11):
        precision, recall, f_score = fit(k, features, total_trues)
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
    print("\nPrecisions:", precisions)
    print("Recalls:", recalls)
    print("F-Scores:", f_scores)

    x = range(1, 11)
    plt.plot(x, precisions, label="Precision")
    plt.plot(x, recalls, label="Recall")
    plt.plot(x, f_scores, label="F-Score")
    plt.xlabel("K Value")
    plt.title('K-means Clustering - Cosine Similarity')
    plt.legend()
    plt.show()