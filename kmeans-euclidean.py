# This Python script is an implementation of the K-Means clustering algorithm. In this file, Euclidean distance is used
# to update the cluster means. Unnormalized / normalized data is used as the input, a graph is produced comparing the
# Precision, Recall and F-Score for K values from 1-10, for both the unnormalized and normalized inputs.

# Written by Matthew Rothery - 201028655

# Import numpy, combination formula nCr, and pyplot
import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt


# Define method to import feature data from word file documents
def import_data():
    print("Importing data...")
    # Tuple of different types of words
    word_types = ('animals', 'countries', 'fruits', 'veggies')
    # Initialize array to store count of words for each word type
    word_type_counts = np.zeros(len(word_types))

    feature_data = []
    # For each type of word
    for type in word_types:
        # Open file of that word type
        with open(type) as file:
            # For each line in that word type file
            for line in file:
                # Strip any blanks and split by spaces between each feature
                instance = line.strip().split(' ')
                # Convert each feature value from string to float
                instance = [float(x) for x in instance[1:]]
                # Add 1 to count for that word type
                word_type_counts[word_types.index(type)] += 1
                # Append word type value and instance array of feature values to feature data array
                feature_data.append([word_types.index(type), instance])
    # Convert feature data array to numpy array
    feature_data = np.array(feature_data)

    # For normalized data, make a new array from the old feature data
    normalized_feature_data = np.array(feature_data)
    # For each value in array, run through normalize method
    for i in range(len(feature_data)):
        for j in range(len(feature_data[i][1])):
            normalized_feature_data[i][1][j] = normalize(feature_data[i][1][j], feature_data[i][1])

    # Calculate total trues of data set by calling combination function for each word type, and sum up
    total_trues = 0
    for n in word_type_counts:
        total_trues += comb(n, 2)
    return feature_data, normalized_feature_data, total_trues


# Define method to initialize centroids (cluster centres). Takes parameters of k value and feature data
def init_centroids(k, feature_data):
    num_instances = len(feature_data)
    num_features = len(feature_data[0][1])
    centroids = np.zeros((k, num_features))

    # For each cluster, pick a random set of features and store as centroid
    for k in range(k):
        for i in range(num_features):
            centroids[k][i] = feature_data[np.random.randint(0, num_instances)][1][np.random.randint(0, num_features)]
    return centroids


# Define method to calculate distance between word instance and centroid. Takes centroids and feature data as params.
def calc_dist(centroids, feature_data):
    distances = np.zeros((len(feature_data), len(centroids)))

    # For each word instance, compute distance between word and each cluster centroid, using Euclidean metric
    for i in range(len(feature_data)):
        for k in range(len(centroids)):
            distances[i][k] = euclidean(feature_data[i][1], centroids[k])
    return distances


# Define method to assign each word instance to the closest cluster. Takes feature data and distances data as params.
def assign_cluster(feature_data, distance_data):
    assigned_data = []

    # For each word instance, call closest_cluster method on the calculated distances to each cluster
    for i in range(len(distance_data)):
        cluster_id = closest_cluster(distance_data[i])

        # Append to assigned_data array the closest cluster id, array of features and actual label
        assigned_data.append([cluster_id, feature_data[i][1], feature_data[i][0]])
    assigned_data = np.array(assigned_data)
    return assigned_data


# Define method to update centroids based on the mean of the word instances
# Takes centroids and assigned word instances as parameters
def update_centroids(old_centroids, assigned_data):
    new_centroids = np.zeros(old_centroids.shape)

    # For each cluster, declare a new array to store feature means
    for k in range(len(new_centroids)):
        means = np.zeros(len(new_centroids[0]))
        # For each word instance
        for instance in range(len(assigned_data)):
            # If the word instance is assigned to the current cluster
            if assigned_data[instance][0] == k:
                # Then set the means array to be the mean of each feature element assigned to that cluster
                means = np.mean(assigned_data[instance][1], axis=0)
        # Set new centroids of that cluster to be means of features assigned to that cluster
        new_centroids[k] = means

    # Now, for each element of the new_centroids
    for k in range(len(new_centroids)):
        for f in range(len(new_centroids[0])):
            # If the element was equal to 0 (as in not closest to that cluster)
            if new_centroids[k][f] == 0:
                # Pick a new random feature value to be the new_centroids value
                new_centroids[k][f] = assigned_data[np.random.randint(0, len(new_centroids))][1][
                    np.random.randint(0, len(new_centroids[0]))]
    return new_centroids


# Define method to evaluate performance of clustering algorithm.
# Takes k value, assigned word instances and total trues of input data as parameters
def evaluate(num_clusters, assigned_data, total_trues):
    # Declare true positives and total positives variables as 0
    true_positives = 0
    total_positives = 0

    # For each cluster, declare total positives variable and list to store true positives for each word type
    for k in range(num_clusters):
        total_positives_cluster = 0
        true_positives_array = [0, 0, 0, 0]
        # For each word instance
        for i in range(len(assigned_data)):
            # If the cluster id of that word instance is the same as the current cluster
            if assigned_data[i][0] == k:
                # Add 1 to total positives for that cluster
                total_positives_cluster += 1
            # For each word type
            for j in (range(len(true_positives_array))):
                # If the cluster id is the same as the current cluster, and the label is the same as the word type
                if assigned_data[i][0] == k and assigned_data[i][2] == j:
                    # Add 1 to the true positives list for that word type
                    true_positives_array[j] += 1
        # Round any non integers, and call combination method on the most common word type for that cluster
        true_positive_array = [round(x) for x in true_positives_array]
        true_positives += comb(max(true_positive_array), 2)

        # Round any non integers, and call combination method on total positives for the cluster
        total_positives_cluster = round(total_positives_cluster)
        total_positives += comb(total_positives_cluster, 2)

    # Define precision, recall and f-score from true positives, total positives and total trues
    precision = true_positives / total_positives
    recall = true_positives / total_trues
    f_score = (2 * precision * recall) / (precision + recall)

    # Print to console evaluation result for that cluster
    print("For", num_clusters, "Clusters:", "\tPrecision:", precision, "Recall:", recall, "F-Score:", f_score, "\n")
    return precision, recall, f_score


# Define fit method, which will be called each time the clustering algorithm is run
# Takes k value, feature data and total trues of the feature data as parameters
def fit(k, features, total_trues):
    # Initialize centroids by calling init_centroids method
    centroids = init_centroids(k, features)
    # Declare fit = False (end state) and number of iterations
    fit = False
    iterations = 0
    # While the algorithm hasnt fit to the feature data
    while not fit:
        # Increase iterations by 1
        iterations += 1
        # Calculate distances from centroids to features (by calling calc_dist method)
        distances = calc_dist(centroids, features)
        # Assign each word instance to nearest cluster (by calling assign_cluster method)
        assigned = assign_cluster(features, distances)
        # Update centroids with means of features that are assigned to that cluster (by calling update_centroids)
        new_centroids = update_centroids(centroids, assigned)

        # Print to console that it is still fitting, every 50 iterations
        if iterations % 50 == 0:
            print("Fitting...")

        # If the new centroids are equal to the old centroids, then algorithm has converged. Print to console
        if np.array_equal(centroids, new_centroids):
            fit = True
            print("Model converged after", iterations, "iterations")

        # If the iterations reach 200 (max), then end fitting and print to console
        if iterations == 200:
            fit = True
            print("Fitting stopped after", iterations, "iterations")
        # Set old centroids to the newly updated centroids
        centroids = new_centroids
    # Generate precision, recall and f-score for the k value by calling the evaluate method
    precision, recall, f_score = evaluate(k, assigned, total_trues)
    return precision, recall, f_score


# Define a method to normalize each feature of input data, using numpy linalg norm function.
def normalize(x, vector):
    return x / np.linalg.norm(vector)


# Define method to calculate closest cluster to a word, by finding argmin of the array of distances to each cluster
def closest_cluster(dist_array):
    return np.argmin(dist_array)


# Define method to calculate euclidean distance of 2 vectors a and b, using numpy linalg norm function
def euclidean(a, b):
    return np.linalg.norm((a-b))

# When program is run
if __name__ == '__main__':
    # Call import data method
    features, normalized_features, total_trues = import_data()

    # Instantiate blank lists to store evaluation data for both unnormalized and normalized input
    precisions = []
    recalls = []
    f_scores = []
    precisions_norm = []
    recalls_norm = []
    f_scores_norm = []

    # Perform fitting of unnormalized input
    print("Fitting unnormalized feature data...")
    # For values of K = 1 - 10
    for k in range(1, 11):
        # Call fit method, store output of evaluation values
        precision, recall, f_score = fit(k, features, total_trues)
        # Append evaluation values to appropriate lists
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
    # Once fit for each K value, output lists
    print("\nPrecisions:", precisions)
    print("Recalls:", recalls)
    print("F-Scores:", f_scores)

    # Perform fitting of normalized input
    print("\nFitting normalized feature data...")
    # For values of K = 1 - 10
    for k in range(1, 11):
        # Call fit method, store output of evaluation values
        precision, recall, f_score = fit(k, normalized_features, total_trues)
        # Append evaluation values to appropriate lists
        precisions_norm.append(precision)
        recalls_norm.append(recall)
        f_scores_norm.append(f_score)
    # Once fit for each K value, output lists
    print("\nPrecisions (Normalised):", precisions_norm)
    print("Recalls (Normalised):", recalls_norm)
    print("F-Scores (Normalised):", f_scores_norm)

    # Plot both unnormalized and normalized inputs, showing precision, recall and f-score as function of K value
    x = range(1, 11)
    plt.subplot(1, 2, 1)
    plt.plot(x, precisions, label="Precision")
    plt.plot(x, recalls, label="Recall")
    plt.plot(x, f_scores, label="F-Score")
    plt.xlabel("K Value")
    plt.title('Unnormalized')

    plt.subplot(1, 2, 2)
    plt.plot(x, precisions_norm, label="Precision")
    plt.plot(x, recalls_norm, label="Recall")
    plt.plot(x, f_scores_norm, label="F-Score")
    plt.xlabel("K Value")
    plt.title('Normalized')

    plt.suptitle("K-means Clustering - Euclidean Distance", size=16)
    plt.legend()
    plt.show()