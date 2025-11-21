import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """
    Parses the specific mouse.txt format provided in the assignment.
    """
    with open(filename) as f:
        content = f.readlines()

    # Split lines by spaces and flatten the list
    content = [x.strip().split(' ') for x in content]
    lst_content = sum(content, [])

    # Parse using stride of 3 (x, y, label)
    x_str = lst_content[0::3]
    y_str = lst_content[1::3]
    data_labels = np.array(lst_content[2::3])

    x = np.array([float(i) for i in x_str])
    y = np.array([float(i) for i in y_str])

    return x, y, data_labels


def plot_original_data(x, y, data_labels):
    """
    Plots the raw data with ground truth labels.
    """
    labels = ['Head', 'Ear_left', 'Ear_right', 'Noise']
    unique_labels = np.unique(data_labels)

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('Original data (Ground Truth)')
    ax = fig.add_subplot(111)

    ax.set(xlabel='x', ylabel='y')

    # Handle cases where not all labels (like 'Noise') are present in the data
    for lbl in labels:
        if lbl in unique_labels:
            idxs = np.where(data_labels == lbl)[0]
            ax.scatter(x[idxs], y[idxs], label=lbl, alpha=0.6)

    ax.legend()
    plt.savefig('original_data.jpg')
    plt.show()


def plot_mickey_mouse(X, means, soft_clusters):
    """
    Plots the results of the EM clustering.
    Adapted to accept:
    - X: Data points (N, 2)
    - means: The centroids found by EM (K, 2)
    - soft_clusters: The responsibility matrix (N, K)
    """
    K = means.shape[0]
    # Convert soft responsibilities to hard assignments for coloring
    clusters = np.argmax(soft_clusters, axis=1)

    x, y = X[:, 0], X[:, 1]

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('EM Clustered data points')
    ax = fig.add_subplot(111)

    ax.set(xlabel='x', ylabel='y')

    # Plot points colored by cluster
    colors = plt.cm.viridis(np.linspace(0, 1, K))
    for i in range(K):
        idxs = np.where(clusters == i)[0]
        ax.scatter(x[idxs], y[idxs], color=colors[i], alpha=0.5, label=f'Cluster {i + 1}')

    # Plot Centroids
    for i in range(K):
        ax.scatter(means[i, 0], means[i, 1], c='red', marker='x', s=150, linewidths=3,
                   label='Centroid' if i == 0 else "")

    plt.legend()
    plt.savefig("scatter_EM.jpg")
    plt.show()


def plot_objective_function(log_likelihood):
    """
    Plots the Log-Likelihood over iterations.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(log_likelihood, marker='o', linestyle='-')
    plt.title('Log-Likelihood vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.savefig("EM_log_likelihood.jpg")
    plt.show()