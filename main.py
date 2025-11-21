import numpy as np
# If you have a k_means.py file, uncomment the line below:
# from k_means import kmeans
from em import em
# Assuming utils.py exists or you will create it based on your next prompt
from utils import load_data, plot_original_data, plot_mickey_mouse, plot_objective_function


def task_em(X):
    """
    Wrapper function to run the EM algorithm and plot results.
    """
    print("Running EM Algorithm...")

    # Settings for Mickey Mouse data (usually K=3: Head + 2 Ears)
    K = 3
    max_iter = 50

    # Run EM
    means, soft_clusters, log_likelihood = em(X, K, max_iter)

    # Plot results
    print("Plotting results...")
    plot_mickey_mouse(X, means, soft_clusters)
    plot_objective_function(log_likelihood)


def main():
    # Load Data
    # Ensure data/mouse.txt exists or change path
    x, y, data_labels = load_data(filename='data/mouse.txt')

    # Plot raw data
    plot_original_data(x, y, data_labels)

    # Prepare data for EM (N, 2)
    X_mouse = np.array([x, y]).T
    print('X_mouse shape:', X_mouse.shape)


    # ----- Task EM -----
    print('--- Task EM ---')
    task_em(X_mouse)


if __name__ == '__main__':
    main()