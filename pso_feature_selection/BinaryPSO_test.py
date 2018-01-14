# Import modules
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.datasets import make_classification

from pso_feature_selection.BinaryPSO import BinaryPSO as BPSO
from pso_feature_selection.BinaryMutatedPSO import BinaryMutatedPSO as BMPSO

n_features = 30
n_informative = 15
n_redundant = 0
n_classes = 3
X, y = make_classification(n_samples=300, n_features=n_features, n_classes=n_classes,
                           n_informative=n_informative, n_redundant=n_redundant, n_repeated=0,
                           random_state=1)


# Create an instance of the classifier
classifier = linear_model.LogisticRegression()


# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = n_features
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j


def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 15, 'p':2}

# Call instance of PSO
dimensions = n_features # dimensions should be the number of features


optimizer_BPSO = BPSO(n_particles=30, dimensions=dimensions, options=options)
optimizer_BMPSO = BMPSO(n_particles=30, dimensions=dimensions, options=options)

def perform_optimization(optimizer):
    # Perform optimization
    cost, pos = optimizer.optimize(f, print_step=100, iters=1000, verbose=2)

    # Create two instances of LogisticRegression
    cl1 = linear_model.LogisticRegression()

    # Get the selected features from the final positions
    X_selected_features = X[:,pos==1]  # subset

    # Perform classification and store performance in P
    cl1.fit(X_selected_features, y)

    # Compute performance
    subset_performance = (cl1.predict(X_selected_features) == y).mean()

    print('Subset performance: %.3f' % subset_performance)

perform_optimization(optimizer_BPSO)
perform_optimization(optimizer_BMPSO)
