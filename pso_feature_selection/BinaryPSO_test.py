# Import modules
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import chi2
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest

from pso_feature_selection.read_csv_file import read_spect_dataset
from pso_feature_selection.BinaryPSO import BinaryPSO as BPSO
from pso_feature_selection.BinaryMutatedPSO import BinaryMutatedPSO as BMPSO


n_features = 50
n_informative = 15
n_redundant = 0
n_classes = 3
X_train, y_train = make_classification(n_samples=200, n_features=n_features, n_classes=n_classes,
                           n_informative=n_informative, n_redundant=n_redundant, n_repeated=0,
                           random_state=1)
X_test = X_train
y_test = y_train

# X_train, y_train = read_spect_dataset(train=True)
# X_test, y_test = read_spect_dataset(train=False)
# n_features = len(X_train[0])

# Create an instance of the classifier
classifier = linear_model.LogisticRegression()
classifier1 = linear_model.LogisticRegression()
classifier1.fit(X_train, y_train)
subset_performance = (classifier1.predict(X_test) == y_test).mean()
print('Subset performance: %.3f' % subset_performance)


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
        X_subset = X_train
    else:
        X_subset = X_train[:, m == 1]

    # Perform classification and store performance in P
    classifier.fit(X_subset, y_train)
    P = (classifier.predict(X_subset) == y_train).mean()
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


def f_entropy(x, alpha=0.8):
    n_particles = x.shape[0]
    f_entropy_value = np.ndarray(n_particles)

    for particle_index in range(n_particles):
        # compute the D1 value for the current particle
        D1 = 0
        for feature_index in range(n_features):
            # if the feature is selected by the current particle
            if x[particle_index][feature_index] == 1:
                # compute the mutual information between feature and class
                p_x_c = sklearn.metrics.normalized_mutual_info_score(X_train[:, feature_index], y_train)
                D1 += p_x_c

        # compute the R1 value for the current particle
        R1 = 0
        for feature_index_1 in range(n_features-1):
            for feature_index_2 in range(feature_index_1 + 1, n_features):
                # if the both features are selected by the current particle
                if x[particle_index][feature_index_1] == 1 and x[particle_index][feature_index_2] == 1:
                    # compute the mutual information between features
                    p_x_x = sklearn.metrics.normalized_mutual_info_score(X_train[:, feature_index_1], X_train[:, feature_index_2])
                    R1 += p_x_x

        # normalize
        R1 = R1 / (n_features * n_features)  # redundancy between each pair of features
        D1 = D1 / n_features  # mutual information between feature and label

        f_particle_entropy_value = abs(alpha*R1 - (1-alpha)*D1)
        f_entropy_value[particle_index] = f_particle_entropy_value

    return np.array(f_entropy_value)


def f_entropy_combined(x, alpha1=0.80, alpha2=0.88, beta=0.8):
    f_entropy_combined_value = beta * f(x, alpha1) + (1-beta) * f_entropy(x, alpha2)
    return np.array(f_entropy_combined_value)


# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 15, 'p':2}

# Call instance of PSO
dimensions = n_features # dimensions should be the number of features


optimizer_BPSO = BPSO(n_particles=30, dimensions=dimensions, options=options)
optimizer_BMPSO = BMPSO(n_particles=30, dimensions=dimensions, options=options)


def perform_optimization(optimizer, f):
    # Perform optimization
    cost, pos = optimizer.optimize(f, print_step=1, iters=10, verbose=2)

    # Create two instances of LogisticRegression
    cl1 = linear_model.LogisticRegression()

    # Get the selected features from the final positions
    X_selected_features_train = X_train[:, pos == 1]  # subset
    X_selected_features_test = X_test[:, pos == 1]  # subset

    cl1.fit(X_selected_features_train, y_train)

    # Compute performance
    subset_performance = (cl1.predict(X_selected_features_test) == y_test).mean()

    print('Subset performance: %.3f\n\n' % subset_performance)

# perform_optimization(optimizer_BPSO, f)
# perform_optimization(optimizer_BMPSO, f)

# perform_optimization(optimizer_BPSO, f_entropy)
# perform_optimization(optimizer_BMPSO, f_entropy)

perform_optimization(optimizer_BPSO, f_entropy_combined)
# perform_optimization(optimizer_BMPSO, f_entropy_combined)

# Tree-based feature selection
clf = ExtraTreesClassifier()
clf = clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X_test)
print(X_new.shape)
cl2 = linear_model.LogisticRegression()
cl2.fit(X_new, y_test)
subset_performance = (cl2.predict(X_new) == y_test).mean()
print('Tree-based subset performance: %.3f\n\n' % subset_performance)


# X_new = SelectKBest(chi2, k=15).fit_transform(X, y)
# print(X_new.shape)
# cl2 = linear_model.LogisticRegression()
# cl2.fit(X_new, y)
# subset_performance = (cl2.predict(X_new) == y).mean()
# print('Chi2 subset performance: %.3f' % subset_performance)
