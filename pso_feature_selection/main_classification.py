import scipy
import pandas as pd
from scipy.io import arff
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from skmultilearn.adapt import MLkNN
from sklearn.cross_validation import train_test_split
from scipy import sparse
from bisect import bisect_left

classes = ['Class1', 'Class2', 'Class3', 'Class4',
           'Class5', 'Class6', 'Class7', 'Class8',
           'Class9', 'Class10', 'Class11', 'Class12',
           'Class13', 'Class14']

X, y = make_multilabel_classification(sparse=True, n_features=20, n_labels=20, return_indicator='sparse', allow_unlabeled=False)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = MLkNN(k=20)

# train
classifier.fit(X_train, Y_train)

# predict
predictions = classifier.predict(X_test)

print accuracy_score(Y_test, predictions)
