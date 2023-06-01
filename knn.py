import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        
        # majority vote, most commen class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))