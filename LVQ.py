
import numpy as np
import logging
import random
import sklearn
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import sys
from sklearn.metrics import accuracy_score

# Set up a specific logger with our desired output level
my_logger = logging.getLogger('LVQ')
my_logger.setLevel(logging.INFO)

# Add the log message handler to the logger
handler = logging.StreamHandler()
my_logger.addHandler(handler)



def euclidean_distance(vect1, vect2):
    dist = np.linalg.norm(vect1 - vect2)
    return dist

# Codebook Init Functions

# distance functions
distance_functions = {
    "euclidean" : euclidean_distance
}

# create a single random codeboook
def random_codebook(train, rnd=None):
    n_records = train.shape[0]
    n_features = train.shape[1]
    my_logger.debug("Creating random Codebook with %s records and %s features" % (n_records, n_features))

    codebook = np.zeros(shape=(1, n_features))
    for i in range(n_features):
        my_logger.debug('Codebook: %s' % codebook)

        random_sample = train[random.randrange(n_records)]
        my_logger.debug("Random Sample: %s" % random_sample)

        random_feature = random_sample[i]
        my_logger.debug("Setting Feature %s to %s" % (i, random_feature))

        codebook[0, i] = random_feature

    my_logger.debug('Created Random Codebook: %s' % codebook)

    return codebook

def random_class_codebook(train, rnd=None, filterclass = None):
    n_records = train.shape[0]
    n_features = train.shape[1]
    my_logger.debug("Creating random class Codebook with %s records and %s features" % (n_records, n_features))

    codebook = np.zeros(shape=(1, n_features))

    if not filterclass:
        random_sample = train[random.randrange(n_records)]
        filterclass = random_sample[-1]

    train_class = [sample for sample in train if sample[-1] == filterclass]

    for i in range(n_features):
        my_logger.debug('Codebook: %s' % codebook)

        random_sample = train_class[random.randrange(len(train_class))]
        my_logger.debug("Random Sample: %s" % random_sample)

        random_feature = random_sample[i]
        my_logger.debug("Setting Feature %s to %s" % (i, random_feature))

        codebook[0, i] = random_feature

    my_logger.debug('Created Random Codebook: %s' % codebook)

    return codebook

# codebook init functions
codebook_inits = {
    "random" : random_codebook,
    "class": random_class_codebook
}
########################################################################################################################
########################################################################################################################
########################################################################################################################

class LVQ(sklearn.base.BaseEstimator):

    def __init__(self, lrate=0.3, epochs=10, n_codebooks=10, distance_func=euclidean_distance, init_codebook=random_codebook, rnd_seed=0):
        self.lrate = lrate
        self.epochs = epochs
        self.n_codebooks = n_codebooks
        self.distance_func = distance_func
        self.init_codebook = init_codebook
        self.rnd_seed = rnd_seed

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Set global random seed
        random.seed(self.rnd_seed)

        reshaped_y = np.array(y, copy=False, subok=True, ndmin=2).T
        training_set = np.concatenate((X, reshaped_y), axis=1)

        self.codebooks = self._train_codebooks(training_set, self.n_codebooks, self.lrate, self.epochs, self.init_codebook)

        return self

    def predict(self, X):

        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        return [self._get_bmu(self.codebooks, x)[-1] for x in X]

    def score(self, X, y):
        predictions = self.predict(X)

        return accuracy_score(y, predictions)

    def _get_bmu(self, codebooks, vector_sample):
        code_distances = []

        my_logger.debug("Getting BMU for %s in %s" % (vector_sample, codebooks))
        # we assume that the last element of the vectors is the class
        for codebook in codebooks:
            dist = self.distance_func(codebook[:-1], vector_sample)
            code_distances.append((codebook, dist))


        code_distances.sort(key=lambda tup: tup[1])

        bmu = code_distances[0][0]


        my_logger.debug("BMU is %s" % bmu)
        return bmu

    def _train_codebooks(self, training_matrix, n_codebooks, lrate, epochs, init_codebook):
        num_features = training_matrix.shape[1]

        # create init codebook matrix
        codebooks = np.zeros(shape=(n_codebooks, num_features))
        for i in range(n_codebooks):
            codebooks[i] = init_codebook(training_matrix)
        my_logger.debug('Init-Codebook-Matrix: %s' % codebooks)


        # Epochs: At the top level, the process is repeated for a fixed number of epochs or exposures of the training data.
        for epoch in range(epochs):
            # Within an epoch, each training pattern is used one at a time to update the set of codebook vectors

            rate = lrate * (1.0 - (epoch / float(epochs)))  # linear decay learning rate schedule
            sum_error = 0.0     # helpful when debugging the training function

            for training_pattern in training_matrix:
                # For a given training pattern, each feature of a best matching codebook vector
                # is updated to move it closer or further away.

                bmu = self._get_bmu(codebooks, training_pattern[:-1])  # update only best matching
                my_logger.debug("BMU for %s is %s" % (training_pattern, bmu))

                for feature_num in range(num_features - 1):
                    my_logger.debug('Training Feature Number %s' % feature_num)
                    error = training_pattern[feature_num] - bmu[feature_num]
                    my_logger.debug("Error for bmu_feature %s and training_feature %s is %s" % (bmu[feature_num], training_pattern[feature_num], error))

                    sum_error += error ** 2
                    bmu_class = bmu[-1]
                    pattern_class = training_pattern[-1]
                    if bmu_class == pattern_class:  # test if class value is the same
                        my_logger.debug("BMU and Training Pattern have same class")
                        bmu[feature_num] += rate * error
                    else:
                        my_logger.debug("BMU and Training Pattern have different class")
                        bmu[feature_num] -= rate * error
                my_logger.debug("Trained BMU is %s" % bmu)

            my_logger.debug('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))

        return codebooks