__author__ = 'brandonkelly'

import numpy as np


class PMIGraph(object):

    def __init__(self, nprior=1, verbose=False):
        self.nprior = nprior
        self.verbose = verbose
        self.train_pairs = None
        self.train_marginal = None
        self.joint_probs = None
        self.pmi = None

    def fit(self, X):
        """
        Fit the graph using pairwise mutual information for the edge weights.

        :param X: A binary-valued (nsamples, nfeatures) array marking the instances that each feature appears in a
            sample or not. If X[i, j] = 1 then the jth feature appears in the ith sample; otherwise if X[i, j] then the
            jth feature is absent from the ith sample.
        :return: self.
        """
        if np.any(np.logical_or(X == 1, X == 0)):
            raise ValueError("Only values of 0 or 1 allowed for input feature matrix.")
        nsamples, nfeatures = X.shape

        # first get number of times each feature occured in a sample
        self.train_marginal = X.sum(axis=0)

        # now find number of times a pair of features appeared in a sample
        self.train_pairs = np.zeros((nfeatures, nfeatures))
        self.joint_probs = np.zeros((nfeatures, nfeatures))
        self.pmi = np.zeros((nfeatures, nfeatures))
        if self.verbose:
            print 'Finding pairs involving column:'
        for j in xrange(nfeatures-1):
            print '  ', j + 1, '...'
            active_set1 = X[:, j] == 1
            for k in xrange(j + 1, nfeatures):
                active_set2 = X[:, k] == 1
                pairs = np.logical_and(active_set1, active_set2)
                self.train_pairs[j, k] = np.sum(pairs)
                # prior probability is equal to that expected assuming statistical independence
                marginal_prob1 = self.train_marginal[j] / nsamples
                marginal_prob2 = self.train_marginal[k] / nsamples
                prior_prob = marginal_prob1 * marginal_prob2
                # estimated joint probability is posterior expectation under a beta prior
                self.joint_probs[j, k] = (self.train_pairs[j, k] + prior_prob * self.nprior) / (nsamples + self.nprior)
                # compute the pairwise mutual information
                self.pmi[j, k] = np.log(self.joint_probs[j, k]) - np.log(marginal_prob2) - np.log(marginal_prob1)

        # make symmetric
        self.train_pairs += self.train_pairs.T
        self.joint_probs += self.joint_probs.T
        self.pmi += self.pmi.T

        return self

    def predict(self, X):
        """
        Return the number of times each feature pair appeared in a sample for the input array. There is no actual
        prediction that occurs here, but this method is named 'predict' to enable compatibility with scikit-learn's
        GridSearchCV class for using cross-validation to choose the prior sample size.

        :param X: A binary-valued (nsamples, nfeatures) array marking the instances that each feature appears in a
            sample or not. If X[i, j] = 1 then the jth feature appears in the ith sample; otherwise if X[i, j] then the
            jth feature is absent from the ith sample.
        :return: An (nfeatures, nfeatures) array containing the number of times each feature pair appears in a sample.
        """
        nsamples, nfeatures = X.shape
        if nfeatures != len(self.train_marginal):
            raise ValueError("Number of columns of input X array differs from training data.")
        npairs = np.zeros((nfeatures, nfeatures))
        if self.verbose:
            print 'Counting pairs for column:'
        # find number of times a pair of features appeared in a sample
        for j in xrange(nfeatures-1):
            print '  ', j + 1, '...'
            active_set1 = X[:, j] == 1
            for k in xrange(j + 1, nfeatures):
                active_set2 = X[:, k] == 1
                pairs = np.logical_and(active_set1, active_set2)
                npairs[j, k] = np.sum(pairs)

        # make symmetric
        npairs += npairs.T

        return npairs

    def visualize(self):
        pass