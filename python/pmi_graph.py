__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV
from sklearn import manifold
from sklearn.cluster import AffinityPropagation
from scipy.special import gammaln
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def binomial_loglik(graph, X):
    nsamples, nfeatures = X.shape
    npairs = graph.predict(X)
    loglik = 0.0
    for j in xrange(nfeatures):
        this_loglik = gammaln(nsamples + 1) - gammaln(npairs[j, j + 1:] + 1) - gammaln(nsamples - npairs[j, j + 1:] + 1)
        this_loglik += npairs[j, j + 1:] * np.log(graph.joint_probs[j, j + 1:]) + \
            (nsamples - npairs[j, j + 1:]) * np.log(1.0 - graph.joint_probs[j, j + 1:])
        loglik += np.sum(this_loglik)

    return loglik


class PMIGraph(BaseEstimator):
    def __init__(self, nprior=1.0, verbose=False):
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
        if not np.any(np.logical_or(X == 1, X == 0)):
            raise ValueError("Only values of 0 or 1 allowed for input feature matrix.")
        nsamples, nfeatures = X.shape

        # first get number of times each feature occured in a sample
        self.train_marginal = X.sum(axis=0).astype(np.int) + 1

        # now find number of times a pair of features appeared in a sample
        self.train_pairs = np.zeros((nfeatures, nfeatures), dtype=int)
        self.joint_probs = np.zeros((nfeatures, nfeatures))
        self.pmi = np.zeros((nfeatures, nfeatures))
        if self.verbose:
            print 'Finding pairs involving column:'
        for j in xrange(nfeatures - 1):
            if self.verbose:
                print '  ', j + 1, '...'
            active_set1 = X[:, j] == 1
            for k in xrange(j + 1, nfeatures):
                active_set2 = X[:, k] == 1
                pairs = np.logical_and(active_set1, active_set2)
                self.train_pairs[j, k] = np.sum(pairs)
                # prior probability is equal to that expected assuming statistical independence
                marginal_prob1 = self.train_marginal[j] / float(nsamples + 1)
                marginal_prob2 = self.train_marginal[k] / float(nsamples + 1)
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
        npairs = np.zeros((nfeatures, nfeatures), dtype=int)
        if self.verbose:
            print 'Counting pairs for column:'
        # find number of times a pair of features appeared in a sample
        for j in xrange(nfeatures - 1):
            if self.verbose:
                print '  ', j + 1, '...'
            active_set1 = X[:, j] == 1
            for k in xrange(j + 1, nfeatures):
                active_set2 = X[:, k] == 1
                pairs = np.logical_and(active_set1, active_set2)
                npairs[j, k] = np.sum(pairs)

        # make symmetric
        npairs += npairs.T

        return npairs

    def cluster(self, normalize=False):
        if normalize:
            # use normalized PMI for similarity metric
            similarity = self.pmi / -np.log(self.joint_probs)
            similarity[np.diag_indices_from(similarity)] = 1.0
        else:
            similarity = self.pmi
            similarity[np.diag_indices_from(similarity)] = 1.1 * similarity.max()
        clustering = AffinityPropagation(affinity='precomputed', verbose=self.verbose,
                                         preference=similarity.min())
        clusters = clustering.fit_predict(similarity)
        if self.verbose:
            print 'Found', len(np.unique(clusters)), 'clusters.'

        return clusters

    def visualize(self, cluster=False, savefile=None, doshow=True, seed=None, node_labels=None, label_idx=None):
        if node_labels is None:
            node_labels = []
        if label_idx is None:
            label_idx = []
        if len(label_idx) != len(node_labels):
            raise ValueError("Length of node_labels must be the same as label_idx.")

        # use normalized PMI for similarity metric
        similarity = self.pmi / -np.log(self.joint_probs)
        similarity[np.diag_indices_from(similarity)] = 1.0

        # compute the 2-d manifold and the projection of the data onto it. this defines the node positions
        distance = -(similarity - 1.0)  # convert to [-2.0, 0.0] and then make positive
        node_position_model = manifold.TSNE(verbose=self.verbose, metric='precomputed', learning_rate=100,
                                            random_state=seed)
        node_positions = node_position_model.fit_transform(distance).T

        if cluster:
            # also include cluster information in the visualization
            clusters = self.cluster(normalize=True)

        plt.figure(1, facecolor='k', figsize=(10, 8))
        plt.clf()
        ax = plt.axes([0., 0., 1., 1.])
        plt.axis('off')

        # Plot the nodes using the coordinates of our embedding
        base_symbol_size = self.train_marginal / float(self.train_marginal.max()) + 0.05
        if cluster:
            # color ingredient nodes by cluster
            plt.scatter(node_positions[0], node_positions[1], s=300 * base_symbol_size, c=clusters,
                        cmap=plt.cm.spectral_r)
        else:
            plt.scatter(node_positions[0], node_positions[1], s=300 * base_symbol_size,
                        cmap=plt.cm.spectral_r, c='DodgerBlue')

        # Display a graph of ingredients commonly found together based on pointwise mutual information (PMI)
        non_zero = np.triu(similarity, k=1) > np.percentile(similarity[similarity > 0], 95.0)

        start_idx, end_idx = np.where(non_zero)
        #a sequence of (*line0*, *line1*, *line2*), where::
        #            linen = (x0, y0), (x1, y1), ... (xm, ym)
        segments = [[node_positions[:, start], node_positions[:, stop]]
                    for start, stop in zip(start_idx, end_idx)]
        values = similarity[non_zero]
        lc = LineCollection(segments,
                            zorder=0, cmap=plt.cm.hot,
                            norm=plt.Normalize(values.min(), np.percentile(values, 95.0)))
        lc.set_array(values)
        lc.set_linewidths(2 * values)
        ax.add_collection(lc)
        # plt.colorbar(lc)

        for label, node_idx in zip(node_labels, label_idx):
            plt.text(node_positions[0, node_idx], node_positions[1, node_idx], label, size=14, color='White')

        plt.xlim(node_positions[0].min() - .15 * node_positions[0].ptp(),
                 node_positions[0].max() + .10 * node_positions[0].ptp(),)
        plt.ylim(node_positions[1].min() - .03 * node_positions[1].ptp(),
                 node_positions[1].max() + .03 * node_positions[1].ptp())

        if savefile is not None:
            plt.savefig(savefile)
        if doshow:
            plt.show()


def fit_pmi_graph_cv(X, verbose=False, n_jobs=1, cv=7, doplot=False, graph=None):
    nsamples, nfeatures = X.shape
    if graph is None:
        graph = PMIGraph()
    param_grid = {'nprior': np.logspace(0.0, np.log10(100 * nsamples), 20)}  # the prior sample sizes
    grid_search = GridSearchCV(graph, param_grid, scoring=binomial_loglik, n_jobs=n_jobs, cv=cv, verbose=verbose).fit(X)

    if verbose:
        print 'Used cross-validation to choose a prior sample size of', grid_search.best_params_['nprior']

    if doplot:
        pgrid = []
        mean_score = []
        for grid_score in grid_search.grid_scores_:
            pgrid.append(grid_score[0]['nprior'])
            mean_score.append(grid_score[1])
        plt.plot(pgrid, mean_score, '-')
        plt.semilogx(pgrid, mean_score, 'o')
        plt.xlabel('nprior')
        plt.ylabel('Binomial loglik')
        plt.semilogx(2 * [grid_search.best_params_['nprior']], plt.ylim(), 'k-')
        plt.show()

    return grid_search.best_estimator_


if __name__ == "__main__":
    # do quick test on the class assuming the feature events are independent
    nsamples = 1000
    nfeatures = 100
    mprob = np.random.uniform(0.0, 1.0, nfeatures)
    jprob = np.zeros((nfeatures, nfeatures))
    pmi = np.zeros((nfeatures, nfeatures))
    for j in range(nfeatures - 1):
        for k in range(j + 1, nfeatures):
            jprob[j, k] = mprob[j] * mprob[k]
            pmi[j, k] = np.log(jprob[j, k]) - np.log(mprob[j]) - np.log(mprob[k])

    X = np.zeros((nsamples, nfeatures), dtype=int)
    for i in range(nsamples):
        bern_trials = np.random.uniform(0.0, 1.0, nfeatures) < mprob
        X[i] = bern_trials

    npairs_true = np.zeros((nfeatures, nfeatures), dtype=int)
    for i in xrange(nsamples):
        for j in xrange(nfeatures - 1):
            if X[i, j] == 1:
                for k in xrange(j + 1, nfeatures):
                    if X[i, k] == 1:
                        npairs_true[j, k] += 1

    npairs_true += npairs_true.T
    # first test when the prior is negligible
    graph = PMIGraph(verbose=True).fit(X)
    assert np.all(npairs_true == graph.train_pairs)
    assert np.all(X.sum(axis=0) == graph.train_marginal)
    print "True PMI:"
    print pmi.astype(np.float16)

    print "Estimated PMI:"
    print graph.pmi.astype(np.float16)

    npairs = graph.predict(X)
    assert np.all(npairs_true == npairs)

    # now test for case when prior dominates
    graph = PMIGraph(nprior=1e6, verbose=True).fit(X)
    assert np.all(npairs_true == graph.train_pairs)
    print "True PMI:"
    print pmi

    print "Estimated PMI:"
    print graph.pmi

    npairs = graph.predict(X)
    assert np.all(npairs_true == npairs)

    # finally, do cross-validation
    graph = fit_pmi_graph_cv(X, verbose=True, n_jobs=7, cv=7, doplot=True)
    print 'CV PMI:'
    print graph.pmi