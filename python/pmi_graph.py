__author__ = 'brandonkelly'

import numpy as np


class PMIGraph(object):

    def __init__(self, nprior=1.0):
        self.nprior = nprior
        self.train_pairs = None
        self.train_marginal = None
        self.joint_probs = None

    def fit(self, X):
        pass

    def predict(self, X):
        pass