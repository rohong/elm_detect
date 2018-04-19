#!/usr/bin/env python
# --*-- coding: utf-8 --*--
"""
Methods to get the probability of a change-point in an 1D array.
Based on the algorithm introduced in
Ryan P. Adams, David J.C. MacKay, Bayesian Online Changepoint Detection,
arXiv 0710.3742 (2007)

Created on April 18, 2018
Author: Rongjie Hong
"""

from __future__ import division
import numpy as np
from scipy import stats


def detect(data, hazard_func, obs_likelihood):
    """
    Evaluate the growth probabilities
    :param data: input data, an 1D array
    :param hazard_func: hazard function defined below
    :param obs_likelihood: observation likelihood,
                           a student test object defined below
    :return R: growth probabilities
    :return maxes: indices of maximums
    """
    maxes = np.zeros(len(data) + 1)
    R = np.zeros((len(data) + 1, len(data) + 1), dtype=np.float64)
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        pred_probs = obs_likelihood.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t + 1)))

        # Evaluate the growth probabilities - shift the probabilities down
        # and to the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t + 2, t + 1] = R[0:t + 1, t] * pred_probs * (1 - H)

        # Evaluate the probability that there *was* a change-point and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0:t + 1, t] * pred_probs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])

        # Update the parameter sets for each possible run length.
        obs_likelihood.update_theta(x)

        maxes[t] = R[:, t].argmax()

    return R, maxes


def constant_hazard(lam, r):
    return 1 / lam * np.ones(r.shape)


class StudentT:
    """
    Predictive distribution, using exponential statistical models.
    """

    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.asarray([alpha])
        self.beta0 = self.beta = np.asarray([beta])
        self.kappa0 = self.kappa = np.asarray([kappa])
        self.mu0 = self.mu = np.asarray([mu])

    def pdf(self, data):
        """
        Student t distribution
        :param data: Input data, 1D array
        :return: Student distribution, 1D array
        """
        return stats.t.pdf(x=data,
                           df=2 * self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa + 1) /
                                         (self.alpha * self.kappa)))

    def update_theta(self, data):
        """
        Update distribution every time step.
        :param data: Input data, 1D array
        :return: None
        """
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (self.beta0, self.beta + (self.kappa * (data - self.mu) ** 2) /
             (2. * (self.kappa + 1.))))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0
