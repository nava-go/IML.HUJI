from __future__ import annotations
from typing import Tuple, NoReturn
import sys
sys.path.append("G:\האחסון שלי\year2\semesterB\iml\IML.HUJI")
from IMLearn.base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_error = 1
        for j in range(X.shape[1]):
            for sign in (-1, 1):
                thr, error = self._find_threshold(X[:, j], y, sign)
                if(error<min_error):
                    min_error = error
                    self.threshold_ = thr
                    self.sign_ = sign
                    self.j_ = j


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        col = X[:, self.j_]
        return np.where(col>=self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        ind_sort = np.argsort(values)
        sorted_values = values[ind_sort]
        # we dont make them np.sign becaue we want that the error will
        # be computed relative to the weight
        sorted_labels = labels[ind_sort]
        min_ind, min_error = 0, 1
        temp_y = np.full((values.shape[0],), sign)
        for i in range(values.shape[0] + 1):
            miss_error = np.sum(np.where(temp_y != np.sign(sorted_labels), np.abs(sorted_labels), 0))
            if(miss_error<min_error):
                min_error = miss_error
                min_ind = i
            if(i != values.shape[0]):
                temp_y[i] = -sign
        if(min_ind==values.shape[0]):
            threshold = sorted_values[min_ind-1] + 1
        else:
            threshold = sorted_values[min_ind]
        return (threshold, min_error)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        temp_y = self._predict(X)
        return np.sum(np.where(temp_y != np.sign(y), np.abs(y), 0))

if __name__ == '__main__':
    X = np.array([[10, 2], [20, 1], [30, 1]])
    y = np.array([1, -1, 1])
    ds = DecisionStump()
    ds.fit(X, y)
    y_pred = ds.predict(X)
    print(y_pred)