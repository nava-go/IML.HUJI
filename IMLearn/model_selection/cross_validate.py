from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds // s not include si

    validation_score: float
        Average validation score over folds // just si
    """


    m_samples = y.shape[0]
    len_fold = m_samples // cv
    sum_error_train, sum_error_validation = 0,0
    for i in range(cv):
        index_to_delete = list(range(i*len_fold, (i+1)*len_fold))
        temp_X = np.delete(X, index_to_delete, axis=0)
        temp_y = np.delete(y, index_to_delete, axis=0)
        estimator.fit(temp_X, temp_y)

        predict_train = estimator.predict(temp_X)
        sum_error_train += scoring(temp_y, predict_train)

        predict_validate = estimator.predict(X[index_to_delete])
        sum_error_validation += scoring(y[index_to_delete], predict_validate)
    return sum_error_train / cv, sum_error_validation/cv





