from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.array([np.sum([y==k]) for k in self.classes_]) / y.shape[0]
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        map_class_ind = {}
        for i in range(self.classes_.shape[0]):
            self.mu_[i] = np.mean(X[y==self.classes_[i]], axis=0 )
            map_class_ind[self.classes_[i]] = i
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        X_minus_mu = np.array([X[i]- self.mu_[map_class_ind[y[i]]] for i in range(X.shape[0])])
        for x_i in X_minus_mu:
            self.cov_ += np.outer(x_i, x_i)
        # to un-bias estimator, derive by 1/(m-k)
        self.cov_ /= (X.shape[0]-self.classes_.shape[0])
        self._cov_inv = np.linalg.inv(self.cov_)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihoods = self.likelihood(X)
        # to each row (sample) return the index of the class that give the maximum likelihood
        X_index_class = np.argmax(likelihoods, axis=1)
        return self.classes_[X_index_class]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihoods = np.zeros((X.shape[0], self.classes_.shape[0]))
        n_features = X.shape[1]
        for i in range(self.classes_.shape[0]):
            mult = np.sum((X - self.mu_[i]) @ inv(self.cov_) * (X - self.mu_[i]), axis=1)
            likelihoods[:,i] = (1 / np.sqrt(np.power(2 * np.pi, n_features) * det(self.cov_))) * \
            np.exp(-0.5 * mult)  * self.pi_[i]
        return likelihoods

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))