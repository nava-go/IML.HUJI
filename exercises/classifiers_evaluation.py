import sys
sys.path.append("G:\האחסון שלי\year2\semesterB\iml\IML.HUJI")
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from IMLearn.utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "G:\האחסון שלי\year2\semesterB\iml\IML.HUJI\datasets\linearly_separable.npy"), ("Linearly Inseparable", "G:\האחסון שלי\year2\semesterB\iml\IML.HUJI\datasets\linearly_inseparable.npy")]:
        # Load dataset
        X, Y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def loss_callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit._loss(X, Y))

        Perceptron(callback=loss_callback)._fit(X, Y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(pd.DataFrame({'iteration num': list(range(
            len(losses))), 'loss': losses}), x= "iteration num", y= "loss", title= f'{n}: Loss in function of the Training Iteration')
        fig.show()



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    from IMLearn.metrics import accuracy

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("G:\האחסון שלי\year2\semesterB\iml\IML.HUJI\datasets\\"+f)
        #train_x, train_y, test_x, test_y = split_train_test(X, y)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        y_predict_lda = lda.predict(X)
        accuracy_lda = accuracy(y, y_predict_lda)

        gassian_naive_bayes = GaussianNaiveBayes()
        gassian_naive_bayes.fit(X, y)
        y_predict_gassian = gassian_naive_bayes.predict(X)
        accuracy_gnb = accuracy(y, y_predict_gassian)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        symbols = np.array(["circle", "diamond", "square"])

        scatter_LDA = go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color = y_predict_lda, symbol = symbols[y]))

        scatter_Gassian = go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color = y_predict_gassian, symbol = symbols[y] ))

        # Add traces for data-points setting symbols and colors
        fig = make_subplots(rows=1, cols=2,
                      subplot_titles=(f"Gaussian Naive Bayes, accuracy: {accuracy_gnb}", f"LDA, accuracy: {accuracy_lda}")) \
            .add_traces( [scatter_Gassian, scatter_LDA], rows=[1, 1], cols=[1, 2]) \
            .update_layout(title=f"Datasets: {f}", margin=dict(t=100)).update_layout(showlegend=False)


        # Add `X` dots specifying fitted Gaussians' means
        x_lda = go.Scatter(x= lda.mu_[:, 0] , y=lda.mu_[:, 1], mode='markers',
                           marker=dict(symbol = "x", color="black"))
        x_gnb = go.Scatter(x= gassian_naive_bayes.mu_[:, 0] , y=gassian_naive_bayes.mu_[:, 1], mode='markers',
                           marker=dict(symbol = "x", color="black"))
        fig.add_traces([x_gnb, x_lda], rows=[1, 1], cols=[1, 2])


        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(lda.classes_.shape[0]):
            ellipses_lda = get_ellipse(np.transpose(lda.mu_[i]), lda.cov_)
            ellipses_gnb = get_ellipse(np.transpose(gassian_naive_bayes.mu_[i]), np.diag( gassian_naive_bayes.vars_[i]))
            fig.add_traces([ellipses_gnb, ellipses_lda], rows=[1, 1], cols=[1, 2])
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()