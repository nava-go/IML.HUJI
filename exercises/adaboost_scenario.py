import numpy as np
from typing import Tuple
import sys
sys.path.append("G:\האחסון שלי\year2\semesterB\iml\IML.HUJI")
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    train_loss = np.zeros((n_learners,))
    test_loss = np.zeros((n_learners,))
    for t in range(1, n_learners+1):
        train_loss[t-1] = adaboost.partial_loss(train_X, train_y, t)
        test_loss[t - 1] = adaboost.partial_loss(test_X, test_y, t)
    axis_x = list(range(1, n_learners+1))
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=axis_x,
        y=train_loss,
        name="train errors"))

    fig1.add_trace(go.Scatter(
        x=axis_x,
        y=test_loss,
        name="test errors"
    ))

    fig1.update_layout(
        title="(1) Train and Test Errors in function of the number of fitted learners",
        xaxis_title="numbers of fitted learners",
        yaxis_title="error",
        legend_title="data sets:",
    )

    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    symbols = np.array(["circle", "x"])
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    test_y_corrected = np.where(test_y<0, 0, 1) # replace -1 in 0

    fig = make_subplots(rows=1, cols=4, subplot_titles=[f"iteration num: {t}" for t in T],
                       horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        def temp_predict(X: np.ndarray) -> np.ndarray:
            return adaboost.partial_predict(X, t)
        fig.add_traces([decision_surface(temp_predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y_corrected, symbol=symbols[test_y_corrected], colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows= 1, cols= i+1)

    fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()



    # Question 3: Decision surface of best performing ensemble
    min_ensemble_size = np.argmin(test_loss) + 1
    accuracy = 1- test_loss[min_ensemble_size-1]
    fig3 = go.Figure()

    def temp_predict(X: np.ndarray) -> np.ndarray:
        return adaboost.partial_predict(X, min_ensemble_size)

    fig3.add_traces([decision_surface(temp_predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y_corrected, symbol=symbols[test_y_corrected],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])

    fig3.update_layout(
        title=f"(3) Decision surface of ensemble size: {min_ensemble_size}, accuracy: {accuracy}",
        xaxis_title="feature 1",
        yaxis_title="feature 2",
    )

    fig3.show()

    # Question 4: Decision surface with weighted samples
    D = (adaboost.D_ / np.max(adaboost.D_)) * 10
    train_y_corrected = np.where(train_y<0, 0, 1) # replace -1 in 0

    fig4 = go.Figure()

    fig4.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(size= D, color=train_y_corrected, symbol=symbols[train_y_corrected],
                                            colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1)))])

    fig4.update_layout(
        title=f"(4) Decision surface of ensemble size: 250",
        xaxis_title="feature 1",
        yaxis_title="feature 2",
    )

    fig4.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    # Q.5.
    fit_and_evaluate_adaboost(noise=0.4)