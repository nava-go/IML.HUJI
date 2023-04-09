from __future__ import annotations
import numpy as np
import sys
sys.path.append("G:\האחסון שלי\year2\semesterB\iml\IML.HUJI")
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    y_noiseless = (X+3) * (X+2) *(X+1) * (X-1) * (X-2)
    y_noised = y_noiseless + np.random.normal(0, noise, n_samples)
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), pd.DataFrame(y_noised) , 2/3)

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=X,
        y=y_noiseless,
        name="true model (noiseless)"))

    fig1.add_trace(go.Scatter(
        x=train_x.to_numpy()[:,0],
        y=train_y.to_numpy()[:,0],
        name="train samples",
        mode='markers'
    ))

    fig1.add_trace(go.Scatter(
        x=test_x.to_numpy()[:,0],
        y=test_y.to_numpy()[:,0],
        name="test samples",
        mode='markers'
    ))
    fig1.update_layout(
        title= f"(1) Num Sampels: {n_samples}, noise: {noise}",
        xaxis_title="x",
        yaxis_title="f(x)",
        legend_title="data sets:",
    )
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_error = np.zeros(11)
    validation_error = np.zeros(11)
    for k in range(11):
        estimator = PolynomialFitting(k)
        train_error[k], validation_error[k] = cross_validate(estimator,
                                                             train_x.to_numpy()[:,0],
                                                             train_y.to_numpy()[:,0],
                                                             mean_square_error,
                                                             5)


    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(11)),
        y=train_error,
        name="train error", mode = 'markers'))
    fig2.add_trace(go.Scatter(
        x=list(range(11)),
        y=validation_error,
        name="validation error", mode='markers'))
    fig2.update_layout(
        title=f"Train and validation error as function of polynom degree. Num samples: {n_samples}. Noise: {noise}.",
        xaxis_title="k - polynomial degrees",
        yaxis_title="error",
        legend_title="data sets:",
    )
    fig2.show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_error)
    validation_error = validation_error[best_k]
    estimator3 = PolynomialFitting(best_k)
    estimator3.fit(train_x.to_numpy()[:,0], train_x.to_numpy()[:,0])
    predict_test = estimator3.predict(test_x.to_numpy()[:,0])
    test_error = mean_square_error(predict_test, test_y.to_numpy()[:,0])
    print("num sampels:", n_samples, ",noise:",noise ,",k*: ", best_k, " ,validation error:", validation_error, " ,test error:", test_error)



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x,test_x, train_y, test_y = train_test_split(X, y, train_size=n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    possibole_lambdas = np.linspace(0, 2, n_evaluations)
    possibole_lambdas = np.delete(possibole_lambdas, 0)
    train_error_ridge = np.zeros(n_evaluations-1)
    validation_error_ridge = np.zeros(n_evaluations-1)
    for i,lam in enumerate(possibole_lambdas):
        estimator = RidgeRegression(lam)
        train_error_ridge[i], validation_error_ridge[i] = cross_validate(estimator,
                                                             train_x,
                                                             train_y,
                                                             mean_square_error,
                                                             5)

    train_error_lasso = np.zeros(n_evaluations-1)
    validation_error_lasso = np.zeros(n_evaluations-1)
    for i,lam in enumerate(possibole_lambdas):
        estimator = Lasso(lam)
        train_error_lasso[i], validation_error_lasso[i] = cross_validate(estimator,
                                                             train_x,
                                                             train_y,
                                                             mean_square_error,
                                                             5)

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=possibole_lambdas,
        y=train_error_ridge,
        name="Ridge train error"))
    fig7.add_trace(go.Scatter(
        x=possibole_lambdas,
        y=validation_error_ridge,
        name="Ridge validation error"))
    fig7.add_trace(go.Scatter(
        x=possibole_lambdas,
        y=train_error_lasso,
        name="Lasso train error"))
    fig7.add_trace(go.Scatter(
        x=possibole_lambdas,
        y=validation_error_lasso,
        name="Lasso validation error"))
    fig7.update_layout(
        title="(7) Train error and validation error - Ridge and Lasso",
        xaxis_title="lambda - regularization parameter",
        yaxis_title="error",
        legend_title="data sets:",
    )
    fig7.show()
    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = possibole_lambdas[np.argmin(validation_error_ridge)]
    best_lasso = possibole_lambdas[np.argmin(validation_error_lasso)]
    print(f"best regularization parameter: ridge: {best_ridge}, Lasso: {best_lasso}")

    estimator_ridge = RidgeRegression(best_ridge)
    estimator_lasso = Lasso(best_lasso)
    estimator_LSE = LinearRegression()

    estimator_ridge.fit(train_x, train_y)
    estimator_lasso.fit(train_x, train_y)
    estimator_LSE.fit(train_x, train_y)

    test_error_ridge = estimator_ridge.loss(test_x, test_y)
    test_error_lasso = mean_square_error(estimator_lasso.predict(test_x), test_y)
    test_error_LSE = estimator_LSE.loss(test_x, test_y)
    print(f"Test error of the fitted models: ridge: {test_error_ridge}, Lasso: {test_error_lasso}, LSE: {test_error_LSE}")



if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500,noise=10)
    select_regularization_parameter()
