import sys
sys.path.append("G:\האחסון שלי\year2\semesterB\iml\IML.HUJI")
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.chromium_args = tuple([arg for arg in pio.kaleido.scope.chromium_args if arg != "--disable-dev-shm-usage"])
pio.templates.default = "simple_white"



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df.dropna(inplace=True)

    #delete rows with invalid data
    df = df[df['id'] != 0]
    df = df[df['price'] > 0]
    df = df[df['bedrooms'] >= 0]
    df = df[df['bathrooms'] >= 0]
    df = df[df['sqft_living'] > 0]
    df = df[df['floors'] > 0]
    df = df[df['sqft_above'] > 0]
    df = df[df['sqft_basement'] >= 0]
    df = df[df['yr_built'] > 0]
    df = df[df['yr_renovated'] >= 0] # can bw zero, whan never been renovate
    df = df[df['sqft_living15'] > 0]
    df = df[df['sqft_lot15'] > 0]
    #delete: id, sqft_living
    del df['id']
    del df['sqft_living'] # becusue its the direct sum: sqft_above+ sqft_basement
    del df['lat']
    del df['long']
    #re-catogory
    #date
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df = pd.get_dummies(df, columns=['month'])
    del df['date']
    #year built/year renovate
    df['max_yr_built_or_renovated'] = df[['yr_built','yr_renovated']].max(axis=1)
    df = df.rename(columns={'yr_renovated':'is_renovate'})
    df['is_renovate'] = (df['is_renovate'] != 0).astype(int) #check
    del df['yr_built']
    #zipcodeing
    df = pd.get_dummies(df, columns=['zipcode'])
    # the geo cols (lat, long) - may we should also pay attention to them (map to some squere?)

    #spilt the pichers from the response
    response = df.pop('price')
    return df, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    map_col_pirson = {}
    for col in X:
        pirson = np.cov(X[col], y)[0, 1] / (np.std(X[col]) *np.std(y))
        map_col_pirson[col] = pirson
        fig = px.scatter(pd.DataFrame({'x':X[col], 'y': y}), x="x", y="y", trendline="ols",
                         title  = f"The Corr. Between {col} and the Price (response), Pearson Correlation: {pirson}",
                         labels= {"x": f"{col} values", "y":"r$prices (response values)$" })
        #fig.show()
        fig.write_image(f"{output_path}\\{col}.png", format="png")
    #print("Overview on the pirson corr to each feature: ", map_col_pirson)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('G:\האחסון שלי\year2\semesterB\iml\IML.HUJI\datasets\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, 'G:\האחסון שלי\year2\semesterB\iml\IML.HUJI\exercises')

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    estimator = LinearRegression()
    P = list(range(10, 101))
    mean_loss = np.zeros((91,))
    std_mean_loss = np.zeros((91,))
    for p in P:
        loss = np.zeros(10)
        for i in range(10):
            cur_sample, cur_response, temp0, temp1 = split_train_test(train_x, train_y, (p/100))
            estimator.fit(cur_sample.to_numpy(), cur_response.to_numpy())
            loss[i] = estimator.loss(test_x.to_numpy(), test_y.to_numpy())
        mean_loss[p-10] = np.mean(loss)
        std_mean_loss[p-10] = np.std(loss)

    fig = go.Figure([go.Scatter(x=P, y=mean_loss, mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                          go.Scatter(x=P, y=mean_loss-2*std_mean_loss, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
                          go.Scatter(x=P, y=mean_loss+2*std_mean_loss, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False),],
                layout=go.Layout(
                  title=r"$\text{(4) Mean Loss as function of Training Set Percentage}$",
                  xaxis_title="$\\text{training set percentage}$",
                  yaxis_title="$\\text{mean loss}$"))
    fig.show()


