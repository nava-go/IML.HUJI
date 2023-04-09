import sys
sys.path.append("G:\האחסון שלי\year2\semesterB\iml\IML.HUJI")
import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.day_of_year
    df = df[df['Temp'] > -20]
    return df

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('G:\האחסון שלי\year2\semesterB\iml\IML.HUJI\datasets\City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    #scatter plot:
    df['Year'] = df['Year'].astype(str)
    df_israel = df[df['Country'] == 'Israel']
    fig = px.scatter(df_israel, x="DayOfYear", y="Temp", color="Year",
                     title="(2.1) Temperature in Function of Day of year")
    fig.show()
    # we think that a polinom in 3 or 4 degree will be suitable
    #bar plot:
    df_israel_month = df_israel.groupby('Month').agg({'Temp':'std'})
    fig2 = px.bar(df_israel_month, x=list(range(1, 13)), y='Temp',
                        title="(2.2) STD in Function of Month of year",
                  labels={"x": "month", "Temp":"r$STD$"})
    fig2.show()

    # Question 3 - Exploring differences between countries
    df_county_month_mean = df.groupby(['Country', 'Month'])['Temp'].mean().reset_index()
    df_county_month_std = df.groupby(['Country', 'Month'])['Temp'].std().reset_index()
    fig3 = px.line(pd.DataFrame({'mean_temp': df_county_month_mean['Temp'],
                                                               'std_temp': df_county_month_std['Temp'],
                                                               'Country': df_county_month_mean['Country'],
                                                               'Month': df_county_month_mean['Month']}),
                                                 x='Month', y='mean_temp', color='Country', error_y='std_temp',
                                                 labels={'x': "Month", 'mean_temp':"mean temperature"},
                                                title="(3) Average Monthly Temperature")
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    X = df_israel["Month"]
    y = df_israel["Temp"]
    train_x, train_y, test_x, test_y = split_train_test(X, y)
    loss = []
    for k in range(1, 11):
        estimator = PolynomialFitting(k)
        estimator.fit(train_x.to_numpy(), train_y.to_numpy())
        loss.append(round(estimator.loss(test_x, test_y), 2))
    print("test error (loss):" , loss)
    fig4 = px.bar(loss, x=list(range(1, 11)), y=loss,
                  title="(4) Loss in Function of Degree K",
                  labels={"x": "polynom degree k", "y": "loss values"})
    fig4.show()
    #in deg 5 we get the minimum loss value, but we should consider the variance


    # Question 5 - Evaluating fitted model on different countries
    estimator5 = PolynomialFitting(5)
    estimator5.fit(X, y)
    countries = ['South Africa', 'Jordan', 'The Netherlands']
    loss = []
    for country in countries:
        df_country = df[df['Country'] == country]['Month']
        y = df[df['Country'] == country]['Temp']
        loss.append(estimator5.loss(df_country.to_numpy(), y.to_numpy()))
    fig5 = px.bar(loss, x=countries, y=loss,
                  title="(5) Loss in Function of Countries",
                  labels={"x": "countries", "y": "loss value"})
    fig5.show()
