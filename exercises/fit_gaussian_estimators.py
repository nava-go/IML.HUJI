import sys
sys.path.append("G:\האחסון שלי\year2\semesterB\iml\IML.HUJI")
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    sample1 = np.random.normal(mu, sigma, 1000)
    estimator1 = UnivariateGaussian()
    estimator1.fit(sample1)
    print("(", estimator1.mu_, ",", estimator1.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    samples_size = np.arange(10, 1010, 10)
    distance_estimated_mean_to_expectation = []
    for size in samples_size:
        X = sample1[:size]
        distance_estimated_mean_to_expectation.append(abs(estimator1.fit(X).mu_ - mu))
    go.Figure([go.Scatter(x=samples_size, y=distance_estimated_mean_to_expectation, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(2) Absolute Distance Between The Estimated And True Value Of The Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$n\\text{ - number of samples}$",
                  yaxis_title="r$distance$")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    estimator1.fit(sample1)
    pdf_sample1 = estimator1.pdf(sample1)
    go.Figure([go.Scatter(x=sample1, y=pdf_sample1, mode='markers',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(3) Probability Density Function Of Sample Values}$",
                  xaxis_title="$x\\text{ - sample values}$",
                  yaxis_title="r$pdf values$")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu4 = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    samples = np.random.multivariate_normal(mu4, cov, 1000)
    estimator = MultivariateGaussian()
    estimator.fit(samples)
    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    p = 200
    f = np.linspace(-10, 10, p)
    loglikelihood = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            mu = [f[i], 0, f[j], 0]
            loglikelihood[i][j] = MultivariateGaussian.log_likelihood(mu, cov, samples)
    go.Figure([go.Heatmap(x=f, y=f, z=loglikelihood)],
              layout=go.Layout(
                  title=r"$\text{(5) Heatmap Of The Log Likelihood In Function Of f1 And f3}$",
                  xaxis_title="$\\text{f3}$",
                  yaxis_title="$\\text{f1}$")).show()

    # Question 6 - Maximum likelihood
    max_f1, max_f3 = np.unravel_index(np.argmax(loglikelihood, axis=None), loglikelihood.shape)
    max_loglikelihood = loglikelihood[max_f1][max_f3]
    print("f1: ", f[max_f1], ", f3: ", f[max_f3], ", max loglikelyhood: ", max_loglikelihood)



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
