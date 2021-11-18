import numpy as np
import pandas as pd
from scipy.optimize import minimize


def train_test_split(X, y, train_start, train_end, test_start, test_end):
    return X[train_start:train_end, 2:], y[train_start:train_end, 2:].flatten(), \
           X[test_start:test_end, 2:], y[test_start:test_end, 2:].flatten()


def ordinary_least_square_loss(predicted, actual, sample_weights=None):
    sum_squared_error = np.sum((predicted - actual) ** 2)
    mean_error = sum_squared_error / float(len(actual))
    return mean_error


class CustomLinearModel:
    """
    Linear model: Y = XC, fit by minimizing the provided loss_function
    """

    def __init__(self, loss_function=ordinary_least_square_loss,
                 X=None, Y=None, sample_weights=None, coeff_init=None, regularization=0, rho=1):
        self.regularization = regularization
        self.coeff = None
        self.loss_function = loss_function
        self.sample_weights = sample_weights
        self.coeff_init = coeff_init
        self.rho = rho

        self.X = X
        self.Y = Y

    def predict(self, X):
        prediction = np.matmul(X, self.coeff)
        return (prediction)

    def model_error(self):
        error = self.loss_function(
            self.predict(self.X), self.Y, sample_weights=self.sample_weights
        )
        return (error)

    def elastic_net_penalty(self, coeff):
        self.coeff = coeff
        penalty = (1 - self.rho) * self.l1_regularization() + self.rho * self.l2_regularization()
        return self.model_error() + penalty

    def l2_regularization(self):
        return sum(self.regularization * np.array(self.coeff) ** 2)

    def l1_regularization(self):
        return sum(self.regularization * abs(self.coeff))

    def fit(self, maxiter=250):
        # Initialize coeff estimates (you may need to normalize your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.coeff_init) == type(None):
            # set coeff_init = 1 for every feature
            self.coeff_init = np.array([1] * self.X.shape[1])
        else:
            # Use provided initial values
            pass

        if self.coeff is not None and all(self.coeff_init == self.coeff):
            print("Model already fit once; continuing fit with more iterations.")

        res = minimize(self.elastic_net_penalty, self.coeff_init, method='BFGS', options={'maxiter': maxiter})
        self.coeff = res.x
        self.coeff_init = self.coeff


# Calculate root mean squared error
def rsqr_metric(predicted, actual):
    sum_error = np.sum((predicted - actual) ** 2)
    mean_error = sum_error / np.sum(actual ** 2)
    return 1 - mean_error


def train(X, y, recursive=False, n=10, loss_function=ordinary_least_square_loss, sample_weights=None, coeff_init=None, regularization=0, rho=1):
    min_error = float("inf")
    all_errors = []
    coefficients = None
    if recursive:
        for i in range(196500,201900,100):
            x_train = X[X["level_1"] < i].to_numpy()[:,2:]
            y_train = y[y["level_1"] < i].to_numpy()[:,2:]
            x_test = X[(X["level_1"] >= i) & (X["level_1"] < i + 100)].to_numpy()[:,2:]
            y_test = y[(y["level_1"] >= i) & (y["level_1"] < i + 100)].to_numpy()[:,2:]

            model = CustomLinearModel(
                loss_function=loss_function,
                sample_weights=sample_weights,
                coeff_init=coeff_init,
                regularization=regularization,
                rho=rho,
                X=x_train, Y=y_train
            )
            model.fit()
            error = ordinary_least_square_loss(np.dot(x_test, model.coeff), y_test)
            all_errors.append(error)
            if error < min_error:
                min_error = error
                coefficients = model.coeff
    else:
        i = 196307
        while i + n*100 < 201900:
            x_train = X[(X["level_1"] >= i) & (X["level_1"] < i + n*100)].to_numpy()[:,2:]
            y_train = y[(y["level_1"] >= i) & (y["level_1"] < i + n*100)].to_numpy()[:,2:]
            x_test = X[(X["level_1"] >= i + n*100) & (X["level_1"] < i + (n+1)*100)].to_numpy()[:,2:]
            y_test = y[(y["level_1"] >= i + n*100) & (y["level_1"] < i + (n+1)*100)].to_numpy()[:,2:]

            model = CustomLinearModel(
                loss_function=loss_function,
                sample_weights=sample_weights,
                coeff_init=coeff_init,
                regularization=regularization,
                rho=rho,
                X=x_train, Y=y_train
            )
            model.fit()
            error = ordinary_least_square_loss(np.dot(x_test, model.coeff), y_test)
            all_errors.append(error)
            if error < min_error:
                min_error = error
                coefficients = model.coeff
            i = i + 100

    return min_error, all_errors, coefficients


if __name__ == '__main__':
    X = pd.read_csv("full_predictor_set_bfill.csv", index_col=0)
    y = pd.read_csv("returns.csv", index_col=0)

    print(X.shape)
    print(y.shape)

    max_r2, all_r2, coefficients = train(X, y)

    print(max_r2)
    # Takes about a minute to compete one fit
