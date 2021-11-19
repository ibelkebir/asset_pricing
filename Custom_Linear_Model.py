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


def recursive_train(model, X, y, rolling_size):
    min_error = float('inf')
    all_errors = []
    coefficients = None

    # During each iteration, fit the model with train set, and then test on the validation set
    # Record the output error in all_errors
    # Compare with the current model's error with the min_error
    # if current error leq than min_error, update the min with the current one and update the coefficients
    # else proceed to next iteration

    return min_error, all_errors, coefficients


if __name__ == '__main__':
    X_df = pd.read_csv("full_predictor_set_bfill.csv", index_col=0)
    y_df = pd.read_csv("returns.csv", index_col=0)

    X = X_df.to_numpy()
    y = y_df.to_numpy()

    print(X.shape)
    print(y.shape)

    X_train, y_train, X_test, y_test = train_test_split(X, y, 0, 2100000, 2100001, len(X) - 1)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    l2_model = CustomLinearModel(
        loss_function=ordinary_least_square_loss,
        X=X_train, Y=y_train, regularization=0.00012
    )
    l2_model.fit()
    print(l2_model.coeff)
    print(rsqr_metric(l2_model.predict(X_train), y_train))
    print(rsqr_metric(l2_model.predict(X_test), y_test))
