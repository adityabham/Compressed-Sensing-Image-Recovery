import random
import numpy as np
from sklearn import linear_model


def test_train_split(A, B, m):
    # Split A and B matrices to create testing and training
    test_indices = random.sample(range(B.shape[0]), m)
    test_indices.sort()
    train_indices = []
    for idx in range(B.shape[0]):
        if idx not in test_indices:
            train_indices.append(idx)

    A_test = A[test_indices]
    B_test = B[test_indices]
    A_train = A[train_indices]
    B_train = B[train_indices]

    return A_test, B_test, A_train, B_train


def random_CV(A, B, S, K):
    lambda_range = [1e-6, 5.5e-5, 1e-5, 5.5e-4, 1e-4, 5.5e-3, 1e-3, 5.5e-2, 1e-2, 0.055,
                    1e-1, 0.5, 1, 10, 55, 100, 1000, 10000, 100000, 1000000]

    mse_vals = []
    # Random CV process
    for cv_iter in range(0, 20):
        lambda_val = lambda_range[cv_iter]

        # Create testing and training data
        A_test, B_test, A_train, B_train = test_train_split(A, B, S // 6)
        c_con = A_train[0][0]
        A_train_prime = np.delete(A_train, 0, 1)

        # Create and fit lasso model
        lasso_model = linear_model.Lasso(alpha=lambda_val, max_iter=1e5, fit_intercept=True)
        lasso_model.fit(A_train_prime, B_train)
        dct_coef_lasso = lasso_model.coef_.reshape((K * K) - 1, 1)
        dct_coef = np.insert(dct_coef_lasso, 0, [lasso_model.intercept_ / c_con]).reshape(K * K, 1)

        # Evaluating model
        B_hat = np.matmul(A_test, dct_coef)
        mse_val = np.square(np.subtract(B_test, B_hat)).mean()
        mse_vals.append(mse_val)

    # Return optimal lambda value
    min_mse_val = min(mse_vals)
    min_index = mse_vals.index(min_mse_val)
    opt_lambda = lambda_range[min_index]

    return opt_lambda
