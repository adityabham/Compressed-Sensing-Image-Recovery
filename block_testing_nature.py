import image_methods
import block_methods
import basis_methods
import sampling_method
import cv_process
import numpy as np
from sklearn import linear_model

# Load image into matrix
image_matrix = image_methods.import_image("nature.png")

# Break image into K x K blocks
K = 16
blocks = block_methods.generate_blocks(image_matrix, K)

# Transformation Matrix
T = basis_methods.transformation_mat_gen(K, K)

# Selected Block
test_block = blocks[70]
# print(test_block)
C = test_block.reshape(K * K, 1)

# Sampling
S = 100
A, B, C_sampled = sampling_method.block_sampling(C, T, S)
# print(C_sampled.reshape(K, K))

# Split
opt_lambda = cv_process.random_CV(A, B, S, K)
c_con = A[0][0]
A_prime = np.delete(A, 0, 1)

# CV
lasso_model = linear_model.Lasso(alpha=opt_lambda, max_iter=1e5, fit_intercept=True)
lasso_model.fit(A_prime, B)
dct_coef_lasso = lasso_model.coef_.reshape((K * K) - 1, 1)
dct_coef = np.insert(dct_coef_lasso, 0, [lasso_model.intercept_ / c_con]).reshape(K * K, 1)

estimated_block = np.matmul(T, dct_coef)

for idx in range(0, C_sampled.shape[0]):
    if C_sampled[idx][0] == -1:
        C_sampled[idx] = estimated_block[idx]

# print(np.square(np.subtract(C, C_sampled)).mean())
print(C)
print(C_sampled)

recov_block = C_sampled.reshape(K, K)
# blocks[70] = recov_block
# print("----------------")
# print(blocks[70])
