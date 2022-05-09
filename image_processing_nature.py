import numpy as np
import image_methods
import block_methods
import basis_methods
import sampling_method
import cv_process
from sklearn import linear_model
from scipy import signal

# Load image into matrix
image_matrix = image_methods.import_image("nature.png")

# Break image into K x K blocks
K = 16
blocks = block_methods.generate_blocks(image_matrix, K)

# Transformation Matrix
T = basis_methods.transformation_mat_gen(K, K)

for block_index in range(0, blocks.shape[0]):
    # Selected Block
    test_block = blocks[block_index]
    C = test_block.reshape(K * K, 1)

    # Sampling
    S = 150
    A, B, C_sampled = sampling_method.block_sampling(C, T, S)

    # CV
    opt_lambda = cv_process.random_CV(A, B, S, K)
    c_con = A[0][0]
    A_prime = np.delete(A, 0, 1)

    lasso_model = linear_model.Lasso(alpha=opt_lambda, max_iter=1e5, fit_intercept=True)
    lasso_model.fit(A_prime, B)
    dct_coef_lasso = lasso_model.coef_.reshape((K * K) - 1, 1)
    dct_coef = np.insert(dct_coef_lasso, 0, [lasso_model.intercept_ / c_con]).reshape(K * K, 1)
    estimated_block = np.matmul(T, dct_coef)

    for idx in range(0, C_sampled.shape[0]):
        if C_sampled[idx][0] == -1:
            C_sampled[idx] = estimated_block[idx]

    recovered_block = C_sampled.reshape(K, K)
    blocks[block_index] = recovered_block

merged_blocks = block_methods.merge_blocks(blocks, image_matrix, K)
image_methods.create_recovered_image(merged_blocks)
merged_blocks_filtered = signal.medfilt2d(merged_blocks, kernel_size=3)
image_methods.create_recovered_image(merged_blocks_filtered)

block_methods.block_mse(image_matrix, merged_blocks, merged_blocks_filtered)
