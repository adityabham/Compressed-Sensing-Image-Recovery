import numpy as np


def generate_blocks(image_mat, k):
    y_len, x_len = image_mat.shape
    return image_mat.reshape(y_len // k, k, -1, k).swapaxes(1, 2).reshape(-1, k, k)


def merge_blocks(input_blocks, image_mat, k):
    y_len, x_len = image_mat.shape
    num_row_blocks = (x_len // k)
    row_arr = []
    all_row_arr = []
    for block in input_blocks:
        row_arr.append(block)
        if len(row_arr) == num_row_blocks:
            all_row_arr.append(np.concatenate(row_arr, axis=1))
            row_arr.clear()

    return np.concatenate(all_row_arr, axis=0)


def block_mse(original_matrix, recovered_matrix, recovered_matrix_filtered):
    num_rows = original_matrix.shape[0]
    num_col = original_matrix.shape[1]

    om_vc = original_matrix.reshape(num_rows * num_col, 1)
    rm_vec = recovered_matrix.reshape(num_rows * num_col, 1)
    rmf_vec = recovered_matrix_filtered.reshape(num_rows * num_col, 1)

    mse_1 = np.square(np.subtract(om_vc, rm_vec)).mean()
    print("Before filtering MSE:")
    print(mse_1)

    mse_2 = np.square(np.subtract(om_vc, rmf_vec)).mean()
    print("After filtering MSE:")
    print(mse_2)



