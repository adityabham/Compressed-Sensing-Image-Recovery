import math
import numpy as np


def transform_col_gen(x, y, P, Q):
    t_d = []
    for u in list(range(1, P + 1)):
        t_d_col = []
        if u == 1:
            a_u = math.sqrt(1 / P)
        else:
            a_u = math.sqrt(2 / P)

        for v in list(range(1, Q + 1)):
            if v == 1:
                b_v = math.sqrt(1 / Q)
            else:
                b_v = math.sqrt(2 / Q)
            p_exp = (math.pi * (2 * x - 1) * (u - 1)) / (2 * P)
            q_exp = (math.pi * (2 * y - 1) * (v - 1)) / (2 * Q)
            p_val = a_u * math.cos(p_exp)
            q_val = b_v * math.cos(q_exp)
            t_d_col_elem = p_val * q_val
            t_d_col.append(t_d_col_elem)

        t_d_col = np.array(t_d_col).reshape(len(t_d_col), 1)
        t_d.append(t_d_col)

    t_d = np.concatenate(t_d, axis=1)
    return t_d


def transformation_mat_gen(P, Q):
    T = []
    for x in list(range(1, P + 1)):
        for y in list(range(1, Q + 1)):
            t_d = transform_col_gen(x, y, P, Q)
            td = t_d.reshape(t_d.shape[0] * t_d.shape[1], 1)
            T.append(td)

    T = np.concatenate(T, axis=1)
    T = np.transpose(T)
    T = T.astype(float)
    return T

