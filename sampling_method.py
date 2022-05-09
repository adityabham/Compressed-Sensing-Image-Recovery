import numpy as np
import random
import copy


def block_sampling(C, T, S):
    sample_indices = random.sample(range(C.shape[0]), S)
    sample_indices.sort()
    A = T[sample_indices]
    B = C[sample_indices]
    C_sampled = copy.deepcopy(C)

    for idx in range(0, C.shape[0]):
        if idx in sample_indices:
            continue
        else:
            C_sampled[idx][0] = -1

    return A, B, C_sampled
