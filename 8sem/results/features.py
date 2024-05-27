import numpy as np


def AV(matrix):
    """Угловой момент второго порядка"""
    
    normalized_matrix = matrix / np.sum(matrix)
    av = np.sum(np.square(normalized_matrix))
    return av


def D(matrix):
    """Обратный момент различий"""
    
    d = np.sum(matrix / (1 + np.square(np.arange(matrix.shape[0]))))
    return d
