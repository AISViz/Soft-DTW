import numpy as np
from sdtw import SoftDTW, distance as sdtw_distance


import tensorflow as tf

from softdtwkeras.SDTWLoss import SDTWLoss

if __name__ == '__main__':
    x_ = np.array([[0,0], [0.9,0], [0,0], [0.5,1], [0,0], [0,0]])
    y_ = np.array([[0,0], [0,0], [0.1,0], [0.9,1], [0,0], [0.5,0]])

    x_tf = tf.convert_to_tensor(x_, dtype=tf.float64)
    y_tf = tf.convert_to_tensor(y_, dtype=tf.float64)

    D = sdtw_distance.SquaredEuclidean(x_, y_)

    tff_sdtw__= SDTWLoss(gamma=0.001)
    val_tensor = tff_sdtw__.unit_loss(x_tf, y_tf)

    sdtw = SoftDTW(D, gamma=0.001)
    # soft-DTW discrepancy, approaches DTW as gamma -> 0
    value = sdtw.compute()
    # gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
    E = sdtw.grad()
    # gradient w.r.t. X, shape = [m, d]
    G = D.jacobian_product(E)

    print(value)
    print(val_tensor)
