import numpy as np
from sdtw import SoftDTW, soft_dtw_tf as tf_soft_dtw
from sdtw import distance as sdtw_distance

import tensorflow as tf


if __name__ == '__main__':
    x_ = np.array([[0,0], [0.9,0], [0,0], [0.5,1], [0,0], [0,0]])
    y_ = np.array([[0,0], [0,0], [0.1,0], [0.9,1], [0,0], [0.5,0], [0,0]])

    x_tf = tf.convert_to_tensor(x_, dtype=tf.float64)
    y_tf = tf.convert_to_tensor(y_, dtype=tf.float64)

    D = sdtw_distance.SquaredEuclidean(x_, y_)

    tff_sdtw__= tf_soft_dtw.SoftDTWTF(gamma=0.001)
    val_single = tff_sdtw__(x_tf, y_tf)
    val_ = tff_sdtw__.batchbatch_loss(
        tf.reshape(x_tf, (1, x_tf.shape[0], x_tf.shape[1])),
                                      tf.reshape(y_tf, (1, y_tf.shape[0], y_tf.shape[1]) ))

    sdtw = SoftDTW(D, gamma=0.001)
    # soft-DTW discrepancy, approaches DTW as gamma -> 0
    value = sdtw.compute()
    # gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
    E = sdtw.grad()
    # gradient w.r.t. X, shape = [m, d]
    G = D.jacobian_product(E)

    print(value)
