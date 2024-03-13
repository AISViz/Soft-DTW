import tensorflow as tf
import numpy as np

from softdtwkeras.soft_dtw_fast import py_softmin3


class SDTWLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 1.0):
        super(SDTWLoss, self).__init__()
        self.gamma = gamma

    def squared_euclidean_compute_tf(self, a: tf.Tensor, b: tf.Tensor) -> None:
        """
        Computes pairwise distances between each elements of A and each elements of B.
        Args:
          A,    [m,d] matrix
          B,    [n,d] matrix
        Returns:
          D,    [m,n] matrix of pairwise distances
        """

        # return pairwise euclidead difference matrix
        D = tf.reduce_sum((tf.expand_dims(a, 1) - tf.expand_dims(b, 0)) ** 2, 2)
        return D

    def call(self, y_true, y_pred):
        # tmp = []
        # for b_i in range(0, y_true.shape[0]):
        #     dis_ = self.unit_loss(y_true[b_i], y_pred[b_i])
        #     tmp.append(dis_)
        # return tf.reduce_sum(tf.convert_to_tensor(tmp))

        # def compute_loss(pair):
        #     y_true_i, y_pred_i = pair
        #     return self.unit_loss(y_true_i, y_pred_i)
        #
        # losses = tf.map_fn(compute_loss, (y_true, y_pred), dtype=tf.float32)
        # return tf.reduce_sum(losses)

        # batch execution
        batch_Distances_ = self.batch_squared_euclidean_compute_tf(y_true, y_pred)
        losses = tf.map_fn(self.unit_loss_from_D, batch_Distances_, dtype=tf.float32)
        return tf.reduce_sum(losses)

    def unit_loss(self, y_true, y_pred):
        D_ = self.squared_euclidean_compute_tf(y_true, y_pred)
        m, n = tf.shape(D_)[0], tf.shape(D_)[1]

        # Allocate memory.
        R_ = tf.fill((m + 2, n + 2), tf.constant(np.inf, dtype=tf.float32))
        R_ = tf.tensor_scatter_nd_update(R_, [[0, 0]], [0.0])

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # D is indexed starting from 0.
                R_ = tf.tensor_scatter_nd_update(
                    R_,
                    [[i, j]],
                    [D_[i - 1, j - 1] + py_softmin3(R_[i - 1, j], R_[i - 1, j - 1], R_[i, j - 1], self.gamma)]
                )

        return R_[m, n]

    def batch_squared_euclidean_compute_tf(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Computes pairwise distances between each elements of A and each elements of B.
        Args:
          a,    [batch, m, d] tensor
          b,    [batch, n, d] tensor
        Returns:
          D,    [batch, m, n] tensor of pairwise distances
        """

        # Expand dimensions to enable broadcasting
        a_expanded = tf.expand_dims(a, axis=2)  # Shape: [batch, m, 1, d]
        b_expanded = tf.expand_dims(b, axis=1)  # Shape: [batch, 1, n, d]

        # Compute pairwise squared Euclidean distances
        squared_diff = tf.reduce_sum(tf.square(a_expanded - b_expanded), axis=-1)  # Shape: [batch, m, n]

        return squared_diff

    def unit_loss_from_D(self, D_):
        m, n = tf.shape(D_)[0], tf.shape(D_)[1]

        # Allocate memory.
        R_ = tf.fill((m + 2, n + 2), tf.constant(np.inf, dtype=tf.float32))
        R_ = tf.tensor_scatter_nd_update(R_, [[0, 0]], [0.0])

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # D is indexed starting from 0.
                R_ = tf.tensor_scatter_nd_update(
                    R_,
                    [[i, j]],
                    [D_[i - 1, j - 1] + py_softmin3(R_[i - 1, j], R_[i - 1, j - 1], R_[i, j - 1], self.gamma)]
                )

        return R_[m, n]