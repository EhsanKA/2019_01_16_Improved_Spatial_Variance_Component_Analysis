import numpy as np
import tensorflow as tf
from codes.utils import *
from codes.preparing_data import *
from codes.loader import *
from codes.reconstructing_with_numpy import *
import os

ix = tf.placeholder(shape=[None, None], dtype=tf.float64, name="x")
y = tf.placeholder(shape=[None, 1], dtype=tf.float64, name="y")
position = tf.placeholder(shape=[None, None], dtype=tf.float64, name="position")

# randoms = tf.random_uniform([5], minval=0, maxval=100, dtype=tf.float64)
randoms = [1., 1., 1., 1., 20.]
sig1 = tf.Variable(randoms[0], dtype=tf.float64, name="sig1")
sig2 = tf.Variable(randoms[1], dtype=tf.float64, name="sig2")
sig3 = tf.Variable(randoms[2], dtype=tf.float64, name="sig3")
sig4 = tf.Variable(randoms[3], dtype=tf.float64, name="sig4")

lr = tf.placeholder(shape=[], dtype=tf.float64, name='lr')

# cov_matrix1 sigma* matrix
x = ix - tf.reduce_mean(ix, 0)
xxt = tf.matmul(x, tf.transpose(x))
sig1_pow2 = sig1*sig1

cov_matrix1 = sig1_pow2 * xxt


# cov_matrix2 sigma* matrix
def env_pos(x):
    return tf.exp(((-1) * 0.5 * x) / (tf.pow(l, 2)))


l = tf.Variable(randoms[4], dtype=tf.float64, name="l")
zzt = tf.map_fn(env_pos, position)
sig2_pow2 = sig2*sig2
cov_matrix2 = sig2_pow2 * zzt

# cov_matrix3 sigma* matrix
u = tf.ones_like(xxt)
v = tf.eye(tf.shape(xxt)[0])
mask = tf.cast(u, dtype=tf.float64) - tf.cast(v, dtype=tf.float64)
weight = tf.math.multiply(mask, zzt)
zx = tf.matmul(weight, x)
zxxzt = tf.matmul(zx, tf.transpose(zx))
sig3_pow2 = sig3*sig3
cov_matrix3 = sig3_pow2 * zxxzt

# cov_matrix4 sigma* matrix
noise__ = tf.eye(tf.shape(xxt)[0])
noise = tf.cast(noise__, dtype=tf.float64)
sig4_1 = sig4
sig4_pow2 = sig4 * sig4_1
cov_matrix4 = sig4_pow2 * noise

# main covariance matrix
cov_matrix = cov_matrix1 + cov_matrix2 + cov_matrix3 + cov_matrix4

# log likelihood term1
# cov_det = tf.linalg.det(cov_matrix)
cov_inv = tf.linalg.inv(cov_matrix)
loss_term1 = tf.matmul(tf.matmul(tf.transpose(y), cov_inv), y)

# log likelihood term2
primary_loss_term2 = tf.linalg.slogdet(cov_matrix)
loss_term2 = primary_loss_term2[0] * primary_loss_term2[1]

# loss function :log likelihood without constant vairable n/2log(pi)
loglikelihood = tf.add(loss_term1, loss_term2)
loss = 0.5 * loglikelihood

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Main #####
import time

_path = '../'
_sample_name = 'P1_SAy10x1_G1'

_, locations, protein_names = loading_pure_data(path=_path, sample_name=_sample_name)
dist = distance_matrix(locations)
all_x, all_y = preprocessing_data(path=_path, sample_name=_sample_name)

loss_for_multiple_random_input = []
opt_loss_for_multiple_random_input = []
sigmas_for_multiple_random_input = []
opt_sigmas_for_multiple_random_input = []
# selected_proteins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,25]
selected_proteins = [0, 1, 5, 7, 10, 15, 21, 25]
# selected_proteins = [1]

# initial values are not important yet so i do not save theme


epochs = 400
replicates = 1

with tf.Session() as sess:
    for replica in range(replicates):
        loss_for_selected_proteins = []
        opt_loss_for_selected_proteins = []
        sigmas_for_selected_proteins = []
        opt_sigmas_selected_proteins = []

        threshold = 0.1
        learning_rate = 0.08
        learning_rate_1 = 0.08
        learning_rate_2 = 1.5

        for j in selected_proteins:
            loss_for_one_protein = []
            sigmas_for_one_protein = []
            sess.run(tf.global_variables_initializer())
            i = 0
            s1_opt, s2_opt, s3_opt, s4_opt, l_opt, a, b, c, d, lt = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            min_loss_ever, last_loss, new_loss = np.inf, -np.inf, np.inf
            t0 = time.time()
            random_values = []
            print(j, ': ')

            while (i < epochs) and (abs(last_loss - new_loss) > threshold) and (
                    last_loss == np.inf or min_loss_ever == np.inf or last_loss == min_loss_ever or (
                    (last_loss - min_loss_ever) / abs(min_loss_ever) < 10)):
                #             while( (i < epochs) and (abs(last_loss-new_loss)>threshold)):
                a, b, c, d, lt, opt, run_loss, l_term1, l_term2, cov1, cov2, cov3, cov4, s1p2, s2p2, s3p2, s4p2= sess.run(
                    [sig1, sig2, sig3, sig4, l, optimizer, loss, loss_term1, loss_term2, cov_matrix1, cov_matrix2, cov_matrix3, cov_matrix4, sig1_pow2, sig2_pow2, sig3_pow2,sig4_pow2],
                    feed_dict={ix: all_x[j], y: all_y[j], position: dist, lr: learning_rate})

                run_loss = run_loss[0]
                i += 1
                last_loss = new_loss
                new_loss = run_loss

                loss_for_one_protein.append(run_loss)
                sigmas_for_one_protein.append([a, b, c, d, lt])

                if min_loss_ever > run_loss:
                    min_loss_ever = run_loss
                    s1_opt, s2_opt, s3_opt, s4_opt, l_opt = (a, b, c, d, lt)

                    # use these parameters for test numpy reconstruction
                    # l_term1_opt, l_term2_opt, cov1_opt, cov2_opt, cov3_opt, cov4_opt = l_term1, l_term2, cov1, cov2, cov3, cov4
                    # s1p2_opt, s2p2_opt, s3p2_opt, s4p2_opt = s1p2, s2p2, s3p2, s4p2
                #                         cm1_opt, cm2_opt, cm3_opt, cm4_opt = (cm1, cm2, cm3, cm4)
                if run_loss > 1500:
                    learning_rate = learning_rate_2
                else:
                    learning_rate = learning_rate_1
                print('loss: ', run_loss)

            opt_sigmas_one_protein = [s1_opt, s2_opt, s3_opt, s4_opt, l_opt]
            opt_loss_for_one_protein = min_loss_ever

            loss_for_selected_proteins.append(loss_for_one_protein)
            opt_loss_for_selected_proteins.append(opt_loss_for_one_protein)
            sigmas_for_selected_proteins.append(sigmas_for_one_protein)
            opt_sigmas_selected_proteins.append(opt_sigmas_one_protein)

            print(i)

            # loading opt conds!
            # sess.run([tf.assign(sig1, s1_opt), tf.assign(sig2, s2_opt), tf.assign(sig3, s3_opt), tf.assign(sig4, s4_opt), tf.assign(l, l_opt)])
            print("sig1 is: ", s1_opt, "sig2 is: ", s2_opt, "sig3 is: ", s3_opt, "sig4 is: ", s4_opt, "The L is: ", l_opt, "Loss is: ", min_loss_ever)

        loss_for_multiple_random_input.append(loss_for_selected_proteins)
        opt_loss_for_multiple_random_input.append(opt_loss_for_selected_proteins)
        sigmas_for_multiple_random_input.append(sigmas_for_selected_proteins)
        opt_sigmas_for_multiple_random_input.append(opt_sigmas_selected_proteins)



saveTensorflowParams(sigmas_for_multiple_random_input,opt_sigmas_for_multiple_random_input)

svca_loss = loading_losses_of_svca(protein_names, selected_proteins, _path, _sample_name)
comparing_losses(svca_loss, opt_loss_for_multiple_random_input[0], selected_proteins)
