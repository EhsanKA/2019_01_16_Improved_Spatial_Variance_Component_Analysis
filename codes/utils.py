import numpy as np
import pandas as pd
import scipy
import scipy as sp
from scipy import special
import tensorflow as tf
import matplotlib.pyplot as plt
import os


####### functions

def distance_matrix(position):
    distance = np.zeros((position.shape[0], position.shape[0]), dtype=np.float64)
    for i in range(position.shape[0]):
        for j in range(position.shape[0]):
            distance[i][j] = (position[i][0] - position[j][0]) ** 2 + (position[i][1] - position[j][1]) ** 2
    return distance


# def env_pos(x):
#     return tf.exp(((-1) * 0.5 * x) / (tf.pow(l, 2)))


# their functions
def quantile_normalise_phenotype(phenotype):
    # take ranks and scale to uniform
    phenotype_order = pd.Series(phenotype).rank().astype(float)
    phenotype_order = phenotype_order.values
    phenotype_order /= (phenotype_order.max() + 1.)

    # transform uniform to gaussian using probit
    mean = 0
    sigma = 1
    phenotype_norm = mean + 2. ** 0.5 * sigma * scipy.special.erfinv(2. * phenotype_order - 1.)
    phenotype_norm = np.reshape(phenotype_norm, [len(phenotype_norm), 1])

    return phenotype_norm


def covar_rescaling_factor_efficient(C):
    """
    Returns the rescaling factor for the Gower normalizion on covariance matrix C
    the rescaled covariance matrix has sample variance of 1
    """
    n = tf.shape(C)[0]
    #     n = C.shape[0]
    P = tf.eye(n, dtype=tf.float64) - tf.math.divide(tf.ones((n, n), dtype=tf.float64), tf.cast(n, tf.float64))
    #     P = sp.eye(n) - sp.ones((n,n))/float(n)
    CP = C - tf.reduce_mean(C, 0)
    #     CP = C - C.mean(0)[:, sp.newaxis]
    trPCP = tf.math.reduce_sum(tf.matmul(P, tf.transpose(CP)))
    #     trPCP = sp.sum(P * CP)
    r = tf.cast(n - 1, tf.float64) / trPCP
    #     r = (n-1) / trPCP
    return r


def covar_rescaling_factor(C):
    """
    Returns the rescaling factor for the Gower normalizion on covariance matrix C
    the rescaled covariance matrix has sample variance of 1
    """
    n = C.shape[0]
    P = sp.eye(n) - sp.ones((n, n)) / float(n)
    trPCP = sp.trace(sp.dot(P, sp.dot(C, P)))
    r = (n - 1) / trPCP
    return r


def comparing_losses(svca_losses, tensorflow_loss, selected_proteins):
    plt.figure(figsize=(9, 7))
    plt.plot(selected_proteins, tensorflow_loss, c='orange', label="Tensorflow")
    plt.plot(selected_proteins, svca_losses, c='blue', label="SVCA")
    plt.legend(loc='upper right')
    plt.xlabel("proteins " + str(selected_proteins))
    plt.ylabel("loss without fix term")
    plt.title("comparing the losses with multiply l init value is: 0.2")
    plt.savefig('../results/comparison/comparing_the_losses_of_SVCA_and_Tensorflow_model.png', dpi=500)
    plt.show()


def saveTensorflowParams(sigmas_for_multiple_random_input, opt_sigmas_for_multiple_random_input):
    np.save('../experiments_data/reproducing/sigmas_for_multiple_random_input',
            np.array((sigmas_for_multiple_random_input)))
    np.save('../experiments_data/reproducing/opt_sigmas_for_multiple_random_input',
            np.array((opt_sigmas_for_multiple_random_input)))


def loadTensorflowParams():
    sigmas_for_multiple_random_input = np.load('../experiments_data/reproducing/sigmas_for_multiple_random_input.npy')
    opt_sigmas_for_multiple_random_input = np.load(
        '../experiments_data/reproducing/opt_sigmas_for_multiple_random_input.npy')
    return sigmas_for_multiple_random_input, opt_sigmas_for_multiple_random_input


