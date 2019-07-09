import numpy as np
import pandas as pd
import scipy
import scipy as sp
from scipy import special
import tensorflow as tf
import matplotlib.pyplot as plt
from codes.reconstructing_with_numpy import *
import os


####### functions

def distance_matrix(position):
    distance = np.zeros((position.shape[0], position.shape[0]), dtype=np.float64)
    for i in range(position.shape[0]):
        for j in range(position.shape[0]):
            distance[i][j] = (position[i][0] - position[j][0]) ** 2 + (position[i][1] - position[j][1]) ** 2
    return distance


def env_pos(x):
    return tf.exp(((-1) * 0.5 * x) / (tf.pow(l, 2)))


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
    plt.title("comparing the losses")
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


def producing_gowers(protein_index, expressions, positions, sigmas):
    kinships = making_kinships(positions, protein_index, expressions, sigmas[0], sigmas[1], sigmas[2], sigmas[3],
                               sigmas[4])
    g1 = 1. / covar_rescaling_factor(kinships[1])
    g2 = 1. / covar_rescaling_factor(kinships[2])
    g3 = 1. / covar_rescaling_factor(kinships[3])
    g4 = 1. / covar_rescaling_factor(kinships[4])
    return [g1, g2, g3, g4]

# a test for before function

# _path = '../'
# _sample_name = 'P1_SAy10x1_G1'
#
# expressions, locations, protein_names = loading_pure_data(_path, _sample_name)
# dist = distance_matrix(locations)
# all_x, all_y = preprocessing_data(_path, _sample_name)
#
# sigmas_for_multiple_random_input, opt_sigmas_for_multiple_random_input = loadTensorflowParams()
# sigs = opt_sigmas_for_multiple_random_input[0][0]
#
# a = producing_gowers(1, expressions, locations, sigs)