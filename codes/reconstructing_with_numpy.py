import math
import numpy as np
from codes.preparing_data import *
import os

# this method returns all four matrices with their multipliers + sum of theme
def making_kinships(pos, protein_index, data, sigma1, sigma2, sigma3, sigma4, l):
    t1 = making_intrinsic(protein_index, data)
    t2 = making_environment(pos=pos, l=l)
    t3 = making_cc_interaction(protein_index, data, t2)
    t1 *= sigma1 ** 2
    t2 *= sigma2 ** 2
    t3 *= sigma3 ** 2
    t4 = (sigma4 ** 2) * making_noise(pos.shape[0])
    output = [t1 + t2 + t3 + t4, t1, t2, t3, t4]

    return output


def making_intrinsic(protein_index, data):
    data = np.delete(data, protein_index, 1)
    data = data - np.mean(data, axis=0)
    output = np.matmul(data, data.T)

    return output


def making_environment(pos, l):
    env = distance_matrix(pos)
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            env[i][j] = np.exp((-0.5) * env[i][j] / (l ** 2))

    return env


def distance_matrix(position):
    distance = np.zeros((position.shape[0], position.shape[0]), dtype=np.float32)
    for i in range(position.shape[0]):
        for j in range(position.shape[0]):
            distance[i][j] = (position[i][0] - position[j][0]) ** 2 + (position[i][1] - position[j][1]) ** 2

    return distance


def making_cc_interaction(protein_index, data, env):
    data1 = np.delete(data, protein_index, 1)
    m = np.mean(data1, 0)
    data = data1 - m
    mask = np.ones_like(env) - np.eye(np.shape(env)[0])
    weight = np.multiply(mask, env)
    zx = np.matmul(weight, data)
    zxxz = np.matmul(zx, zx.T)

    return zxxz


def making_noise(n):
    return np.eye(n)


# reproducing loss with numpy
def loglikelihood(x, sigma):
    sign, logdet = np.linalg.slogdet(sigma)
    t2 = sign * logdet
    inverse = np.linalg.inv(sigma)
    t3 = np.matmul(x.T, np.matmul(inverse, x))
    #     print(t2)
    #     print(t3)

    return 0.5 * (t2 + t3)[0][0]


# test for protein number 1 where selected_proteins = [1]

_path = '../'
_sample_name = 'P1_SAy10x1_G1'

expressions, locations, protein_names = loading_pure_data(_path, _sample_name)
dist = distance_matrix(locations)
all_x, all_y = preprocessing_data(_path, _sample_name)

sigmas_for_multiple_random_input, opt_sigmas_for_multiple_random_input = loadTensorflowParams()
sigs = opt_sigmas_for_multiple_random_input[0][0]

kin = making_kinships(locations, 1, expressions, sigs[0], sigs[1], sigs[2], sigs[3], sigs[4])
print(loglikelihood(all_y[1], kin[0]))
