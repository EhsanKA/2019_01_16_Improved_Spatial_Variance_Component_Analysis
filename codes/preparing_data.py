import numpy as np
from codes.utils import *
from codes.loader import *

def preprocessing_data(path, sample_name):
    expressions, possitions, protein_names = loading_pure_data(path=path, sample_name=sample_name)

    proteins_number = len(protein_names)
    # np.delete(np.array(expressions), 5, axis=1)

    z_sample = expressions[:, 0]
    y_sample = quantile_normalise_phenotype(z_sample)
    y_sample = y_sample.reshape((y_sample.shape[0], 1))

    x_sample = np.delete(expressions, 0, axis=1)

    all_x = np.zeros((x_sample.shape[0], x_sample.shape[1]))
    all_y = np.zeros((y_sample.shape[0], y_sample.shape[1]))

    for i in range(len(protein_names)):

        z_sample = expressions[:, i]
        y_sample = quantile_normalise_phenotype(z_sample)
        y_sample = y_sample.reshape((y_sample.shape[0], 1))

        x_sample = np.delete(expressions, i, axis=1)

        all_y = np.append(all_y, y_sample, axis=0)
        all_x = np.append(all_x, x_sample, axis=0)

    all_x = all_x.reshape((proteins_number +1,x_sample.shape[0], x_sample.shape[1]))[1:, :, :]
    all_y = all_y.reshape((proteins_number +1,y_sample.shape[0], y_sample.shape[1]))[1:, :, :]

    return all_x, all_y
