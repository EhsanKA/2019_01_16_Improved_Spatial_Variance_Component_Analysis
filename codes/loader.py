import pandas as pd
import numpy as np
import os

## we set the root path for this code and the subdirectories are codes, pure_data, ...
_path= '../'
_sample_name = 'P1_SAy10x1_G1'
## loading positions and expressions

def loading_pure_data(path=_path, sample_name=_sample_name ):
    exp_path = path+ 'pure_data/' + sample_name
    expressions = pd.read_csv(exp_path + '/expressions.txt',sep=" ")
    posit = pd.read_csv(exp_path + '/positions.txt', sep=",", header=None)
    protein_names = list(expressions.columns)


    return np.array(expressions), np.array(posit), protein_names
  
## loading all outputs of svca model
## for now we just load for each protein 4 number as 4 gower multiplier

def loading_results_of_svca(protein_names, path=_path, sample_name='P1_SAy10x1_G1'):
    
    results_path = path + 'pure_data/'+ sample_name + '/results/'
    signatures = []
    ## r=root, d=directories, f = files
    files = []
    for r, d, f in os.walk(results_path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))


    svca_signs = []
    for i in range(len(prot_names)):
        file = pd.read_csv(results_path + prot_names[i]+ '_0_interactions_effects.txt',sep=" ")
        svca_signs.append(file.values[0].tolist())

    return svca_signs


def loading_losses_of_svca(protein_names, selected_proteins, path=_path, sample_name=_sample_name):

    path = path+ 'pure_data/' + sample_name + '/results/'
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))

    their_loss = []
    for i in selected_proteins:

        file = pd.read_csv(path + protein_names[i]+ '_0_interactions_lmls.txt',sep=" ", header= None)
        their_loss.append(file.values[0][0])
    return their_loss
