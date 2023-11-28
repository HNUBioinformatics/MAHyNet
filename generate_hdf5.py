import numpy as np
import pandas as pd
import h5py
import glob
import os
#from unicode import *

def seq_to_matrix(seq,seq_matrix,seq_order):

    for i in range(len(seq)):
        if ((seq[i] == 'A') | (seq[i] == 'a')):
            seq_matrix[seq_order, i, 0] = 1
        if ((seq[i] == 'C') | (seq[i] == 'c')):
            seq_matrix[seq_order, i, 1] = 1
        if ((seq[i] == 'G') | (seq[i] == 'g')):
            seq_matrix[seq_order, i, 2] = 1
        if ((seq[i] == 'U') | (seq[i] == 'u')):
            seq_matrix[seq_order, i, 3] = 1

    return seq_matrix

def genarate_matrix_for_train(seq_shape,seq_series):
    """
    genarate matrix for train
    :param shape: (seq number, sequence_length, 4)
    :param seq_series: dataframe of all sequences
    :return:seq
    """
    seq_matrix = np.zeros(seq_shape)
    for i in range(seq_series.shape[0]):
        seq_tem = seq_series[i]
        seq_matrix = seq_to_matrix(seq_tem, seq_matrix, i)
    return seq_matrix


def mkdir(path):
    """
    make dictionary
    :param path
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

    
def generate_dataset_matrix(file_path):
    """
    generate matrix of the data set(the path)
    :param file_path:
    :return:
    """
    filenames = glob.glob(file_path+"\*.data")
    for allFileFa in filenames:
        AllTem = allFileFa.split("\\")[-1].split(".")[0]
        #print(AllTem)
        output_dir = allFileFa.split(AllTem)[0].replace("motif_discovery", "HDF5")
        #print(allFileFa+'#')
        #print(AllTem+'*')
        #print(output_dir+'%')
        SeqLen = 81
        ChipSeqlFileFa = pd.read_csv(allFileFa, sep=' ', header=None, index_col=None,engine ='python')
        #print(ChipSeqlFileFa,'')
        seq_series = np.asarray(ChipSeqlFileFa.loc[:, 1])
        seq_name = np.asarray(ChipSeqlFileFa.loc[:, 0])
        seq_matrix_out = genarate_matrix_for_train((seq_series.shape[0], SeqLen, 4), seq_series)
        seq_label_out = np.asarray(ChipSeqlFileFa.loc[:, 2])
        #print(seq_name)
        mkdir(output_dir)
        f = h5py.File(output_dir + AllTem + ".hdf5",'w')
        #seq_label_out_encode = []
       # for j in seq_label_out:
            #seq_label_out_encode.append(j.encode())

        f.create_dataset("sequences", data=seq_matrix_out)
        f.create_dataset("labs", data=seq_label_out)
        f.create_dataset("seq_idx", data=seq_name)
        f.close()
        print(f)

if __name__ == '__main__':
    base = {0:"A",1:"C",2:"G",3:"U"}
    # You need modify path
    allFileFaList = glob.glob(r"F:\Download\lengent\venv\MAHyNet-main\demo\motif_discovery\*")
    for FilePath in allFileFaList:
        generate_dataset_matrix(FilePath)
