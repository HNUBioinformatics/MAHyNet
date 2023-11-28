import glob
from sklearn.metrics import average_precision_score
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import sklearn
from sklearn.metrics import roc_auc_score, auc, roc_curve
import seaborn as sns
import pickle

from scipy import stats
from scipy.stats import kstest

def get_label(rec):
    data = h5py.File(rec, 'r')
    sequence_code = data['sequences']
    label = data['labs']
    return label
ls = []
def AUC(label, pred):

    roc_auc_score(label, pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(label, pred)
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    print(roc_auc)

    return roc_auc


def get_model_result(result_path, data_path):
    path = result_path
    result_list = glob.glob(path + '\\*.npy')

    auc = []
    random_seed = []
    kernel_length_list=[]
    local_window = []
    data_info_list = []
    kernel_number_list = []
    mode=[]

    for rec in result_list:
        data_info = rec.split("\\")[-2]
        label_path = data_path + str(data_info) + '\\test.hdf5'
        label = get_label(label_path)
        label = np.array(label)
        pred = np.load(rec)

        local_window.extend([rec.split("_")[-3]])
        random_seed.extend([rec.split("_")[-9]])
        kernel_length_list.extend([rec.split("_")[-5]])
        kernel_number_list.extend([rec.split("_")[-11]])
        mode.extend([rec.split("_")[-1].split(".")[0]])
        auc.extend([AUC(label, pred)])
        data_info_list.extend([data_info])
    return auc, random_seed, kernel_length_list,local_window, data_info_list, kernel_number_list,mode


#You need modify path
result_path = r"F:\Download\lengent\venv\MAHyNet-main\demo\result_window\*"
data_path = "F:\Download\lengent\\venv\MAHyNet-main\demo\HDF5\\"


# initial the list
auc = []
random_seed = []
kernel_length_list=[]
local_window = []
data_info_list = []
kernel_number_list = []
data_info_list = []
mode_list=[]
# calculate all thing we need
now_auc, now_random_seed, now_kernel_length_list,now_local_window,  now_data_info_list, now_kernel_number_list ,now_mode= get_model_result(result_path, data_path)

auc.extend(now_auc)
random_seed.extend(now_random_seed)
kernel_length_list.extend( now_kernel_length_list)
local_window.extend(now_local_window)
data_info_list.extend(now_data_info_list)
kernel_number_list.extend(now_kernel_number_list)
mode_list.extend(now_mode)


dic = {
    'data_set': data_info_list,
    'AUC': auc,
    'random seed': random_seed,
    'kernel_length':kernel_length_list,
    'local_window': local_window,
    'kernel_number': kernel_number_list,
    'mode':mode_list,
}
# list to DataFrame
data_all = pd.DataFrame(dic)
data_all['random seed'] = data_all['random seed'].astype(int)
data_all['kernel_length'] = data_all['kernel_length'].astype(int)
data_all['local_window'] = data_all['local_window'].astype(int)
data_all['kernel_number'] = data_all['kernel_number'].astype(int)
data_all['mode'] = data_all['mode'].astype(int)
data_all['AUC'] = data_all['AUC'].astype(float)



# save the result to csv file
data_all.to_csv('result_window.csv', index=False)
