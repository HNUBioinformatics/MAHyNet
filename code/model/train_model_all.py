import sys
from build_model import *
from data_load2 import *
import os
import numpy as np
import pickle
import random
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold
import numpy
from math import *
import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_auc_score, auc, roc_curve
import keras
import keras.callbacks
from keras.layers import Conv1D, Activation, Dropout,regularizers
import keras.backend

plt.switch_backend('agg')

def mkdir(path):
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        return True
    else:
        return False

def AUC(label, pred):
    roc_auc_score(label, pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(label, pred)
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    return roc_auc

def train_Hybrid_model(number_of_kernel,
                    ker_len, input_shape,input_shape1,
                    batch_size,
                    epoch_num,
                    data_info,
                    modelsave_output_prefix,
                    random_seed,
                    local_window_size,
                    mode):

    model1 = keras.models.Sequential()
    model1 = build_model_one(model1,
                           number_of_kernel,
                           ker_len,
                           input_shape=input_shape,
                           local_window_size=local_window_size,
                           mode=mode)
    model2 = keras.models.Sequential()
    model2 = build_model_two(model2,
                             number_of_kernel,
                             ker_len,
                             input_shape=input_shape1,
                             local_window_size=local_window_size,
                             mode=mode)
    model = keras.models.Sequential()
    model.add(Merge([model1 ,  model2], mode='concat'))
    model.add(keras.layers.core.Dense(output_dim=150, name='Dense_l1'))
    model.add(Dropout(0.5))
    model.add(keras.layers.core.Dense(output_dim=50, name='Dense_l2'))
    model.add(Dropout(0.5))
    model.add(keras.layers.core.Dense(output_dim=1, name='Dense_l3'))
    model.add(keras.layers.Activation("sigmoid"))
    sgd = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    # set the result path
    output_path = modelsave_output_prefix + "/" + str(data_info)  # ....
    mkdir(output_path)
    output_prefix = output_path + "/" \
                    + "model-KernelNum_" + str(number_of_kernel) \
                    + "_random-seed_" + str(random_seed) \
                    + "_batch-size_" + str(batch_size) \
                    + '_kernel-length_' + str(ker_len) \
                    + '_localwindow_' + str(local_window_size)\
                    + '_mode_' + str(mode)

    modelsave_output_filename = output_prefix + "_checkpointer.hdf5"
    prediction_save_path = output_prefix + '.npy'

    if os.path.exists(prediction_save_path):
        print('file has existed')
        return 0, 0
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=8,
                                                 verbose=0)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelsave_output_filename,
                                                   verbose=1,
                                                   save_best_only=True)

    history = model.fit([X_train,X1_train],
                         Y_train,
                        epochs=epoch_num,
                        batch_size=batch_size,
                        validation_data=([X_val,X1_val],Y_val),
                        verbose=0,
                        callbacks=[checkpointer, earlystopper]
                        )

    model.load_weights(modelsave_output_filename)
    prediction = model.predict([X_test,X1_test])
    auc= AUC(Y_test, prediction)
    cvscores.append(auc)
    print(auc)
    c=len(cvscores)
    if c%10==0:
        AUCs= numpy.mean(cvscores)
        np.save(prediction_save_path, AUCs)
        #np.save(prediction_save_path1, AUCs)
        print(c)
        print(AUCs)
        cvscores.clear()
  
    return history, auc


# read the hyper-parameters
data_path = sys.argv[1]
data_path1 =sys.argv[2]
result_path = sys.argv[3]
data_info = sys.argv[4]
number_of_kernel = int(sys.argv[5])
random_seed = int(sys.argv[6])
local_window = int(sys.argv[7])
GPU_SET = sys.argv[8]

# set the hyper-parameters
ker_len = 15
batch_size = 32
epoch_num = 100

np.random.seed(random_seed)
random.seed(random_seed)
tf.set_random_seed(random_seed)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_SET
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X , Y = get_data(data_path)
idx = X.shape[0]
input_shape = (X.shape[1],X.shape[2])
cvscores = []

X1, Y1 = get_data(data_path1)
idx1 = X1.shape[0]
input_shape1 = (X1.shape[1], X1.shape[2])



for i in range(10):  
    print(i, "times: ")
    print(input_shape, idx)
    print(input_shape1, idx1)
    X_test =X[int(idx * i * 0.1):int(idx * (i + 1) * 0.1), ]  
    X1_test =X1[int(idx1 * i * 0.1):int(idx1 * (i + 1) * 0.1), ]
    Y_test = Y[int(idx * i * 0.1):int(idx * (i + 1) * 0.1), ]
    Y1_test = Y1[int(idx1 * i * 0.1):int(idx1 * (i + 1) * 0.1), ]
    if i + 1 <= max(range(10)):  
        X_val = X[int(idx * (i + 1) * 0.1):int((i + 2) * idx * 0.1)] 
        X1_val = X1[int(idx1 * (i + 1) * 0.1):int((i + 2) * idx1 * 0.1)]
        Y_val = Y [int(idx * (i + 1) * 0.1):int((i + 2) * idx * 0.1)]  
        Y1_val = Y1[int(idx1 * (i + 1) * 0.1):int((i + 2) * idx1 * 0.1)]

        X_train = np.delete(X , range(int(idx * i * 0.1), int(idx * (i + 2) * 0.1)), axis=0) 
        X1_train = np.delete(X1, range(int(idx1 * i * 0.1), int(idx1 * (i + 2) * 0.1)), axis=0)
        Y_train = np.delete(Y, range(int(idx * i * 0.1), int(idx * (i + 2) * 0.1)), axis=0) 
        Y1_train = np.delete(Y1, range(int(idx1 * i * 0.1), int(idx1 * (i + 2) * 0.1)), axis=0)

    else:  
        X_val =X[:int(((i + 1) % 8) * idx * 0.1)] 
        X1_val = X1[:int(((i + 1) % 8) * idx1 * 0.1)]
        Y_val = Y[:int(((i + 1) % 8) * idx * 0.1)]
        Y1_val = Y1[:int(((i + 1) % 8) * idx1 * 0.1)]

        X_train = np.delete(X, range(int(idx * i * 0.1), int(idx * (i + 1) * 0.1)), axis=0)
        X1_train = np.delete(X1, range(int(idx1 * i * 0.1), int(idx1 * (i + 1) * 0.1)), axis=0)
        X_train = np.delete(X_train, range(int(((i + 1) % 8) * idx * 0.1)), axis=0)
        X1_train = np.delete(X1_train, range(int(((i + 1) % 8) * idx1 * 0.1)), axis=0)
        Y_train = np.delete(Y, range(int(idx * i * 0.1), int(idx * (i + 1) * 0.1)), axis=0)
        Y1_train = np.delete(Y1, range(int(idx1 * i * 0.1), int(idx1 * (i + 1) * 0.1)), axis=0)
        Y_train = np.delete(Y_train, range(int(((i + 1) % 8) * idx * 0.1)), axis=0)
        Y1_train = np.delete(Y1_train, range(int(((i + 1) % 8) * idx1 * 0.1)), axis=0)

    mode=1
    History_Soft, prediction_Soft = train_Hybrid_model(number_of_kernel=number_of_kernel,
                                                    ker_len=ker_len,
                                                    input_shape=input_shape,
                                                    input_shape1=input_shape1,
                                                    batch_size=batch_size,
                                                    epoch_num=epoch_num,
                                                    data_info=data_info,
                                                    modelsave_output_prefix=result_path,
                                                    random_seed=random_seed,
                                                    local_window_size=local_window,
                                                    mode=mode)

for i in range(10):  
    print(i, "times: ")
    print(input_shape, idx)
    print(input_shape1, idx1)
    X_test =X[int(idx * i * 0.1):int(idx * (i + 1) * 0.1), ]  
    X1_test =X1[int(idx1 * i * 0.1):int(idx1 * (i + 1) * 0.1), ]
    Y_test = Y[int(idx * i * 0.1):int(idx * (i + 1) * 0.1), ]
    Y1_test = Y1[int(idx1 * i * 0.1):int(idx1 * (i + 1) * 0.1), ]
    if i + 1 <= max(range(10)):  
        X_val = X[int(idx * (i + 1) * 0.1):int((i + 2) * idx * 0.1)] 
        X1_val = X1[int(idx1 * (i + 1) * 0.1):int((i + 2) * idx1 * 0.1)]
        Y_val = Y [int(idx * (i + 1) * 0.1):int((i + 2) * idx * 0.1)]  
        Y1_val = Y1[int(idx1 * (i + 1) * 0.1):int((i + 2) * idx1 * 0.1)]

        X_train = np.delete(X , range(int(idx * i * 0.1), int(idx * (i + 2) * 0.1)), axis=0) 
        X1_train = np.delete(X1, range(int(idx1 * i * 0.1), int(idx1 * (i + 2) * 0.1)), axis=0)
        Y_train = np.delete(Y, range(int(idx * i * 0.1), int(idx * (i + 2) * 0.1)), axis=0) 
        Y1_train = np.delete(Y1, range(int(idx1 * i * 0.1), int(idx1 * (i + 2) * 0.1)), axis=0)

    else:  
        X_val =X[:int(((i + 1) % 8) * idx * 0.1)] 
        X1_val = X1[:int(((i + 1) % 8) * idx1 * 0.1)]
        Y_val = Y[:int(((i + 1) % 8) * idx * 0.1)]
        Y1_val = Y1[:int(((i + 1) % 8) * idx1 * 0.1)]

        X_train = np.delete(X, range(int(idx * i * 0.1), int(idx * (i + 1) * 0.1)), axis=0)
        X1_train = np.delete(X1, range(int(idx1 * i * 0.1), int(idx1 * (i + 1) * 0.1)), axis=0)
        X_train = np.delete(X_train, range(int(((i + 1) % 8) * idx * 0.1)), axis=0)
        X1_train = np.delete(X1_train, range(int(((i + 1) % 8) * idx1 * 0.1)), axis=0)
        Y_train = np.delete(Y, range(int(idx * i * 0.1), int(idx * (i + 1) * 0.1)), axis=0)
        Y1_train = np.delete(Y1, range(int(idx1 * i * 0.1), int(idx1 * (i + 1) * 0.1)), axis=0)
        Y_train = np.delete(Y_train, range(int(((i + 1) % 8) * idx * 0.1)), axis=0)
        Y1_train = np.delete(Y1_train, range(int(((i + 1) % 8) * idx1 * 0.1)), axis=0)

    mode=0
    History_Soft, prediction_Soft = train_Hybrid_model(number_of_kernel=number_of_kernel,
                                                    ker_len=ker_len,
                                                    input_shape=input_shape,
                                                    input_shape1=input_shape1,
                                                    batch_size=batch_size,
                                                    epoch_num=epoch_num,
                                                    data_info=data_info,
                                                    modelsave_output_prefix=result_path,
                                                    random_seed=random_seed,
                                                    local_window_size=local_window,
                                                    mode=mode)





















