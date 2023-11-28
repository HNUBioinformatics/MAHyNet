import sys
from build_model import *
from data_load1 import *
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
from sklearn.model_selection import train_test_split
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
    model.add(Merge([model1, model2], mode='concat'))
    model.add(keras.layers.core.Dense(output_dim=150, name='Dense_l1'))
    model.add(Dropout(0.5))
    model.add(keras.layers.core.Dense(output_dim=50, name='Dense_l2'))
    model.add(Dropout(0.5))
    model.add(keras.layers.core.Dense(output_dim=1, name='Dense_l3'))
    model.add(keras.layers.Activation("sigmoid"))
    sgd = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    # set the result path
    output_path = modelsave_output_prefix + "\\" + str(data_info) 
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
    np.save(prediction_save_path, prediction)
    
    return history, prediction

# read the hyper-parameters
data_path = sys.argv[1]
data_path1 =sys.argv[2]
result_path = sys.argv[3]
data_info = sys.argv[4]
number_of_kernel = int(sys.argv[5])
random_seed = int(sys.argv[6])
local_window = int(sys.argv[7])
GPU_SET = sys.argv[8]


np.random.seed(random_seed)
random.seed(random_seed)
tf.set_random_seed(random_seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_SET
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X_test, Y_test, X, Y= get_data(data_path)
X1_test, Y1_test, X1, Y1 = get_data(data_path1)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=None)
X1_train, X1_val, Y1_train, Y1_val = train_test_split(X1, Y1, test_size=0.1, random_state=None)


input_shape= (X_train.shape[1],X_train.shape[2] )
input_shape1= (X1_train.shape[1],X1_train.shape[2] )


print(X_train.shape[1])
print(number_of_kernel)

# set the hyper-parameters
batch_size = 32
epoch_num = 100
#ker_len_list=[5,11,15,17,25,35,41]
ker_len_list=[15]
for ker_len in ker_len_list:
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








