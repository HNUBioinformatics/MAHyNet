from Muattention import *
import keras
import keras.callbacks
from keras.layers import Conv1D, Activation, Dropout,regularizers
import keras.backend
from myattention import MyAttention
from keras.layers.core import Flatten

from ePooling import *
def build_model_one(model_template, number_of_kernel, kernel_length, input_shape, local_window_size=19,mode=0):

    def Relu(x):
        return keras.backend.relu(x, alpha=0.5, max_value=10)

    model_template.add(Conv1D(
        input_shape=input_shape,
        kernel_size=kernel_length,
        filters=number_of_kernel,
        padding='valid',
        strides=1))
    model_template.add(Activation(Relu))

    model_template.add(keras.layers.pooling.MaxPooling1D(pool_length=local_window_size,
                                                         stride=None, border_mode='valid')) 
    model_template.add(Dropout(0.7))
    model_template.add(keras.layers.GRU(
        output_dim=number_of_kernel,
        return_sequences=True))
    model_template.add(Activation(Relu))
    model_template.add(Multi_Head_attention(out_dim=number_of_kernel))
    model_template.add(keras.layers.pooling.MaxPooling1D(pool_length=15,
                                                         stride=None, border_mode='same'))
    model_template.add(GlobalExpectationPooling1D(mode=mode, m_trainable=False, m_value=1))
    return model_template

def build_model_two(model_template, number_of_kernel, kernel_length, input_shape, local_window_size=19,mode=0):
        def Relu(x):
            return keras.backend.relu(x, alpha=0.5, max_value=10)

        model_template.add(Conv1D(
            input_shape=input_shape,
            kernel_size=kernel_length,
            filters=number_of_kernel,
            padding='valid',
            strides=1))
        model_template.add(Activation(Relu))
        model_template.add(keras.layers.pooling.MaxPooling1D(pool_length=local_window_size,
                                                             stride=None, border_mode='valid'))
        s = 1
        wl = (81 - kernel_length) / s + 1
        H1 = (wl - local_window_size) / s + 1
        wd = (4 - kernel_length) / s + 1
        Hd = (wd - local_window_size) / s + 1
        input = (H1, Hd)

        model_template.add(Conv1D(
            input_shape=input,
            kernel_size=kernel_length,
            filters=number_of_kernel,
            padding='same',
            strides=1))

        model_template.add(Activation(Relu))
        model_template.add(Multi_Head_attention(out_dim=number_of_kernel))
        
        model_template.add(keras.layers.pooling.MaxPooling1D(pool_length=15,
                                                         stride=None, border_mode='same'))
        model_template.add(GlobalExpectationPooling1D(mode=mode, m_trainable=False, m_value=1))
        
        model_template.add(Dropout(0.7))
        return model_template




if __name__ == '__main__':

    pass
