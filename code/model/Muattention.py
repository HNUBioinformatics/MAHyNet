import keras as k
from keras import backend as K
import tensorflow as tf
import numpy as np
import  keras
from myattention import *
from keras.layers import *



class OurLayer(Layer):
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs


class Multi_Head_attention(OurLayer):
    def __init__(self, out_dim, **kwargs):
        super(Multi_Head_attention, self).__init__(**kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        super(Multi_Head_attention, self).build(input_shape)
        self.head1 = MyAttention(out_dim=self.out_dim)
        self.head2 = MyAttention(out_dim=self.out_dim)
        self.head3 = MyAttention(out_dim=self.out_dim)
        self.head4 = MyAttention(out_dim=self.out_dim)
        self.head5 = MyAttention(out_dim=self.out_dim)
        self.head6 = MyAttention(out_dim=self.out_dim)


        self.w0 = keras.layers.core.Dense(self.out_dim, use_bias=False)
    def call(self, inputs):
        # input_size = tf.shape(inputs)
        h1 = self.reuse(self.head1, inputs)
        h2 = self.reuse(self.head2, inputs)
        h3 = self.reuse(self.head3, inputs)
        h4 = self.reuse(self.head4, inputs)
        h5 = self.reuse(self.head5, inputs)
        h6 = self.reuse(self.head6, inputs)
        h_r = self.reuse(self.w0, tf.concat([h1, h2, h3, h4, h5, h6], -1))
        return h_r


    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.out_dim)



if __name__ == '__main__':
    pass
