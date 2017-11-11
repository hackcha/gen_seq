from keras import backend as K
from keras.engine.topology import Layer, InputSpec
import tensorflow as tf

class AttLayer(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = keras.initializations.get('glorot_uniform')
        #
        # self.W_regularizer = regularizers.get(W_regularizer)
        # self.b_regularizer = regularizers.get(b_regularizer)
        #
        # self.W_constraint = constraints.get(W_constraint)
        # self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 name='{}_W'.format(self.name))
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                 initializer='glorot_uniform',
                                     name='{}_b'.format(self.name))
        else:
            self.b = None
        self.uw = self.add_weight(( input_shape[-1],),
                                 initializer='glorot_uniform',
                                  name='{}_u'.format(self.name))
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = K.dot( x, self.W)
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        eij = K.dot( eij , K.expand_dims( self.uw  ) )
        # eij = tf.reduce_sum(tf.multiply(eij, self.uw))
        # eij = tf.reduce_sum(tf.multiply(eij, self.uw))
        eij = tf.squeeze( eij , axis=-1 )
        # print( 'eij shape after : {}'.format( eij.get_shape() ))
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        out = K.sum(weighted_input, axis=1)
        return out


    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]