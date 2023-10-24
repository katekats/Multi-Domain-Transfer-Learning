from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

# class defining the custom attention layer
class AttentionNetwork(Layer):
    def __init__(self, att_dim, **kwargs):
        self.att_dim = att_dim
        super(AttentionNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        """
            Initializes inner weights, W and u, and bias b.
            """
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], self.att_dim),
                                 name='att_W',
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.att_dim,),
                                 name='att_b',
                                 initializer='random_normal',
                                 trainable=True)
        self.u = self.add_weight(shape=(self.att_dim, 1),
                                 name='att_u',
                                 initializer='random_normal',
                                 trainable=True)

        super(AttentionNetwork, self).build(input_shape)

    def call(self, x):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super().get_config()
        config.update({
            "att_dim": self.att_dim
        })
        return config
