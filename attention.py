from keras import backend as K
from keras.layers import Layer
from keras import initializers

class AttentionNetwork(Layer):
    def __init__(self, att_dim, W_initializer='glorot_uniform', **kwargs):
        """
        Initializes the attention network layer.
        
        Args:
        - att_dim (int): The dimension of the attention network.
        - W_initializer (str, optional): Initializer for the weight matrix.
          Default is 'glorot_uniform'.
        """
        self.att_dim = att_dim
        self.W_initializer = initializers.get(W_initializer)
        super(AttentionNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Create the layer's weights.
        
        Args:
        - input_shape (tuple): Shape of the input tensor.
        """
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], self.att_dim),
                                 name='att_W',
                                 initializer=self.W_initializer,
                                 trainable=True)
        self.b = self.add_weight(shape=(self.att_dim,),
                                 name='att_b',
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(self.att_dim, 1),
                                 name='att_u',
                                 initializer=self.W_initializer,
                                 trainable=True)

        super(AttentionNetwork, self).build(input_shape)

    def call(self, x):
        """
        Forward pass.
        
        Args:
        - x (Tensor): Input tensor.
        
        Returns:
        - Tensor: Output tensor after applying attention mechanism.
        """
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        """ Compute the output shape after the forward pass. """
        return input_shape[0], input_shape[-1]

    def get_config(self):
        """ Retrieve the configuration of the layer. """
        config = super().get_config()
        config.update({
            "att_dim": self.att_dim,
            "W_initializer": initializers.serialize(self.W_initializer)
        })
        return config

