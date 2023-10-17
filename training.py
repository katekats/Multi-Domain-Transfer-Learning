import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, TimeDistributed, Dropout, Dense
from attention import AttentionNetwork
from tensorflow.keras.models import Sequential


def build_model(hp):
    """
        Build model function for FT-HAN-BiLSTM architecture with hyper-parameter optimization enabled.
        """
    word_encoder = Sequential([
        Bidirectional(GRU(units=hp.Int("units_1", min_value=25, max_value=100, step=25), return_sequences=True)),
        AttentionNetwork(hp.Choice("attention_dim_1", [50, 100])),
    ])

    model = Sequential([
        TimeDistributed(word_encoder),
        Bidirectional(LSTM(units=hp.Int("units_2", min_value=25, max_value=100, step=25), return_sequences=True)),
        Bidirectional(LSTM(units=hp.Int("units_3", min_value=25, max_value=100, step=25), return_sequences=True)),
        AttentionNetwork(hp.Choice("attention_dim_2", [50, 100])),
        Dropout(hp.Float("rate", min_value=0.0, max_value=0.2, step=0.05)),
        Dense(1)
    ])
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def train(train_ds, val_ds, epochs=50, encoder=None, bilstm=False):
    """
        Builds and trains the model
        :param train_ds: training data_set
        :param val_ds: validation data_sets
        :param epochs: number of epochs for training
        :param encoder: the pre-trained encoder from auto-encoder
        :param bilstm: a flag to use a biLSTM network or not on the sentence-level network
        :return: trained model and history
        """
    
   
    # Word-level layers
    if encoder is not None:
        word_encoder = Sequential([
            encoder.layers[0],
            Bidirectional(LSTM(100, return_sequences=True)),
           Bidirectional(LSTM(50, return_sequences=True)),
            AttentionNetwork(50),
        ])
    else:
        word_encoder = Sequential([
            Bidirectional(LSTM(100, return_sequences=True)),
            Bidirectional(LSTM(50, return_sequences=True)),
            AttentionNetwork(50),
        ])

    # Sentence-level layers
    if bilstm:
        model = Sequential([
            TimeDistributed(word_encoder),
            Bidirectional(LSTM(100, return_sequences=True)),
            Bidirectional(LSTM(50, return_sequences=True)),
            AttentionNetwork(50),
            Dropout(0.1),
            Dense(1)
        ])
    else:
        model = Sequential([
            TimeDistributed(word_encoder),
            Bidirectional(LSTM(100, return_sequences=True)),
            Bidirectional(LSTM(50, return_sequences=True)),
            AttentionNetwork(50),
            Dropout(0.1),
            Dense(1)
        ])

    # Freeze the encoder-part of the auto-encoder, if it is included.
    if encoder is not None:
        model.layers[0].layer.layers[0].trainable = False

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    val_steps = val_ds.cardinality().numpy()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        
                        validation_steps=val_steps,
                        callbacks=[callback])

    return model, history


def test(model, test_ds):
    """
        Evaluates the model
        :param model: trained model
        :param test_ds: test data_sets
        :return: test loss & accuracy
        """
    if isinstance(model, tf.keras.Sequential):
        test_loss, test_acc = model.evaluate(test_ds)

        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)
        return test_loss, test_acc
    else:
        raise TypeError("The model ist not of type tf.keras.Sequential")


