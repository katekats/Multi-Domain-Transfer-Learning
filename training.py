from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, TimeDistributed
import tensorflow as tf

LSTM_UNITS_1 = 100
LSTM_UNITS_2 = 50
ATTENTION_DIM = 50
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-4

def build_model(hp):
    """Build model function for FT-HAN-BiLSTM architecture with hyper-parameter optimization enabled."""
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

def train(train_ds, val_ds, epochs=50, encoder=None):
    """Builds and trains the model."""
    word_encoder_layers = [
        Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True)),
        Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=True)),
        AttentionNetwork(ATTENTION_DIM)
    ]

    if encoder:
        word_encoder_layers.insert(0, encoder.layers[0])

    word_encoder = Sequential(word_encoder_layers)

    model = Sequential([
        TimeDistributed(word_encoder),
        Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True)),
        Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=True)),
        AttentionNetwork(ATTENTION_DIM),
        Dropout(DROPOUT_RATE),
        Dense(1)
    ])

    if encoder:
        model.layers[0].layer.layers[0].trainable = False

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  metrics=['accuracy'])

    val_steps = val_ds.cardinality().numpy()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        validation_steps=val_steps,
                        callbacks=[callback])
    return model, history

def test(model, test_ds):
    """Evaluates the model."""
    if isinstance(model, tf.keras.Sequential):
        test_loss, test_acc = model.evaluate(test_ds)
        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)
        return test_loss, test_acc
    else:
        raise TypeError("The model is not of type tf.keras.Sequential.")
