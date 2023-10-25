from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, TimeDistributed
import tensorflow as tf

from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense, Dropout, TimeDistributed, Sequential
import tensorflow as tf

LSTM_UNITS_1 = 100
LSTM_UNITS_2 = 50
ATTENTION_DIM = 50
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-4

def build_model():
    """Build model function for FT-HAN-BiLSTM architecture."""
    word_encoder = Sequential([
        Bidirectional(GRU(units=LSTM_UNITS_1, return_sequences=True)),
        AttentionNetwork(ATTENTION_DIM),
        Dropout(DROPOUT_RATE)
    ])

    model = Sequential([
        TimeDistributed(word_encoder),
        Bidirectional(LSTM(units=LSTM_UNITS_1, return_sequences=True)),
        Dropout(DROPOUT_RATE),
        Bidirectional(LSTM(units=LSTM_UNITS_2, return_sequences=True)),
        AttentionNetwork(ATTENTION_DIM),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  metrics=['accuracy'])
    return model

def train(train_ds, val_ds, epochs=50, encoder=None):
    """Builds and trains the model."""
    model = build_model()

    if encoder:
        model.layers[0].layer.layers[0].trainable = False

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    val_steps = val_ds.cardinality().numpy()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        validation_steps=val_steps,
                        callbacks=[callback])
    return model, history

def test(model, test_ds):
    """Evaluates the model."""
    test_loss, test_acc = model.evaluate(test_ds)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    return test_loss, test_acc
