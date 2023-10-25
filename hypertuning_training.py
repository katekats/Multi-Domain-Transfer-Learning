from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense, Dropout, TimeDistributed, Sequential
import tensorflow as tf
from kerastuner import HyperModel, RandomSearch

class HANHyperModel(HyperModel):
    def build(self, hp):
        word_encoder = Sequential([
            Bidirectional(GRU(units=hp.Int('GRU_units_1', 50, 150, 25), return_sequences=True)),
            AttentionNetwork(hp.Int('attention_dim_1', 25, 75, 25)),
            Dropout(hp.Float('dropout_1', 0.0, 0.5, 0.05))
        ])

        model = Sequential([
            TimeDistributed(word_encoder),
            Bidirectional(LSTM(units=hp.Int('LSTM_units_1', 50, 150, 25), return_sequences=True)),
            Dropout(hp.Float('dropout_2', 0.0, 0.5, 0.05)),
            Bidirectional(LSTM(units=hp.Int('LSTM_units_2', 50, 150, 25), return_sequences=True)),
            AttentionNetwork(hp.Int('attention_dim_2', 25, 75, 25)),
            Dropout(hp.Float('dropout_3', 0.0, 0.5, 0.05)),
            Dense(1, activation='sigmoid')  # Assuming binary classification
        ])

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
            metrics=['accuracy']
        )

        return model

def train(train_ds, val_ds, epochs=50):
    hypermodel = HANHyperModel()

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='random_search',
        project_name='HAN_BiLSTM'
    )

    tuner.search_space_summary()

    tuner.search(train_ds, validation_data=val_ds, epochs=epochs)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    return model, history

def test(model, test_ds):
    """Evaluates the model."""
    test_loss, test_acc = model.evaluate(test_ds)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    return test_loss, test_acc
