import math
import pickle

import numpy as np
import tensorflow as tf
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, uncertainty_sampling
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import IsolationForest

import data_loader
from attention import AttentionNetwork
from data_loader import Domain
from preprocessor import Preprocessor
import sys
import getopt
import os

max_len = 20
max_sentences = 30
batch_size = 32

model_path = None
output_file = None
tl_approach = "s"
al_metric = "entropy"
use_anomaly_detection = False
if "-i" not in sys.argv or "-o" not in sys.argv or "-a" not in sys.argv or "-m" not in sys.argv:
    print('Usage: active_learning.py -i <model_path> -o <output_file_name> -a <approach> '
          '-m <metric> [--use-outlier-detection=<Boolean>]\n'
          'Approach can be either S, W or A\n'
          'S for sentence-level fine-tuning\n'
          'W for word-level fine-tuning\n'
          'A for all-layers fine-tuning\n\n'
          'Metric can be either Entropy or Uncertainty')
    exit(2)

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:a:o:m:", ["use-outlier-detection="])
except getopt.GetoptError:
    print('Usage: active_learning.py -i <model_path> -o <output_file_name> -a <approach> '
          '-m <metric> [--use-outlier-detection=<Boolean>]\n'
          'Approach can be either S, W or A\n'
          'S for sentence-level fine-tuning\n'
          'W for word-level fine-tuning\n'
          'A for all-layers fine-tuning\n\n'
          'Metric can be either Entropy or Uncertainty')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Usage: active_learning.py -i <model_path> -o <output_file_name> -a <approach> '
              '-m <metric> [--use-outlier-detection=<Boolean>]\n'
              'Approach can be either S, W or A\n'
              'S for sentence-level fine-tuning\n'
              'W for word-level fine-tuning\n'
              'A for all-layers fine-tuning\n\n'
              'Metric can be either Entropy or Uncertainty')
        sys.exit()
    elif opt == "-i":
        model_path = arg
    elif opt == "-o":
        output_file = arg
    elif opt == "-a":
        if arg.lower() == "w":
            print("Word-level fine-tuning approach adopted.")
            tl_approach = "w"
        if arg.lower() == "a":
            print("All-layers fine-tuning approach adopted.")
            tl_approach = "a"
        else:
            print("Sentence-level fine-tuning approach adopted.")
            tl_approach = "s"
    elif opt == "-m":
        if arg.lower() == "entropy":
            print("Active Learning with Entropy Sampling will be applied.")
            al_metric = "entropy"
        else:
            print("Active Learning with Uncertainty Sampling will be applied")
            al_metric = "uncertainty"
    elif opt == "--use-outlier-detection":
        use_anomaly_detection = (arg.lower() == 'true')

# Testing the model exist in the given path
model = tf.keras.models.load_model(model_path, custom_objects={"AttentionNetwork": AttentionNetwork})

# Making sure the necessary folders exist
if not os.path.exists('./prepared_datasets/prepared_datasets'):
    print('It seems the ./prepared_datasets folder is missing. '
          'Please, make sure to pull the latest changes and to unzip the prepared_datasets.zip file')
    sys.exit(1)

if not os.path.exists('./embeddings') or not os.path.exists('./embeddings/general_embeddings.bin'):
    print('no embeddings were found under \'./embeddings/general_embeddings.bin\'.\n'
          'Please, make sure that the fasttext embeddings were produced and exist in the said path.\n'
          'It is also important to that the input model was trained with these embeddings.')
    sys.exit(1)


# preprocessor
preprocessor = Preprocessor(max_sentences, max_len)
preprocessor.set_fasttext_model('./embeddings/general_embeddings.bin')
fasttext_dimension = preprocessor.fasttext_model.get_dimension()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
results = np.zeros((16, 7, 2), dtype='float32')
step = 0

for d in Domain:
    domain_train_ds, domain_val_ds, domain_test_ds = data_loader.load_data(
        './prepared_datasets/prepared_datasets', domain=d, batch_size=32)

    domain_preprocessed_train_ds = domain_train_ds.map(lambda x, y: (
        tf.ensure_shape(
            tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
            [x.shape[0], max_sentences, max_len, fasttext_dimension]), y)
                                         )
    domain_preprocessed_val_ds = domain_val_ds.map(lambda x, y: (
        tf.ensure_shape(
            tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
            [x.shape[0], max_sentences, max_len, fasttext_dimension]), y)
                                     )
    domain_preprocessed_test_ds = domain_test_ds.map(lambda x, y: (
        tf.ensure_shape(
            tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
            [x.shape[0], max_sentences, max_len, fasttext_dimension]), y)
                                       )

    i = 0

    x_train = []
    y_train = []

    for r, l in domain_preprocessed_train_ds.unbatch():
        x_train.append(r.numpy())
        y_train.append(l.numpy())

    x_train = np.array(x_train)
    y_train = np.array(y_train)
   # print(x_train)
    # Isolation Forest
    if use_anomaly_detection:
        # Fit outlier detection on data
        clf = IsolationForest(random_state=3)
        print(x_train.shape)
        clf.fit(x_train)
        # Predict
        pred = clf.predict(x_train)
        # Drop the outliers from x_train and y_train
        idx = np.where(pred == 1)
        x_train = x_train[idx]
        y_train = y_train[idx]

    for ratio in [0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.75]:
        model = tf.keras.models.load_model(model_path, custom_objects={"AttentionNetwork": AttentionNetwork})
        classifier = KerasClassifier(lambda: model)
        classifier.model = model

        # initialize ActiveLearner
        if al_metric == 'entropy':
            learner = ActiveLearner(
                estimator=classifier,
                query_strategy=entropy_sampling,
                verbose=1
            )
        else:
            learner = ActiveLearner(
                estimator=classifier,
                query_strategy=uncertainty_sampling,
                verbose=1
            )
        # Query instances
        query_idx, query_instance = learner.query(x_train, n_instances=math.ceil(len(x_train) * ratio), verbose=0)

        # Select training instances
        domain_preprocessed_train_ds = tf.data.Dataset.from_tensor_slices((x_train[query_idx], y_train[query_idx]))\
            .shuffle(32).batch(batch_size)

        # Freeze layers
        if tl_approach == 's':
            model.layers[0].trainable = False
            model.layers[1].trainable = False
        elif tl_approach == 'w':
            for layer in model.layers:
                layer.trainable = False
            model.layers[0].trainable = True
        #elif tl_approach == 'a':
        #    model.layers[1].trainable = False
         #   for layer in model.layers:
       #         layer.trainable = False
        #    model.layers[2].trainable = True
               # model.layers[4].trainable = True

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-5),
                      metrics=['accuracy'])

        # Train
        val_steps = domain_preprocessed_val_ds.cardinality().numpy()
        model.fit(domain_preprocessed_train_ds,
                  epochs=10,
                  validation_data=domain_preprocessed_val_ds,
                  validation_steps=val_steps,
                  callbacks=[callback])

        # Evaluate
        results[step, i, 0], results[step, i, 1] = model.evaluate(domain_preprocessed_test_ds)

        i += 1
    step += 1

# Print domain results to console
f = 0
for d in Domain:
    print(f'{d.value}')
    for i, ratio in enumerate([0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.75]):
        print(f'Ratio: {ratio}\t{str(results[f, i, 0]).replace(".", ",")}\t{str(results[f, i, 1]).replace(".", ",")}')
    f += 1

# Save results to disk
if not os.path.exists('./results'):
    print('Creating ./results folder...')
    os.mkdir('./results')
pickle.dump(results, open(output_file, "wb"))
print(f'The model was saved to ./results/{output_file}')
