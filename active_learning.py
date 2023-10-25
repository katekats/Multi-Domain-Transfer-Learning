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
import argparse
import tf.keras.models
from enum import Enum



# Constants
USAGE = '''Usage: active_learning.py -i <model_path> -o <output_file_name> -a <approach> 
          -m <metric> [--use-outlier-detection=<Boolean>]
          Approach can be either S, W or A
          S for sentence-level fine-tuning
          W for word-level fine-tuning
          A for all-layers fine-tuning
          Metric can be either Entropy or Uncertainty'''

class Approach(Enum):
    SENTENCE = 's'
    WORD = 'w'
    ALL = 'a'

class Metric(Enum):
    ENTROPY = 'entropy'
    UNCERTAINTY = 'uncertainty'

 class Domain(Enum):
    Books = 'Books'
    Electronics = 'Electronics'
    DVD = 'DVD'
    Kitchen = 'Kitchen'
    Apparel = 'Apparel'
    Camera = 'Camera'
    Health = 'Health'
    Music = 'Music'
    Toys = 'Toys'
    Video = 'Video'
    Baby = 'Baby'
    Magazines = 'Magazines'
    Software = 'Software'
    Sports = 'Sports'
    IMDb = 'IMDb'
    MR = 'MR'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Active Learning Script')
    parser.add_argument('-i', '--input', required=True, help='Model path')
    parser.add_argument('-o', '--output', required=True, help='Output file name')
    parser.add_argument('-a', '--approach', choices=[e.value for e in Approach], required=True, help='Fine-tuning approach')
    parser.add_argument('-m', '--metric', choices=[e.value for e in Metric], required=True, help='Metric type')
    parser.add_argument('--use-outlier-detection', type=bool, default=False, help='Use outlier detection')
    args = parser.parse_args()
    
    return args

def validate_environment():
    if not os.path.exists('./prepared_datasets/prepared_datasets'):
        print('It seems the ./prepared_datasets folder is missing. '
              'Please, make sure to pull the latest changes and to unzip the prepared_datasets.zip file')
        sys.exit(1)

    if not os.path.exists('./embeddings') or not os.path.exists('./embeddings/general_embeddings.bin'):
        print('No embeddings were found under \'./embeddings/general_embeddings.bin\'.\n'
              'Please, ensure that the fasttext embeddings were produced and exist in the said path.\n'
              'Also ensure that the input model was trained with these embeddings.')
        sys.exit(1)

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

def print_approach_message(approach):
    approach_messages = {
        Approach.SENTENCE.value: "Sentence-level fine-tuning approach adopted.",
        Approach.WORD.value: "Word-level fine-tuning approach adopted.",
        Approach.ALL.value: "All-layers fine-tuning approach adopted."
    }
    print(approach_messages.get(approach, "Unknown approach."))

def print_metric_message(metric):
    metric_messages = {
        Metric.ENTROPY.value: "Active Learning with Entropy Sampling will be applied.",
        Metric.UNCERTAINTY.value: "Active Learning with Uncertainty Sampling will be applied"
    }
    print(metric_messages.get(metric, "Unknown metric."))

def preprocess_domain_data(domain_train_ds, domain_val_ds, domain_test_ds, preprocessor):
    def preprocess(x, y):
        processed_x = tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32)
        shape = [x.shape[0], max_sentences, max_len, fasttext_dimension]
        return tf.ensure_shape(processed_x, shape), y

    return (
        domain_train_ds.map(preprocess),
        domain_val_ds.map(preprocess),
        domain_test_ds.map(preprocess)
    )

def create_active_learner(metric, classifier):
    if metric == 'entropy':
        return ActiveLearner(estimator=classifier, query_strategy=entropy_sampling, verbose=1)
    return ActiveLearner(estimator=classifier, query_strategy=uncertainty_sampling, verbose=1)

def filter_outliers(x_train, y_train):
    clf = IsolationForest(random_state=3)
    clf.fit(x_train)
    pred = clf.predict(x_train)
    idx = np.where(pred == 1)
    return x_train[idx], y_train[idx]

def train_and_evaluate_on_ratios(domain_preprocessed_train_ds, x_train, y_train, model_path, tl_approach, results, step):
    for i, ratio in enumerate([0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.75]):
        model = tf.keras.models.load_model(model_path, custom_objects={"AttentionNetwork": AttentionNetwork})
        classifier = KerasClassifier(lambda: model)
        classifier.model = model

        learner = create_active_learner(al_metric, classifier)
        query_idx, _ = learner.query(x_train, n_instances=math.ceil(len(x_train) * ratio), verbose=0)

        domain_preprocessed_train_ds = tf.data.Dataset.from_tensor_slices((x_train[query_idx], y_train[query_idx]))\
            .shuffle(32).batch(batch_size)

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-5),
            metrics=['accuracy']
        )
        val_steps = domain_preprocessed_val_ds.cardinality().numpy()
        model.fit(
            domain_preprocessed_train_ds, epochs=10,
            validation_data=domain_preprocessed_val_ds, validation_steps=val_steps,
            callbacks=[callback]
        )

        results[step, i, 0], results[step, i, 1] = model.evaluate(domain_preprocessed_test_ds)
    return results

def main():
    # Argument Parsing and Environment Validation
    args = parse_arguments()
    model_path = args.input
    output_file = args.output
    tl_approach = args.approach
    al_metric = args.metric
    use_anomaly_detection = args.use_outlier_detection
    
    print_approach_message(tl_approach)
    print_metric_message(al_metric)
    validate_environment()

   

    # Initialization
    max_len = 20
    max_sentences = 30
    batch_size = 32
    
    # Testing the model exist in the given path
    model = tf.keras.models.load_model(model_path, custom_objects={"AttentionNetwork": AttentionNetwork})

    preprocessor = Preprocessor(max_sentences, max_len)
    preprocessor.set_fasttext_model('./embeddings/general_embeddings.bin')
    fasttext_dimension = preprocessor.fasttext_model.get_dimension()

    results = np.zeros((16, 7, 2), dtype='float32')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    # Main Logic
    for step, d in enumerate(Domain):
        domain_train_ds, domain_val_ds, domain_test_ds = data_loader.load_data(
            './prepared_datasets/prepared_datasets', domain=d, batch_size=32
        )
        domain_preprocessed_train_ds, domain_preprocessed_val_ds, domain_preprocessed_test_ds = \
            preprocess_domain_data(domain_train_ds, domain_val_ds, domain_test_ds, preprocessor)

        x_train, y_train = zip(*[(r.numpy(), l.numpy()) for r, l in domain_preprocessed_train_ds.unbatch()])

        if use_anomaly_detection:
            x_train, y_train = filter_outliers(x_train, y_train)

        results = train_and_evaluate_on_ratios(
            domain_preprocessed_train_ds, x_train, y_train, model_path, tl_approach, results, step
        )

    # Result Output
    for f, d in enumerate(Domain):
        print(f'{d.value}')
        for i, ratio in enumerate([0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.75]):
            print(f'Ratio: {ratio}\t{str(results[f, i, 0]).replace(".", ",")}\t{str(results[f, i, 1]).replace(".", ",")}')

    # Saving Results
    if not os.path.exists('./results'):
        print('Creating ./results folder...')
        os.mkdir('./results')
    pickle.dump(results, open(output_file, "wb"))
    print(f'The model was saved to ./results/{output_file}')

if __name__ == '__main__':
    main()