import os
import sys
from sys import exit
import argparse

import fasttext
import tensorflow as tf

import data_loader
import training
from preprocessor import Preprocessor
from data_loader import Domain

import os
import argparse
import tensorflow as tf

# Constants
MAX_LEN = 50
MAX_SENTENCES = 30
BATCH_SIZE = 128

# Functions
def print_usage_and_exit():
    print('Usage: train_general_script.py [-h] [--bilstm=<Boolean>] [--encoder=<path_to_encoder>] [--exclude=<domain>] -n <model_name>\n'
          ' --bilstm, --encoder, and --exclude are optional arguments.')
    sys.exit(2)

# Setup argument parser
parser = argparse.ArgumentParser(description='Train General Script')
parser.add_argument('-n', required=True, type=str, help='Model name')
parser.add_argument('--bilstm', type=lambda x: (x.lower() == 'true'), default=False, help='Use BiLSTM')
parser.add_argument('--encoder', type=str, help='Path to encoder')
parser.add_argument('--exclude', type=str, help='Domain to exclude')

args = parser.parse_args()

# Assign values from arguments
model_name = args.n
use_Bilstm = args.bilstm
encoder_path = args.encoder
excluded_domain = args.exclude

# Load the encoder
if encoder_path:
    encoder = tf.keras.models.load_model(encoder_path)

# Check exclusion domain
if excluded_domain and excluded_domain not in [e.value for e in Domain]:
    print(f'--exclude parameter input: {excluded_domain} is an unknown domain. If you want to exclude a domain from training, '
          f'please choose from {[e.value for e in Domain]}')
    sys.exit(1)

# Directory checking and creation
directories = {
    './prepared_datasets/prepared_datasets': 'It seems the ./prepared_datasets folder is missing. Please, make sure to pull the latest changes and to unzip the prepared_datasets.zip file',
    './embeddings': 'Creating ./embeddings folder...',
    './embeddings/general_embeddings.bin': 'It seems the ./prepared_datasets/prepared_datasets folder is incomplete. Please, make sure to pull the latest changes and to unzip the prepared_datasets.zip file'
}

for dir_path, error_message in directories.items():
    if not os.path.exists(dir_path):
        if dir_path == './embeddings':
            os.mkdir(dir_path)
        else:
            print(error_message)
            sys.exit(1)

if not os.path.exists('./embeddings/general_embeddings.bin') and not os.path.exists('./prepared_datasets/prepared_datasets/ALL_corpus.txt'):
    print('Producing general embeddings using FastText...')



    general_embeddings = fasttext.train_unsupervised('./prepared_datasets/prepared_datasets/ALL_corpus.txt',dim=100, minCount=1)
    general_embeddings.save_model('./embeddings/general_embeddings.bin')

# Load data
train_ds, val_ds, test_ds = data_loader.load_data('./prepared_datasets/prepared_datasets', batch_size=batch_size, exclude_domain=excluded_domain)

# preprocessor
preprocessor = Preprocessor(max_sentences, max_len)
preprocessor.set_fasttext_model('./embeddings/general_embeddings.bin')

# Fasttext:
#fasttext_dimension = preprocessor.fasttext_model.get_dimension()
#preprocessed_train_ds = train_ds.map(lambda x, y: (
#    tf.ensure_shape(
#        tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
#        [x.shape[0], max_sentences, max_len, fasttext_dimension]), y),
##                                     num_parallel_calls=tf.data.experimental.AUTOTUNE,
 #                                    deterministic=False
 #                                    )
#preprocessed_val_ds = val_ds.map(lambda x, y: (
#    tf.ensure_shape(
#        tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
#        [x.shape[0], max_sentences, max_len, fasttext_dimension]), y),
 #                                #num_parallel_calls=tf.data.experimental.AUTOTUNE,
 #                                #deterministic=False
 #                                )
#preprocessed_test_ds = test_ds.map(lambda x, y: (
 #   tf.ensure_shape(
 #       tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
  #      [x.shape[0], max_sentences, max_len, fasttext_dimension]), y),
                                  # num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                  # deterministic=False
  #                                 )
# Training
#model, m_history = training.train(preprocessed_train_ds, preprocessed_val_ds,
   #                               epochs=100, bilstm=use_Bilstm, encoder=encoder)#

# Fasttext:
fasttext_dimension = preprocessor.fasttext_model.get_dimension()
preprocessed_train_ds = train_ds.map(lambda x, y: (
    tf.ensure_shape(
        tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
        [x.shape[0], max_sentences, max_len, fasttext_dimension]), y))
preprocessed_val_ds = val_ds.map(lambda x, y: (
    tf.ensure_shape(
        tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
        [x.shape[0], max_sentences, max_len, fasttext_dimension]), y))
preprocessed_test_ds = test_ds.map(lambda x, y: (
    tf.ensure_shape(
        tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
        [x.shape[0], max_sentences, max_len, fasttext_dimension]), y))
# Training
model, m_history = training.train(preprocessed_train_ds, preprocessed_val_ds,
                                  epochs=50, bilstm=use_Bilstm, encoder=encoder)

# Evaluating
print("Performance on testing data")
model.evaluate(preprocessed_test_ds)

for d in Domain:
    print(f'Performance on {d.value} domain')
    domain_train_ds, domain_val_ds, domain_test_ds = data_loader.load_data(
        './prepared_datasets/prepared_datasets', domain=d, batch_size=32)

    domain_preprocessed_test_ds = domain_test_ds.map(lambda x, y: (
        tf.ensure_shape(
            tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
            [x.shape[0], max_sentences, max_len, fasttext_dimension]), y)
                                       )
    model.evaluate(domain_preprocessed_test_ds)

# Save the model to disk
if not os.path.exists('./models'):
    print('Creating ./models folder...')
    os.mkdir('./models')
model.save(f'models/{model_name}.h5', save_format='h5')
print(f'The model was saved to ./models/{model_name}.h5')
