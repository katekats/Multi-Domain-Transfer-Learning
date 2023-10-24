import getopt
import os
import sys
from sys import exit

import fasttext
import tensorflow as tf

import data_loader
import training
from preprocessor import Preprocessor
from data_loader import Domain

max_len = 50
max_sentences = 30
batch_size = 128

#import zipfile
#with zipfile.ZipFile("prepared_datasets.zip", 'r') as zip_ref:
#    zip_ref.extractall("prepared_datasets")

use_Bilstm = False
encoder_path = None
encoder = None
model_name = None
excluded_domain = None
if "-n" not in sys.argv:
    print('Usage: train_general_script.py [--bilstm=<Boolean> --encoder=<path_to_encoder> --exclude=<domain>] -n  "FT-HAN-BiLSTM"\n'
          ' --bilstm, --encoder and --exclude are optional arguments.')
    exit(2)

try:
    opts, args = getopt.getopt(sys.argv[1:], "hn:", ["bilstm=", "encoder=", "exclude="])
except getopt.GetoptError:
    print('Usage: train_general_script.py [--bilstm=<Boolean> --encoder=<path_to_encoder> --exclude=<domain>] -n <model_name>\n'
          ' --bilstm, --encoder and --exclude are optional arguments.')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Usage: train_general_script.py [--bilstm=<Boolean> --encoder=<path_to_encoder> --exclude=<domain>] -n <model_name>\n'
              ' --bilstm, --encoder and --exclude are optional arguments.')
        sys.exit()
    elif opt == "-n":
        model_name = arg
    elif opt == "--bilstm":
        use_Bilstm = (arg.lower() == 'true')
    elif opt == "--encoder":
        encoder_path = arg
    elif opt == "--exclude":
        excluded_domain = arg
# Load the encoder
if encoder_path is not None:
    encoder = tf.keras.models.load_model(encoder_path)

# Excluded
if excluded_domain is not None and excluded_domain not in [e.value for e in Domain]:
    print(f'--exclude parameter input: {excluded_domain} is an unknown domain. If you want to exclude a domain from training, '
          f'please choose from {[e.value for e in Domain]}')
    sys.exit(1)

# Making sure the necessary folders exist
if not os.path.exists('./prepared_datasets/prepared_datasets'):
    print('It seems the ./prepared_datasets folder is missing. '
          'Please, make sure to pull the latest changes and to unzip the prepared_datasets.zip file')
    sys.exit(1)
if not os.path.exists('./embeddings'):
    print('Creating ./embeddings folder...')
    os.mkdir('./embeddings')

if not os.path.exists('./embeddings/general_embeddings.bin'):
    if not os.path.exists('./prepared_datasets/prepared_datasets/ALL_corpus.txt'):
        print('It seems the ./prepared_datasets/prepared_datasets folder is incomplete. '
              'Please, make sure to pull the latest changes and to unzip the prepared_datasets.zip file')
        sys.exit(1)

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
