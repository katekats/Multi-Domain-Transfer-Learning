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
DATA_PATH = './prepared_datasets/prepared_datasets'
EMBEDDINGS_PATH = './embeddings/general_embeddings.bin'
MODEL_DIR = './models'


def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Train General Script')
    parser.add_argument('-n', required=True, type=str, help='Model name')
    parser.add_argument('--bilstm', type=lambda x: (x.lower() == 'true'), default=False, help='Use BiLSTM')
    parser.add_argument('--exclude', type=str, help='Domain to exclude')
    return parser.parse_args()


def check_directories():
    directories = {
        DATA_PATH: 'It seems the ./prepared_datasets folder is missing...',
        './embeddings': 'Creating ./embeddings folder...'
    }

    for dir_path, message in directories.items():
        if not os.path.exists(dir_path):
            if dir_path == './embeddings':
                os.mkdir(dir_path)
                print(message)
            else:
                print(message)
                sys.exit(1)


def generate_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH) and not os.path.exists(f'{DATA_PATH}/ALL_corpus.txt'):
        print('Producing general embeddings using FastText...')
        general_embeddings = fasttext.train_unsupervised(f'{DATA_PATH}/ALL_corpus.txt', dim=100, minCount=1)
        general_embeddings.save_model(EMBEDDINGS_PATH)


def preprocess_dataset(dataset, preprocessor):
    fasttext_dimension = preprocessor.fasttext_model.get_dimension()

    return dataset.map(lambda x, y: (
        tf.ensure_shape(
            tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
            [x.shape[0], MAX_SENTENCES, MAX_LEN, fasttext_dimension]), y))


def main():
    args = setup_argument_parser()

    model_name = args.n
    use_Bilstm = args.bilstm
    excluded_domain = args.exclude

    # Check exclusion domain
    if excluded_domain and excluded_domain not in [e.value for e in Domain]:
        print(f'--exclude parameter input: {excluded_domain} is an unknown domain...')
        sys.exit(1)

    check_directories()
    generate_embeddings()

    # Load data
    train_ds, val_ds, test_ds = data_loader.load_data(DATA_PATH, batch_size=BATCH_SIZE, exclude_domain=excluded_domain)

    # Preprocessor setup
    preprocessor = Preprocessor(MAX_SENTENCES, MAX_LEN)
    preprocessor.set_fasttext_model(EMBEDDINGS_PATH)

    # Preprocess datasets
    preprocessed_train_ds = preprocess_dataset(train_ds, preprocessor)
    preprocessed_val_ds = preprocess_dataset(val_ds, preprocessor)
    preprocessed_test_ds = preprocess_dataset(test_ds, preprocessor)

    # Training
    model, m_history = training.train(preprocessed_train_ds, preprocessed_val_ds, epochs=50, bilstm=use_Bilstm)

    # Evaluating
    print("Performance on testing data")
    model.evaluate(preprocessed_test_ds)

    for d in Domain:
        print(f'Performance on {d.value} domain')
        domain_train_ds, _, domain_test_ds = data_loader.load_data(DATA_PATH, domain=d, batch_size=32)
        domain_preprocessed_test_ds = preprocess_dataset(domain_test_ds, preprocessor)
        model.evaluate(domain_preprocessed_test_ds)

    # Save the model to disk
    if not os.path.exists(MODEL_DIR):
        print('Creating ./models folder...')
        os.mkdir(MODEL_DIR)

    model_path = f'{MODEL_DIR}/{model_name}.h5'
    model.save(model_path, save_format='h5')
    print(f'The model was saved to {model_path}')


if __name__ == "__main__":
    main()
