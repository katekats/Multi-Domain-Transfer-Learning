import math
import pickle
import numpy as np
import tensorflow as tf
import data_loader
from attention import AttentionNetwork
from data_loader import Domain
from preprocessor import Preprocessor
import sys
import os
import argparse
import graphviz 
import pydot 
from keras.utils.vis_utils import plot_model

# Constants
MAX_LEN = 20
MAX_SENTENCES = 30
BATCH_SIZE = 32

def validate_approach(approach):
    """Validate and return the approach type."""
    approach_mapping = {
        'w': 'Word-level fine-tuning approach adopted.',
        'a': 'All-layers fine-tuning approach adopted.',
        's': 'Sentence-level fine-tuning approach adopted.'
    }
    if approach not in approach_mapping:
        raise ValueError(f"Invalid approach: {approach}. Options are: S (sentence-level), W (word-level), or A (all-layers).")
    print(approach_mapping[approach])
    return approach

def check_directories(dir_map):
    """Check if the necessary directories exist."""
    for dir_path, error_message in dir_map.items():
        if not os.path.exists(dir_path):
            raise FileNotFoundError(error_message)



def preprocess_datasets(preprocessor, domain_train_ds, domain_val_ds, domain_test_ds):
    """Applies preprocessing to the input datasets."""
    datasets = [domain_train_ds, domain_val_ds, domain_test_ds]
    processed_datasets = []
    
    for ds in datasets:
        processed_ds = ds.map(lambda x, y: (
            tf.ensure_shape(
                tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
                [x.shape[0], max_sentences, max_len, fasttext_dimension]), y))
        processed_datasets.append(processed_ds)

    return processed_datasets

def load_and_compile_model(tl_approach, model_path):
    """Loads a model from the given path and configures it according to the approach."""
    # Load the model
    model = tf.keras.models.load_model(model_path, custom_objects={"AttentionNetwork": AttentionNetwork})

    # Modify the model based on the transfer learning approach
    if tl_approach == 's':
        model.layers[0].trainable = False
        model.layers[1].trainable = False
    elif tl_approach == 'w':
        for layer in model.layers:
            layer.trainable = False
        model.layers[0].trainable = True

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-5),
                  metrics=['accuracy'])

    return model
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transfer Learning Script")
    parser.add_argument('-i', required=True, dest="model_path", help="Path to input model")
    parser.add_argument('-o', required=True, dest="output_file", help="Name of the output file")
    parser.add_argument('-a', required=True, dest="approach", help="Fine-tuning approach. Options are: S (sentence-level), W (word-level), or A (all-layers)")

    args = parser.parse_args()

    # Assign values from arguments and validate inputs
    model_path = args.model_path
    output_file = args.output_file
    tl_approach = validate_approach(args.approach.lower())

    # Test if model exists at given path
    model = tf.keras.models.load_model(model_path, custom_objects={"AttentionNetwork": AttentionNetwork})

    # Directory checks
    necessary_directories = {
        './prepared_datasets/prepared_datasets': 'The ./prepared_datasets folder appears to be missing. Ensure you have the latest changes and have unzipped the prepared_datasets.zip file.',
        './embeddings/general_embeddings.bin': 'No embeddings found at \'./embeddings/general_embeddings.bin\'. Ensure fasttext embeddings exist in this location and that the input model was trained with these embeddings.'
    }
    check_directories(necessary_directories)

preprocessor = Preprocessor(MAX_SENTENCES, MAX_LEN)
    preprocessor.set_fasttext_model('./embeddings/general_embeddings.bin')
    fasttext_dimension = preprocessor.fasttext_model.get_dimension()

    # Callback setup
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    
    results = np.zeros((16, 5, 10, 2), dtype='float32')
    step = 0

    for d in Domain:
        print(d.value)
        domain_train_ds, domain_val_ds, domain_test_ds = data_loader.load_data(
            './prepared_datasets/prepared_datasets', domain=d, batch_size=BATCH_SIZE)
        domain_preprocessed_train_ds, domain_preprocessed_val_ds, domain_preprocessed_test_ds = preprocess_datasets(
            preprocessor, domain_train_ds, domain_val_ds, domain_test_ds)

        #for ratio in [0.05, 0.1, 0.15, 0.2, 1]:
        for ratio in [1]:
            print(ratio)
            for j in range(10):
                model = load_and_compile_model(tl_approach, model_path)

                val_steps = math.ceil(domain_preprocessed_val_ds.cardinality().numpy() * ratio)
                train_steps = math.ceil(domain_preprocessed_train_ds.cardinality().numpy() * ratio)
                used_domain_train_ds = domain_preprocessed_train_ds.take(train_steps).cache()
                used_domain_val_ds = domain_preprocessed_val_ds.take(val_steps).cache()

                # Train
                model.fit(used_domain_train_ds, epochs=10,
                          validation_data=used_domain_val_ds,
                          validation_steps=val_steps,
                          callbacks=[callback])
                # Evaluate
                results[step, i, j, 0], results[step, i, j, 1] = model.evaluate(domain_preprocessed_test_ds)
            step += 1

    # Display results
    for d_idx, d in enumerate(Domain):
        print(f'{d.value}')
        #for i, ratio in enumerate([0.05, 0.1, 0.15, 0.2, 1]):
        for i, ratio in enumerate([1]):
            mean = np.mean(results[d_idx, i], axis=0)
            print(f'Ratio: {ratio}\t{str(mean[0]).replace(".", ",")}\t{str(mean[1]).replace(".", ",")}')

    # Save results
    if not os.path.exists('./results'):
        os.mkdir('./results')
    pickle.dump(results, open(output_file, "wb"))
    print(f'The model was saved to ./results/{output_file}')


if __name__ == "__main__":
    main()

