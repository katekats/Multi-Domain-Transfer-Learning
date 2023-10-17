import math
import pickle
import numpy as np
import tensorflow as tf
import data_loader
from attention import AttentionNetwork
from data_loader import Domain
from preprocessor import Preprocessor
import sys
import getopt
import os
import graphviz 
import pydot 
from keras.utils.vis_utils import plot_model


max_len = 20
max_sentences = 30
batch_size = 32

model_path = None
output_file = None
tl_approach = "s"
if "-i" not in sys.argv or "-o" not in sys.argv or "-a" not in sys.argv:
    print('Usage: transfer_learning.py -i <model_path> -o <output_file_name> -a <approach>\n'
          'Approach can be either S, W or A\n'
          'S for sentence-level fine-tuning\n'
          'W for word-level fine-tuning\n'
          'A for all-layers fine-tuning\n')
    exit(2)

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:a:o:")
except getopt.GetoptError:
    print('Usage: transfer_learning.py -i <model_path> -o <output_file_name> -a <approach>\n'
          'Approach can be either S, W or A\n'
          'S for sentence-level fine-tuning\n'
          'W for word-level fine-tuning\n'
          'A for all-layers fine-tuning\n')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Usage: transfer_learning.py -i <model_path> -o <output_file_name> -a <approach>\n'
              'Approach can be either S, W or A\n'
              'S for sentence-level fine-tuning\n'
              'W for word-level fine-tuning\n'
              'A for all-layers fine-tuning\n')
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
results = np.zeros((16, 5, 10, 2), dtype='float32')
step = 0


for d in Domain:
    print(d.value)
    domain_train_ds, domain_val_ds, domain_test_ds = data_loader.load_data(
        './prepared_datasets/prepared_datasets', domain=d, batch_size=32)

    domain_preprocessed_train_ds = domain_train_ds.map(lambda x, y: (
        tf.ensure_shape(
            tf.py_function(preprocessor.preprocess_ds_fasttext, [x], tf.float32),
            [x.shape[0], max_sentences, max_len, fasttext_dimension]), y))
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
    #for ratio in [0.05, 0.1, 0.15, 0.2, 1]:
    for ratio in [1]:
        print(ratio)
        for j in range(10):
            # Load the model
            model = tf.keras.models.load_model(model_path,custom_objects={"AttentionNetwork": AttentionNetwork})

            # Freeze layers
            if tl_approach == 's':
                model.layers[0].trainable = False
                model.layers[1].trainable = False
            elif tl_approach == 'w':
                for layer in model.layers:
                    layer.trainable = False
                model.layers[0].trainable = True
          #  elif tl_approach == 'a':
              #  model.layers[1].trainable = False
                
              #  model.layers[0].layers[0].trainable=False    
               # model.layers[1].trainable = False
               # model.layers[4].trainable = True
            model.layers
            print(len(model.layers))
            plot_model(
            model, to_file='model.png', show_shapes=False, show_dtype=False,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
            model.summary()
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          optimizer=tf.keras.optimizers.Adam(1e-5),
                          metrics=['accuracy'])

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
        i += 1
    step += 1

# Print domain-average results to console
f = 0
for d in Domain:
    print(f'{d.value}')
    #for i, ratio in enumerate([0.05, 0.1, 0.15, 0.2, 1]):
    for i, ratio in enumerate([1]):
        mean = np.mean(results[f, i], axis=0)
        print(f'Ratio: {ratio}\t{str(mean[0]).replace(".", ",")}\t{str(mean[1]).replace(".", ",")}')
    f += 1

# Save results to disk
if not os.path.exists('./results'):
    print('Creating ./results folder...')
    os.mkdir('./results')
pickle.dump(results, open(output_file, "wb"))
print(f'The model was saved to ./results/{output_file}')


