import os
from enum import Enum, unique
from tensorflow.keras import utils
import tensorflow as tf

@unique
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

# global constant for all domains
ALL_DOMAINS = tuple(Domain)

def load_domain(dataset_path, n_val=0.125, batch_size=32, seed=42):
    """
        Loads the train, validation and test-sets from disk.
        :param dataset_path: The folder containing the train/ and test/ subfolders.
        :param n_val: Ratio of validation set
        :param batch_size: Number of observations per batch
        :param seed: A seed for the random training-validation split
        :return: training- validation and test-sets as Tensorflow Datasets
        """
    train_dir = os.path.join(dataset_path, 'train')

    raw_train_ds = utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=n_val,
        subset='training',
        seed=seed)

    # Create a validation set.
    raw_val_ds = utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=n_val,
        subset='validation',
        seed=seed)

    test_dir = os.path.join(dataset_path, 'test')

    # Create a test set.
    raw_test_ds = utils.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size)

    return raw_train_ds, raw_val_ds, raw_test_ds

def load_data(path, domain=ALL_DOMAINS, n_val=0.125, batch_size=32, seed=42, exclude_domain=None):
    """
        Generates the training, validation and test datasets
        :param exclude_domain: Domain to be excluded from training
        :param path: The folder containing all datasets
        :param domain: The domain to be loaded
        :param n_val: Ratio of validation set
        :param batch_size: Number of observations per batch
        :param seed: A seed for the random training-validation split
        :return: training- validation and test-sets as Tensorflow Datasets
        """

    if domain == ALL_DOMAINS:
        first_domain = True
        for d in ALL_DOMAINS:
            d_str = d.value
            if d_str == exclude_domain:
                continue
            print(f'loading {d_str} ...')
            dataset_path = os.path.join(path, d_str)
            train_ds, val_ds, test_ds = load_domain(dataset_path, n_val, batch_size, seed)
            if first_domain:
                raw_train_ds = train_ds
                raw_val_ds = val_ds
                raw_test_ds = test_ds
                first_domain = False
            else:
                raw_train_ds = raw_train_ds.concatenate(train_ds)
                raw_val_ds = raw_val_ds.concatenate(val_ds)
                raw_test_ds = raw_test_ds.concatenate(test_ds)
        return raw_train_ds, raw_val_ds, raw_test_ds
    elif domain in Domain:
        d_str = domain.value
        print(f'loading {d_str} ...')
        dataset_path = os.path.join(path, d_str)
        return load_domain(dataset_path, n_val, batch_size, seed)
    else:
        raise NotImplementedError(f'The domain {domain} is unknown.')
