import os
from enum import Enum, unique
from tensorflow.keras import utils

@unique
class Domain(Enum):
    

def load_domain(dataset_path, domain, n_val=0.125, batch_size=32, seed=42):
    """Loads the train, validation and test-sets for a specific domain."""
    domain_path = os.path.join(dataset_path, domain.value)
    
    # Load training set
    train_path = os.path.join(domain_path, 'train')
    raw_train_ds = utils.text_dataset_from_directory(
        train_path,
        batch_size=batch_size,
        validation_split=n_val,
        subset='training',
        seed=seed
    )

    # Load validation set
    raw_val_ds = utils.text_dataset_from_directory(
        train_path,
        batch_size=batch_size,
        validation_split=n_val,
        subset='validation',
        seed=seed
    )

    # Load test set
    test_path = os.path.join(domain_path, 'test')
    raw_test_ds = utils.text_dataset_from_directory(
        test_path,
        batch_size=batch_size
    )

    return raw_train_ds, raw_val_ds, raw_test_ds

def load_data(path, domain=None, exclude_domain=None, n_val=0.125, batch_size=32, seed=42):
    """
    Generates the training, validation, and test datasets.
    """
    if domain is None:  # Load all domains
        train_ds_list, val_ds_list, test_ds_list = [], [], []
        for d in Domain:
            if d.value != exclude_domain:
                print(f'loading {d.value} ...')
                train_ds, val_ds, test_ds = load_domain(path, d, n_val, batch_size, seed)
                train_ds_list.append(train_ds)
                val_ds_list.append(val_ds)
                test_ds_list.append(test_ds)

        return (
            tf.data.experimental.concatenate(train_ds_list),
            tf.data.experimental.concatenate(val_ds_list),
            tf.data.experimental.concatenate(test_ds_list)
        )
    elif isinstance(domain, Domain):  # Load a specific domain
        return load_domain(path, domain, n_val, batch_size, seed)
    else:
        raise ValueError(f'The domain {domain} is unknown.')
