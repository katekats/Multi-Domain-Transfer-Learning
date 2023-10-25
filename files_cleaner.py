import os
import fasttext
from sklearn.model_selection import train_test_split


def generate_corpus_file(path, output_path):
    """
     Merge all reviews in a single "corpus file"
    :param path: Path to folder containing all domains and all reviews
    :param output_path: Destination path for the corpus file
    """
    with open(output_path, 'w', encoding="latin-1") as outfile:
        for root, dirs, files in os.walk(path):
            for file in files:
                with open(os.path.join(root, file), "r", encoding="latin-1") as infile:
                    outfile.write(infile.read())


def produce_embeddings(path, output_path):
    """
    Train a fasttext embedding model on given text, and save it.
    :param path: Location of a text file, from which the embeddings should be learned
    :param output_path: Destination path, where the model should be saved
    :return:
    """
    embedding = fasttext.train_unsupervised(path)
    embedding.save_model(output_path)


def get_reviews_from_file(filename):
    """
    Extract reviews from a ".review" file from the Multidomain-sentiment-analysis dataset.
    :param filename: path to file
    :return: An array containing all reviews in the file
    """
    content = []
    read_next = False
    try:
        with open(filename, "r+", encoding="latin-1") as file:
            review = ''
            for line in file:
                if line.rstrip() == '</review_text>':
                    read_next = False
                    content.append(review)
                    review = ''
                if read_next:
                    review += line
                if line.rstrip() == '<review_text>':
                    read_next = True
    except IOError:
        print("[ERR] File does not exist")

    return content


def write_to_file(reviews, path):
    """
    Write each review in a seperate file
    :param reviews: A string array containing reviews
    :param path: Destination path, where the reviews should be written
    """
    for i in range(len(reviews)):
        with open(os.path.join(path, str(i) + '.txt'), 'w', encoding="latin-1") as text_file:
            text_file.write(reviews[i])


def generate_train_test_files(reviews, path, label):
    """
    Split reviews in train and test sets and write them in the correct subfolders
    :param reviews: A string array containing reviews
    :param path: Destination path for reviews
    :param label: True label of the given reviews (normally "pos", "neg" or "unlabeled")
    :return:
    """
    train_path = os.path.join(path, 'train/' + label)
    test_path = os.path.join(path, 'test/' + label)
    os.makedirs(train_path)
    os.makedirs(test_path)
    reviews_train, reviews_test = train_test_split(reviews, test_size=0.2, random_state=42)
    write_to_file(reviews_train, train_path)
    write_to_file(reviews_test, test_path)

def process_multi_domain_sentiment_dataset(root_path):
    for root,  dirs, files in os.walk(root_path):
        for f in files:
            if f == 'positive.review':
                print(f'processing: {root}/{f}')
                reviews = get_reviews_from_file(os.path.join(root, f))
                generate_train_test_files(reviews, root, 'pos')
            if f == 'negative.review':
                print(f'processing: {root}/{f}')
                reviews = get_reviews_from_file(os.path.join(root, f))
                generate_train_test_files(reviews, root, 'neg')
            if f == 'unlabeled.review':
                print(f'processing: {root}/{f}')
                reviews = get_reviews_from_file(os.path.join(root, f))
                generate_train_test_files(reviews, root, 'unlabeled')


def move_files(files, source_folder, destination_folder):
    """
    :param files: files to be relocated
    :param source_folder: Source location
    :param destination_folder: Destination location
    """
    for file in files:
        os.replace(os.path.join(source_folder, file), os.path.join(destination_folder, file))


def split_folder(path, label):
    """
    Split content of a folder into train and test folders
    :param path: Location of the folder to split
    :param label: True label of reviews in the folder to be split
    """
    subfolder = os.path.join(path, label)
    train_path = os.path.join(path, 'train/' + label)
    test_path = os.path.join(path, 'test/' + label)
    os.makedirs(train_path)
    os.makedirs(test_path)
    files = next(os.walk(subfolder))[2]
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    move_files(train_files, subfolder, train_path)
    move_files(test_files, subfolder, test_path)


def process_mr_dataset(root_path):
    split_folder(root_path, 'neg')
    split_folder(root_path, 'pos')

