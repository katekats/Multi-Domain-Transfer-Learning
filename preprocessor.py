import re
import fasttext
import numpy as np
import tensorflow as tf
from nltk import tokenize
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import re

# create the stopwords list
stopwords = stopwords.words('english')
stopwords_keep =['no', 'not', 'nor']
stopwords = list(set(stopwords).difference(set(stopwords_keep)))


class Preprocessor(object):
    def __init__(self, max_sentences=50, max_len=30, fasttext_path=None):
        self.max_sentences = max_sentences
        self.max_len = max_len
        if fasttext_path is not None:
            self.fasttext_model = fasttext.load_model(fasttext_path)

    # Tokenization/string cleaning for dataset
    def clean_str(self, string):
        """
            String/Token cleaning.
            
            """
       
        # remove stopwords
        
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        string = re.sub(r"\"", "", string)
        string = [word for word in x.split() if word not in stopwords]
        return string.strip().lower()

    def set_fasttext_model(self, fasttext_path):
        self.fasttext_model = fasttext.load_model(fasttext_path)

    def preprocess_ds_fasttext(self, r):
        """
            Reshapes the input batch from (batch_size, ) to (batch_size, max_sentences, max_len, fasttext_dimension)
            and replaces each word of each review by its fasttext embedding.
            """
        data = np.zeros((r.shape[0], self.max_sentences, self.max_len, self.fasttext_model.get_dimension()),
                        dtype='float32')
        for i, review in enumerate(r):
            text = self.clean_str(review.numpy().decode("latin-1"))                    
            sentences = tokenize.sent_tokenize(text)[:self.max_sentences]
            for j, sent in enumerate(sentences):
                if j < self.max_sentences:
                    word_tokens = text_to_word_sequence(sent)[:self.max_len]
                    for k, word in enumerate(word_tokens):
                        data[i, j, k] = self.fasttext_model[word]
        return tf.constant(data,
                           shape=(r.shape[0], self.max_sentences, self.max_len, self.fasttext_model.get_dimension()),
                           dtype=tf.float32)
