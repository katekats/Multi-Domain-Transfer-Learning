import re
import fasttext
import numpy as np
import tensorflow as tf
from nltk import tokenize
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize
from keras.preprocessing.text import text_to_word_sequence

# create the stopwords list
stopwords_list = nltk_stopwords.words('english')
stopwords_keep = ['no', 'not', 'nor']
stopwords = list(set(stopwords_list).difference(set(stopwords_keep)))

class Preprocessor(object):
    def __init__(self, max_sentences=50, max_len=30, fasttext_path=None):
        self.max_sentences = max_sentences
        self.max_len = max_len
        if fasttext_path is not None:
            self.fasttext_model = fasttext.load_model(fasttext_path)

    def clean_str(self, string):
        string = re.sub(r"[\\\'\"]", "", string)
        words = [word for word in string.split() if word not in stopwords]
        return " ".join(words).strip().lower()

    def set_fasttext_model(self, fasttext_path):
        self.fasttext_model = fasttext.load_model(fasttext_path)

    def preprocess_ds_fasttext(self, r):
        data = np.zeros((r.shape[0], self.max_sentences, self.max_len, self.fasttext_model.get_dimension()), dtype='float32')
        
        for i, review in enumerate(r):
            text = self.clean_str(review.numpy().decode("latin-1"))                    
            sentences = sent_tokenize(text)[:self.max_sentences]
            for j, sent in enumerate(sentences):
                if j < self.max_sentences:
                    word_tokens = text_to_word_sequence(sent)[:self.max_len]
                    for k, word in enumerate(word_tokens):
                        if word in self.fasttext_model:
                            data[i, j, k] = self.fasttext_model[word]
        
        return tf.constant(data,
                           shape=(r.shape[0], self.max_sentences, self.max_len, self.fasttext_model.get_dimension()),
                           dtype=tf.float32)

