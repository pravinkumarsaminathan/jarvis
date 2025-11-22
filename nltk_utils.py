import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split a sentence into array of words/tokens
    e.g. ["how", "are", "you"]
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Find the root form of the word
    e.g. ["organizing", "organizes"] -> "organ"
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    Example:
        sentence = ["hello", "how", "are", "you"]
        all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag = [0, 1, 0, 1, 0, 0, 0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
