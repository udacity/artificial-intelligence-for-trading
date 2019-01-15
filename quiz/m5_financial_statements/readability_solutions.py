# solutions for readability exercises

# tokenize and clean the text
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from collections import Counter
from nltk.corpus import stopwords

from nltk import word_tokenize
from syllable_count import syllable_count

nltk.download('wordnet')
nltk.download('punkt')

sno = SnowballStemmer('english')
wnl = WordNetLemmatizer()

from nltk.tokenize import RegexpTokenizer

word_tokenizer = RegexpTokenizer(r'[^\d\W]+')

def word_tokenize(sent):
    return [ w for w in word_tokenizer.tokenize(sent) if w.isalpha() ]

from nltk.tokenize import sent_tokenize


def sentence_count(text):
    return len(sent_tokenizer.tokenize(text))

def word_count(sent):
    return len([ w for w in word_tokenize(sent)])

def hard_word_count(sent):
    return len([ w for w in word_tokenize(sent) \
                if syllable_count(wnl.lemmatize(w, pos='v'))>=3 ])

def flesch_index(text):
    sentences = sent_tokenize(text)

    total_sentences = len(sentences)
    total_words = np.sum([ word_count(s) for s in sentences ])
    total_syllables = np.sum([ np.sum([ syllable_count(w) for w in word_tokenize(s) ]) \
                              for s in sentences ])
    
    return 0.39*(total_words/total_sentences) + \
            11.8*(total_syllables/total_words) - 15.59

def fog_index(text):
    sentences = sent_tokenize(text)

    total_sentences = len(sentences)
    total_words = np.sum([ word_count(s) for s in sentences ])
    total_hard_words = np.sum([ hard_word_count(s) for s in sentences ])
    
    return 0.4*((total_words/total_sentences) + \
            100.0*(total_hard_words/total_words))

# solutions for bag-of-words, sentiments, similarity exercises