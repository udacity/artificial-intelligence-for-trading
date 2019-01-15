import numpy as np
from math import log
# for nice number printing
np.set_printoptions(precision=3, suppress=True)

# tokenize and clean the text
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
# tokenize anything that is not a number and not a symbol
word_tokenizer = RegexpTokenizer(r'[^\d\W]+')

nltk.download('stopwords')
nltk.download('wordnet')

sno = SnowballStemmer('english')
wnl = WordNetLemmatizer()

# get our list of stop_words
stop_words = set(stopwords.words('english')) 
# add some extra stopwords
stop_words |= {"may", "business", "company", "could", "service", "result", "product", 
               "operation", "include", "law", "tax", "change", "financial", "require",
               "cost", "market", "also", "user", "plan", "actual", "cash", "other",
               "thereto", "thereof", "therefore"}

# useful function to print a dictionary sorted by value (largest first by default)
def print_sorted(d, ascending=False):
    factor = 1 if ascending else -1
    sorted_list = sorted(d.items(), key=lambda v: factor*v[1])
    for i, v in sorted_list:
        print("{}: {:.3f}".format(i, v))

# convert text into bag-of-words
def clean_text(txt):
    lemm_txt = [ wnl.lemmatize(wnl.lemmatize(w.lower(),'n'),'v') \
                for w in word_tokenizer.tokenize(txt) if \
                w.isalpha() and w not in stop_words ]
    return [ sno.stem(w) for w in lemm_txt if w not in stop_words and len(w) > 2 ]


from collections import defaultdict

def bag_of_words(words):
    bag = defaultdict(int)
    for w in words:
        bag[w] += 1
    return bag

def get_sentiment(txt, wordlist):
    matching_words = [ w for w in txt if w in wordlist ]
    return len(matching_words)/len(txt)

# compute idf
def get_idf(corpus, include_log=True):
    N = len(corpus)
    freq = defaultdict(int)
    words = set()
    for c in corpus:
        words |= set(c)
        
    for w in words:
        freq[w] = sum([ w in c for c in corpus])

    if include_log:
        return { w:log(N/freq[w]) for w in freq }
    else:
        return { w:N/freq[w] for w in freq }


def _tf(freq, avg, include_log=True):
    if include_log:
        return 0 if freq == 0 else (1+log(freq))/(1+log(avg))
    else:
        return freq/avg

def get_tf(txt, include_log=True):
    freq = bag_of_words(txt)
    avg = np.mean(list(freq.values()))
    tf = {w:_tf(f,avg, include_log) for w,f in freq.items()}
    return defaultdict(int, tf)

def get_vector(tf, idf):
    return np.array([ tf[w]*idf[w] for w in idf ])


from numpy.linalg import norm

def sim_cos(u,v):
    return np.dot(u,v)/(norm(u)*norm(v))

def sim_jac(u,v):
    return np.sum(np.minimum(u,v))/np.sum(np.maximum(u,v))