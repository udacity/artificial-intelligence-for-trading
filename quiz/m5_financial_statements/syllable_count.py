import nltk
nltk.download('cmudict')


from nltk.corpus import cmudict
import numpy as np

d = cmudict.dict()

def syllable_count(word):
    try:
        return np.min([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]])
    except KeyError:
        #if word not found in cmudict
        return _syllables(word)

def _syllables(word):
#referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count