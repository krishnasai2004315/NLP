
Open In Colab

import string
import random
import nltk
from nltk.corpus import stopwords, reuters
from collections import Counter, defaultdict
from nltk import FreqDist, ngrams
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('reuters')
sents = reuters.sents()

stop_word = set(stopwords.words('english'))
removal_list = list(stop_word) + list(string.punctuation) + ['\t', 'rt']

unigram = []
bigram = []
trigram = []
for sentence in sents:
    sentence = [word.lower() for word in sentence if word != '.']
    unigram.extend(sentence)
    bigram.extend(list(ngrams(sentence, 2, pad_left=True, pad_right=True)))
    trigram.extend(list(ngrams(sentence, 3, pad_left=True, pad_right=True)))

def remove_stopwords(items):
    filtered = []
    for item in items:
        if isinstance(item, tuple):
            if all(word not in removal_list and word is not None for word in item):
                filtered.append(item)
        else:
            if item not in removal_list:
                filtered.append(item)
    return filtered
unigram = remove_stopwords(unigram)
bigram = remove_stopwords(bigram)
trigram = remove_stopwords(trigram)
freq_uni = FreqDist(unigram)
freq_bi = FreqDist(bigram)
freq_tri = FreqDist(trigram)
d = defaultdict(Counter)
for (a, b, c), freq in freq_tri.items():
    if a is not None and b is not None and c is not None:
        d[(a, b)][c] += freq
def pick_word(counter, fallback_list):
    "Choose a random element weighted by frequency, with a fallback."
    if counter:
        return random.choice(list(counter.elements()))
    elif fallback_list:
        return random.choice(fallback_list)
    else:
        return "the"
prefix = ("he", "is")
s = list(prefix)
print(" ".join(s))
for _ in range(19):
    suffix = pick_word(d[prefix], unigram)
    s.append(suffix)
    print(" ".join(s))
    prefix = (prefix[1], suffix)
     
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package reuters to /root/nltk_data...
[nltk_data]   Package reuters is already up-to-date!
he is
he is six
he is six five
he is six five large
he is six five large prospective
he is six five large prospective new
he is six five large prospective new 18
he is six five large prospective new 18 200
he is six five large prospective new 18 200 railways
he is six five large prospective new 18 200 railways aimed
he is six five large prospective new 18 200 railways aimed asking
he is six five large prospective new 18 200 railways aimed asking class
he is six five large prospective new 18 200 railways aimed asking class money
he is six five large prospective new 18 200 railways aimed asking class money branch
he is six five large prospective new 18 200 railways aimed asking class money branch corp
he is six five large prospective new 18 200 railways aimed asking class money branch corp rates
he is six five large prospective new 18 200 railways aimed asking class money branch corp rates company
he is six five large prospective new 18 200 railways aimed asking class money branch corp rates company index
he is six five large prospective new 18 200 railways aimed asking class money branch corp rates company index mln
he is six five large prospective new 18 200 railways aimed asking class money branch corp rates company index mln contracted
