
Open In Colab

import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from nltk.classify import MaxentClassifier
nltk.download('maxent_treebank_pos_tagger')
from nltk.tag import PerceptronTagger, StanfordTagger
nltk.download('treebank')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
corpus = list(treebank.tagged_sents())
train_data = corpus[:int(0.8 * len(corpus))]
test_data = corpus[int(0.8 * len(corpus)):]
hmm_tagger = hmm.HiddenMarkovModelTrainer().train(train_data)
hmm_accuracy = hmm_tagger.evaluate(test_data)
print(f"HMM Tagger Accuracy: {hmm_accuracy:.4f}")
     
[nltk_data] Downloading package maxent_treebank_pos_tagger to
[nltk_data]     /root/nltk_data...
[nltk_data]   Package maxent_treebank_pos_tagger is already up-to-
[nltk_data]       date!
[nltk_data] Downloading package treebank to /root/nltk_data...
[nltk_data]   Package treebank is already up-to-date!
[nltk_data] Downloading package maxent_ne_chunker to
[nltk_data]     /root/nltk_data...
[nltk_data]   Package maxent_ne_chunker is already up-to-date!
[nltk_data] Downloading package words to /root/nltk_data...
[nltk_data]   Package words is already up-to-date!
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /root/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger is already up-to-
[nltk_data]       date!
/tmp/ipython-input-3128377115.py:15: DeprecationWarning: 
  Function evaluate() has been deprecated.  Use accuracy(gold)
  instead.
  hmm_accuracy = hmm_tagger.evaluate(test_data)
HMM Tagger Accuracy: 0.3647

# First, install nltk in your shell/terminal (NOT inside Python script):
# pip install -U nltk

import nltk
from nltk.corpus import treebank
from nltk.tag import hmm

# Download necessary NLTK data once (can run separately)
nltk.download('treebank')
nltk.download('universal_tagset')  # optional, if you want universal tags

# Load the tagged sentences from treebank corpus
corpus = list(treebank.tagged_sents())

# Split corpus into 80% train and 20% test data
train_data = corpus[:int(0.8 * len(corpus))]
test_data = corpus[int(0.8 * len(corpus)):]

# Train Hidden Markov Model (HMM) tagger on the training data
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Evaluate tagger on the test data
hmm_accuracy = hmm_tagger.evaluate(test_data)

print(f"HMM Tagger Accuracy: {hmm_accuracy:.4f}")

     
[nltk_data] Downloading package treebank to /root/nltk_data...
[nltk_data]   Package treebank is already up-to-date!
[nltk_data] Downloading package universal_tagset to /root/nltk_data...
[nltk_data]   Unzipping taggers/universal_tagset.zip.
/tmp/ipython-input-3323915329.py:24: DeprecationWarning: 
  Function evaluate() has been deprecated.  Use accuracy(gold)
  instead.
  hmm_accuracy = hmm_tagger.evaluate(test_data)
HMM Tagger Accuracy: 0.3647

import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from nltk.classify import MaxentClassifier
from nltk.tag.sequential import ClassifierBasedPOSTagger
nltk.download('treebank')
nltk.download('universal_tagset')
corpus = list(treebank.tagged_sents())
train_data = corpus[:int(0.8 * len(corpus))]
test_data = corpus[int(0.8 * len(corpus)):]
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)
hmm_accuracy = hmm_tagger.evaluate(test_data)
class MaxEntPOSTagger(ClassifierBasedPOSTagger):
    def __init__(self, train_data):
        self._classifier = MaxentClassifier
        super().__init__(train=train_data)

maxent_tagger = MaxEntPOSTagger(train_data)
maxent_accuracy = maxent_tagger.evaluate(test_data)
sentence = "The quick brown fox jumps over the lazy dog".split()
hmm_prediction = hmm_tagger.tag(sentence)
maxent_prediction = maxent_tagger.tag(sentence)
print(f"HMM Tagger Accuracy: {hmm_accuracy:.4f}")
print(f"Maximum Entropy (Log-Linear) Tagger Accuracy: {maxent_accuracy:.4f}")
print(f"HMM Prediction: {hmm_prediction}")
print(f"MaxEnt Prediction: {maxent_prediction}")

     
[nltk_data] Downloading package treebank to /root/nltk_data...
[nltk_data]   Package treebank is already up-to-date!
[nltk_data] Downloading package universal_tagset to /root/nltk_data...
[nltk_data]   Package universal_tagset is already up-to-date!
/tmp/ipython-input-195888549.py:13: DeprecationWarning: 
  Function evaluate() has been deprecated.  Use accuracy(gold)
  instead.
  hmm_accuracy = hmm_tagger.evaluate(test_data)
/tmp/ipython-input-195888549.py:20: DeprecationWarning: 
  Function evaluate() has been deprecated.  Use accuracy(gold)
  instead.
  maxent_accuracy = maxent_tagger.evaluate(test_data)
HMM Tagger Accuracy: 0.3647
Maximum Entropy (Log-Linear) Tagger Accuracy: 0.9320
HMM Prediction: [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NNP'), ('fox', 'NNP'), ('jumps', 'NNP'), ('over', 'NNP'), ('the', 'NNP'), ('lazy', 'NNP'), ('dog', 'NNP')]
MaxEnt Prediction: [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'WDT'), ('jumps', 'NNS'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'VBG')]
