
Open In Colab

from nltk.util import ngrams
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline

def ngram_smoothing(sentence, n):
    tokens = word_tokenize(sentence.lower())
    # Prepare training data: list of tokenized sentences
    train_data, padded_sents = padded_everygram_pipeline(n, [tokens])

    model = Laplace(n)
    model.fit(train_data, padded_sents)
    return model, tokens

sentence = input("Enter a sentence: ")
n = int(input("Enter the value of N for N-grams: "))

model, tokens = ngram_smoothing(sentence, n)

# Prepare context: last n-1 tokens
if n > 1:
    context = tuple(tokens[-(n-1):])
else:
    context = ()

# Generate next 3 words
next_words = model.generate(3, text_seed=context)

print("Next words:", ' '.join(next_words))

     
Enter a sentence: this is a sample code
Enter the value of N for N-grams: 5
Next words: </s> </s> </s>
