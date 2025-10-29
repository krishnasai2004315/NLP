
Open In Colab

import nltk
from nltk.corpus import brown
from nltk.tag import HiddenMarkovModelTagger, PerceptronTagger
from sklearn.metrics import accuracy_score
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) # Added this line
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('brown', quiet=True)
perceptron_tagger = PerceptronTagger()
train_data = brown.tagged_sents(categories='news')[:1000]
hmm_tagger = HiddenMarkovModelTagger.train(train_data)
def perceptron_pos_tagger(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = perceptron_tagger.tag(tokens)
    return tagged
def hmm_pos_tagger(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = hmm_tagger.tag(tokens)
    return tagged
def compare_performance(sentence):
    hmm_result = hmm_pos_tagger(sentence)
    perceptron_result = perceptron_pos_tagger(sentence)
    gold_standard_tags = [tag for _, tag in perceptron_result]
    hmm_predicted_tags = [tag for _, tag in hmm_result]
    perceptron_predicted_tags = [tag for _, tag in perceptron_result]
    hmm_accuracy = accuracy_score(gold_standard_tags, hmm_predicted_tags)
    perceptron_accuracy = accuracy_score(gold_standard_tags, perceptron_predicted_tags)
    print("--- Performance Comparison ---")
    print(f"Input Sentence: '{sentence}'\n")
    print("Pseudo Gold Standard (Perceptron Tags):", gold_standard_tags)
    print("HMM Predicted Tags:                   ", hmm_predicted_tags)
    print("\nHMM Accuracy (vs Perceptron):       ", round(hmm_accuracy, 4))
    print("Perceptron Accuracy (vs itself):    ", round(perceptron_accuracy, 4))
    print("\n--- Detailed Tagging Results ---")
    print("Word\t\tHMM Tag\t\tPerceptron Tag")
    print("-" * 40)
    for (word, p_tag), (_, h_tag) in zip(perceptron_result, hmm_result):
        marker = '!' if p_tag != h_tag else ''
        print(f"{word:<15}{h_tag:<15}{p_tag:<15}{marker}")
input_text = "The quick brown fox jumps over the lazy dog."
compare_performance(input_text)
     
--- Performance Comparison ---
Input Sentence: 'The quick brown fox jumps over the lazy dog.'

Pseudo Gold Standard (Perceptron Tags): ['DT', 'JJ', 'NN', 'NN', 'VBZ', 'IN', 'DT', 'JJ', 'NN', '.']
HMM Predicted Tags:                    ['AT', 'JJ', 'TO', 'BE', 'VBN', 'IN', 'AT', 'JJ', 'NN', '.']

HMM Accuracy (vs Perceptron):        0.5
Perceptron Accuracy (vs itself):     1.0

--- Detailed Tagging Results ---
Word		HMM Tag		Perceptron Tag
----------------------------------------
The            AT             DT             !
quick          JJ             JJ             
brown          TO             NN             !
fox            BE             NN             !
jumps          VBN            VBZ            !
over           IN             IN             
the            AT             DT             !
lazy           JJ             JJ             
dog            NN             NN             
.              .              .              
