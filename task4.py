
Open In Colab

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def tokenize_document(document):
    tokens = word_tokenize(document)
    return [word.lower() for word in tokens if word.isalpha()]
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]
def find_morphology(tokens):
    fdist = FreqDist(tokens)
    return fdist.most_common()
document_path = '/content/sample_text.txt'
document_text = load_document(document_path)
tokens = tokenize_document(document_text)
tokens_without_stopwords = remove_stopwords(tokens)
morphology = find_morphology(tokens_without_stopwords)

print("Morphology of the document:")
for word, frequency in morphology:
    print(f"{word}: {frequency}")
     
Morphology of the document:
nlp: 7
language: 3
natural: 2
computers: 2
human: 2
machine: 2
computational: 2
learning: 2
data: 2
processing: 1
subfield: 1
artificial: 1
intelligence: 1
ai: 1
focuses: 1
interaction: 1
humans: 1
ultimate: 1
goal: 1
enable: 1
understand: 1
interpret: 1
generate: 1
way: 1
valuable: 1
meaningful: 1
applications: 1
include: 1
translation: 1
sentiment: 1
analysis: 1
speech: 1
recognition: 1
chatbots: 1
combines: 1
modeling: 1
statistical: 1
deep: 1
models: 1
recent: 1
advancements: 1
driven: 1
powerful: 1
resources: 1
despite: 1
significant: 1
progress: 1
still: 1
faces: 1
challenges: 1
ambiguity: 1
context: 1
understanding: 1
need: 1
vast: 1
amounts: 1
labeled: 1
researchers: 1
continue: 1
explore: 1
new: 1
methods: 1
improve: 1
accuracy: 1
efficiency: 1
systems: 1
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package punkt_tab to /root/nltk_data...
[nltk_data]   Package punkt_tab is already up-to-date!
