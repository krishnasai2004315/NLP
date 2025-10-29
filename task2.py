
Open In Colab

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
text = "Tokenization without transformers is straight forward with tools like NLTK."
tokens  = word_tokenize(text)
print("token:",tokens)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Tokenization without transformers is straight forward with tools like NLTK."
tokens_transformers = tokenizer(text,return_tensors="pt")
print("transformers tokens:",tokens_transformers)
tokens_transformers_list = tokenizer.convert_ids_to_tokens(tokens_transformers["input_ids"][0].numpy().tolist())
print("transformers Tokens[list]:",tokens_transformers_list)
decoded_text = tokenizer.decode(tokens_transformers["input_ids"][0], skip_special_tokens = True)
print("Decoded Text:",decoded_text)
     
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
token: ['Tokenization', 'without', 'transformers', 'is', 'straight', 'forward', 'with', 'tools', 'like', 'NLTK', '.']
transformers tokens: {'input_ids': tensor([[  101, 19204,  3989,  2302, 19081,  2003,  3442,  2830,  2007,  5906,
          2066, 17953,  2102,  2243,  1012,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
transformers Tokens[list]: ['[CLS]', 'token', '##ization', 'without', 'transformers', 'is', 'straight', 'forward', 'with', 'tools', 'like', 'nl', '##t', '##k', '.', '[SEP]']
Decoded Text: tokenization without transformers is straight forward with tools like nltk.
