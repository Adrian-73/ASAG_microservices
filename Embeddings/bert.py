from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')
model = BertModel.from_pretrained('./bert_model')
def bert(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        word_array = []
        for i in range(embeddings.shape[1]):
            word_array.append(embeddings[0, i].numpy())
        return word_array