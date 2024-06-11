from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import torch
from .bert import bert
from .sentence import SimilarityFunctions, SentenceEmbeddings
from .preprocess import pre_processing
from .evaluate import evaluate
tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')
model = BertModel.from_pretrained('./bert_model')
print("All modules imported successfully")