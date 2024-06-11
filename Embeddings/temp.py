from transformers import BertModel,BertTokenizer

# Define the model
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Save the model locally
model.save_pretrained('./bert_model')
tokenizer.save_pretrained('./bert_tokenizer')