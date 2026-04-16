# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
#from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
model = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-large")


text = "Transformers create contextual embeddings"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

word_embeddings = outputs.last_hidden_state
print(inputs)
print(word_embeddings)
print(word_embeddings.shape)