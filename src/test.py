from transformers import T5Tokenizer
from transformers import T5ForSequenceClassification, T5Model

t5model = T5ForSequenceClassification.from_pretrained("t5-small")
text = "The quick brown fox jumps over the lazy dog"
tokenizer = T5Tokenizer.from_pretrained("t5-small")
inputs = tokenizer(text, return_tensors="pt")
outputs = t5model(**inputs)
print(outputs)