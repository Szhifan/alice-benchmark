from transformers import BertForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A fast brown fox leaps over a sleepy dog."
inputs = tok(text1, text2)
decoded = tok.convert_ids_to_tokens(inputs['input_ids'])
print(decoded)