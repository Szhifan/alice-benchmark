# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B") 

input = "The capital of France is"
output = pipe(input, max_length=50, do_sample=True, temperature=0.7)