import torch
from transformers import AutoTokenizer, AutoModel

# 1. Setup - Using a standard BERT model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Sample Data (Representing your Plutchik dataset)
sentences = [
    "I am so incredibly happy with this result!", # Joy
    "I can't believe you would betray my trust like that.", # Disgust/Anger
]

print(f"--- Processing {len(sentences)} sentences ---")

# 3. Tokenization Step
# This converts text to 'input_ids' (numbers) and 'attention_mask' (1s and 0s)
inputs = tokenizer(
    sentences, 
    padding=True, 
    truncation=True, 
    max_length=12, 
    return_tensors="pt" # Return PyTorch tensors
)

print("\n[Token IDs]:")
print(inputs["input_ids"])

# 4. Embedding Step (The 'Forward Pass')
# We pass the tokens into the BERT model to get the vectors
with torch.no_grad():
    outputs = model(**inputs)

# The 'last_hidden_state' contains the embeddings for every token
# Shape: [Batch_Size, Sequence_Length, Hidden_Dimension]
embeddings = outputs.last_hidden_state

print("\n[Embedding Shape]:")
print(embeddings.shape) 
# Result will be [2, 12, 768] -> 2 sentences, 12 tokens each, 768 dimensions per token

# 5. Sentence-Level Embedding (CLS Token)
# Usually, for Emotion Classification, we just take the first token ([CLS]) 
# as the summary of the whole sentence.
sentence_vectors = embeddings[:, 0, :]
print("\n[Sentence Vector for Sentence 1 (First 5 values)]:")
print(sentence_vectors[0][:5])
