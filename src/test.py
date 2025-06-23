import torch 
from models import AsagCrossEncoder
model_path = "results/cross-encoder/checkpoint/model.pt"
model = AsagCrossEncoder(model_name="bert-base-multilingual-uncased", use_ce_loss=False)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
input_ids = torch.tensor([[101, 102, 103], [201, 202, 203]])
attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
print(outputs)