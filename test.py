import torch 
a = torch.randn(1, 100, 768)
b = torch.randn(1, 100, 768)
window = 10
c = torch.zeros((1, 100, 100))
for i in range(100):
    start = max(0, i - window)
    end = min(100, i + window)
    c[:, i, start:end] = torch.bmm(a[:, i:i+1, :], b[:, start:end, :].transpose(1, 2)).squeeze(1)

for i in c[0]:
    print(i)
    