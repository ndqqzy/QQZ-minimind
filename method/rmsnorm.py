import torch

#rsqrt：开方然后求倒数
t = torch.rsqrt(torch.tensor(4.0))
print(t)

#创建全1张量
t2 = torch.ones(3,4)
print(t2)
