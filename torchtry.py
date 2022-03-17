import torch

#tens = torch.rand(11,45,100)
tens = torch.Tensor([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
print(tens)
print(tens.shape)
tensr = tens.reshape(tens.shape[1], tens.shape[0]* tens.shape[2])
print(tensr.shape)
print(tensr)
print(tens.view(tens.shape[1], tens.shape[0]* tens.shape[2]))
tensi = torch.Tensor([[1,2,3],[4,5,6]])
print(tensi.shape)
print(tensi[:,1])