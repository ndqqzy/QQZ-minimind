import torch
#记录repe所用的一些方法

# condition负责条件过滤，符合条件的张量留下，不符合的用y填补
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([10,20,30,40,50])
condition = x > 3
result = torch.where(condition,x,y)
print(result)#tensor([10, 20, 30,  4,  5])


# 生成指定步长的张量
t1 = torch.arange(0,10,2)#tensor([0, 2, 4, 6, 8]) 
t2 = torch.arange(5,0,-1)#tensor([5, 4, 3, 2, 1])
print(t1,t2)

#外梯，v2每行乘以v1的每个
v1 = torch.tensor([1, 2, 3])#tensor([[ 4,  5,  6],
v2 = torch.tensor([4, 5, 6])#        [ 8, 10, 12],
result = torch.outer(v1, v2)#        [12, 15, 18]])
print(result)

#cat，维度拼接
t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]])#shape=[2,2,3]，2个行2列3的矩阵
t2 = torch.tensor([[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]])
result = torch.cat((t1, t2), dim=0)#dim为0，从第一个维度开始拼接
print(result)
#dim=0，沿着第一个维度拼接，结果是一个[4,2,3]的张量,4个行2列3的矩阵
#  tensor([[[ 1,  2,  3],
#          [ 4,  5,  6]],

#         [[13, 14, 15],
#          [16, 17, 18]],

#         [[ 7,  8,  9],
#          [10, 11, 12]],

#         [[19, 20, 21],
#          [22, 23, 24]]])
result2 = torch.cat((t1, t2), dim=1)#dim为1，从第二个维度开始拼接
print(result2)
#dim=1，沿着第二个维度拼接，结果是一个[2,4,3]的张量,2个行4列3的矩阵
#  tensor([[[ 1,  2,  3],
#          [ 4,  5,  6],
#          [ 7,  8,  9],
#          [10, 11, 12]],

#         [[13, 14, 15],
#          [16, 17, 18],
#          [19, 20, 21],
#          [22, 23, 24]]])
result3 = torch.cat((t1, t2), dim=-1)#dim为2，从第三个维度开始拼接
print(result3)
#dim=2，沿着第三个维度拼接，结果是一个[2,2,6]的张量,2个行2列6的矩阵
#tensor([[[ 1,  2,  3,  7,  8,  9],
#         [ 4,  5,  6, 10, 11, 12]],
#
#        [[13, 14, 15, 19, 20, 21],
#         [16, 17, 18, 22, 23, 24]]])

#unsqueeze，指定位置添加一个维度
t1 = torch.Tensor([1, 2, 3])#torch.Size([3])
t2 = t1.unsqueeze(0)
print(t1.shape)
print(t2)
print(t2.shape)#torch.Size([1, 3])



