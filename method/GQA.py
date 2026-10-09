import torch
import torch.nn as nn


#dropout：随机丢弃神经元，防止过拟合。p是丢弃的概率，p=0.5表示有50%的概率丢弃神经元
dropout_layer = nn.Dropout(p=0.5)

t1 = torch.Tensor([1,2,3])
t2 = dropout_layer(t1)
#这里Dropout丢弃为了保持期望不变，将其他部分扩大两倍，扩大的倍数是1/(1-p)
print(t2)



# linear层,in_features是输入特征的维度，out_features是输出特征的维度，bias是是否使用偏置,就是wx+b中的b
layer = nn.Linear(in_features=3, out_features=5, bias=True)
t1 = torch.Tensor([1,2,3]) #shape=(3,)

t2 = torch.Tensor([[1,2,3]])#shape=(1,3)

output2 = layer(t2)#shape=(1,5),输入维度是3，输出维度是5
print(output2)#tensor([[ 0.0717, -1.2983,  0.8586,  0.4888, -1.9707]]
# 本质就是线性变化，就是对应的张量乘以x再加b，这里w和b是随机的，真实训练中会再optimizer中更新w和b的值，最终得到最优的w和b



#view方法,改变张量的形状，返回一个新的张量
t = torch.tensor([[1, 2, 3,4,5,6],[7, 8, 9,10,11,12]])#shape=(2,6)
t_view1 = t.view(3, 4)#shape=(3,4)
print(t_view1)
t_view2 = t.view(4,3)#shape=(4,3)
print(t_view2)



#tanspose方法,转置张量，返回一个新的张量,或者说交换维度（2，3，3）-》(3，2，3)【transpose（0，2）】
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])#shape=(2,3)  
t1 = t1.transpose(0, 1)#shape=(3,2)
print(t1)



#triu方法，生成一个下三角矩阵，返回一个新的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(torch.triu(x))#tensor([[1, 2, 3],
#                            [0, 5, 6],
#                            [0, 0, 9]])
print(torch.triu(x, diagonal=1))#tensor([[0, 2, 3],
#                                        [0, 0, 6],
#                                        [0, 0, 0]])
print(torch.triu(x, diagonal=2))#tensor([[0, 0, 3],
#                                        [0, 0, 0],
#                                        [0, 0, 0]])
print(torch.triu(x, diagonal=-1))#tensor([[1, 2, 3],
#                                         [4, 5, 6],
#                                         [0, 8, 9]])



#reshape方法，改变张量的形状，返回一个新的张量,和view方法类似，但是reshape可以改变张量的内存布局，而view不可以
x = torch.arange(1,7)
y = torch.reshape(x, (2, 3))
print(y)
#输出：
#tensor([[1, 2, 3],
#        [4, 5, 6]])

#使用-1自动推断
z = torch.reshape(x, (2, -1))
print(z)
#输出：
#tensor([[1, 2, 3],
#        [4, 5, 6]])



