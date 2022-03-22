import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)  # 默认是requires_grad=True, 现在返回的是一个tensor
print(variable)
t_out = torch.mean(tensor * tensor)  # x^2
v_out = torch.mean(variable * variable)  # x^2
print(t_out)
print(v_out)  # 7.5

v_out.backward()  # 模拟 v_out 的误差反向传递

# v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
# 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2

print(variable.grad)  # tensor([[0.5000, 1.0000],[1.5000, 2.0000]])
print(variable)  # variable 还是不变
