import torch
from torch import nn


# # 首先我们定义一个模型
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(3, 4)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(4, 1)
#         self.initialize()
#
#     # 为了方便验证，我们将指定特殊的weight和bias
#     def initialize(self):
#         with torch.no_grad():
#             self.fc1.weight = torch.nn.Parameter(
#                 torch.Tensor([[1., 2., 3.],
#                               [-4., -5., -6.],
#                               [7., 8., 9.],
#                               [-10., -11., -12.]]))
#
#             self.fc1.bias = torch.nn.Parameter(torch.Tensor([1.0, 2.0, 3.0, 4.0]))
#             self.fc2.weight = torch.nn.Parameter(torch.Tensor([[1.0, 2.0, 3.0, 4.0]]))
#             self.fc2.bias = torch.nn.Parameter(torch.Tensor([1.0]))
#
#     def forward(self, x):
#         o = self.fc1(x)
#         o = self.relu1(o)
#         o = self.fc2(o)
#         return o
#
#
# # 全局变量，用于存储中间层的 feature
# total_feat_out = []
# total_feat_in = []
#
#
# # 定义 forward hook function
# def hook_fn_forward(module, input, output):
#     print(module)  # 用于区分模块
#     print('input', input)  # 首先打印出来
#     print('output', output)
#     total_feat_out.append(output)  # 然后分别存入全局 list 中
#     total_feat_in.append(input)
#
#
# model = Model()
#
# modules = model.named_children()  #
# for name, module in modules:
#     module.register_forward_hook(hook_fn_forward)
#     # module.register_backward_hook(hook_fn_backward)
#
# # 注意下面代码中 x 的维度，对于linear module，输入一定是大于等于二维的
# # （第一维是 batch size）。在 forward hook 中看不出来，但是 backward hook 中，
# # 得到的梯度完全不对。
# # 有一篇 hook 的教程就是这里出了错，作者还强行解释，遗毒无穷，
#
# x = torch.Tensor([[1.0, 1.0, 1.0]]).requires_grad_()
# o = model(x)
# o.backward()
#
# print('==========Saved inputs and outputs==========')
# for idx in range(len(total_feat_in)):
#     print('input: ', total_feat_in[idx])
#     print('output: ', total_feat_out[idx])

a = torch.randint(1,10,(2,2))
print(a)
index = a < 5
print(index)
print(torch.sum(index))