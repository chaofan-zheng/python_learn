"""
输出门
    - 从单元（记忆元）中输出条目
输入门
    - 何时将数据读入单元
遗忘门
    - 重置单元内容
    - 门的计算  It = sigmoid(Xt * Wxi + Ht-1 * Whi + bi)
候选记忆元
     - C = tanh(Wt * Wxc + Ht-1 * Whc + bc)
     
"""