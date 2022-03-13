import os
import pandas as pd
import numpy as np

"""
nunpy总结
inputs = inputs.fillna(inputs.mean())  使用平均值填充nan
inputs = pd.get_dummies(inputs, dummy_na=True)  # 实现独热编码

"""

def make_data():
    """
    造数据
    :return:
    """
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')

    data = pd.read_csv(data_file)
    print(data)
    return data


def process_nan(data):
    """
    处理缺失值
    :return:
    """
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    # print(inputs.mean()) # 只对数值dtype进行运算
    inputs = inputs.fillna(inputs.mean())
    print(inputs)

    """
    由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 
    巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 
    缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
    """
    inputs = pd.get_dummies(inputs, dummy_na=True)  # 实现独热编码
    print(inputs)

    # 转换成张量格式
    X, y = np.array(inputs.values), np.array(outputs.values)
    print(X)
    print(y)


if __name__ == '__main__':
    data = make_data()
    process_nan(data)
