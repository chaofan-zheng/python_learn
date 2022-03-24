from collections import defaultdict

# dict =defaultdict( factory_function)
# 工厂函数，当key不存在的时候返回的是这个工厂函数
dict1 = defaultdict(int)
dict2 = defaultdict(set)
dict3 = defaultdict(str)
dict4 = defaultdict(list)
dict1[2] = 'two'

print(dict1[1])
print(dict1[2])
print(dict2[1])
print(dict3[1])
print(dict4[1])
