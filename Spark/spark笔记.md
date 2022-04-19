# RDD基础
- RDD就是一个不可变的分布式对象集合
- RDD 可以是python java scala 中任一类型的对象。也可以是用户自定的对象
- 用户可以使用两种方法创建RDD
    1. 读取外部数据集
    2. 程序中的对象集合，比如list和set

```python
# 转化
pythonlines = lines.filter(lambda line: "Python" in line)
# 行动
pythonlines.first()

# RDD持久化
RDD.presist()
```
- 惰性计算：
    
    - 转化和行动的差别：可以在任何时候定义新的RDD，但只有在第一次的一个行动操作中用到的时候才会进行真正的计算。
    - RDD在每次进行行动操作的时候会进行重新计算。如果想在多个行动操作中同用一个RDD，需要进行持久化（一般是内存）
    - 总的来说，每个spark程序都按如下方式进行工作
        1. 从外部数据创建出输入RDD
        2. 转化操作（filter）进行转化RDD
        3. 需要被重用的中间结果RDD进行persist操作
        4. 再用行动（count()和first）触发一次并行计算
    - 判断是转化操作还是行动操作的方法是，看他的返回值是RDD 还是其他数据类型

```python
# 转化
inputRDD = sc.textFile("README.md")
errorsRDD = inputRDD.filter(lambda x: "error" in x)
# 转化并不会改变原有的RDD，会生成一个全新的RDD
warningsRDD = inputRDD.filter(lambda x: "warning" in x)
badlinesRDD = errorsRDD.union(warningsRDD)
```
