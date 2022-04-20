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

# **总结：**

## RDD常用的转化操作

- map
- flatmap
- filter
- distinct # 去重
- sample # 降采样
- union
- intersection # 交集
- subtract
- cartesian # 笛卡尔积

## RDD常用的行动操作

| 函数名   |      目的      |  示例 |  结果 |
|----------|:-------------:|------:|------:|
| collect() |  返回RDD中的所有元素。需要把所有的内容到单台机器的内存中 | rdd.collect() |  |
| count() |    rdd当中元素的个数   |    |    |
| countByValue() | 各元素的出现次数|   | {(val,count),(val,count)} |
| take(num) | 从RDD中返回num个元素（乱序）  |    |  |
| top(num) | 从RDD中返回num个元素（顺序）  |    |  |
|  takeOrdered(num)(ordering)| 按照自定义顺序返回元素  |    |  |
|  takeSample(withReplacement,num,[seed])| 随机返回元素  |    |  |
|  reduce(func)|   |    |  |
| aggregate() |   |    |  |
| foreach(func) | 对RDD中每一个元素都会使用的函数，但是不会把RDD发回到本地程序  |    |  |

## 持久化（缓存）

- 为避免多次计算同一个RDD，使用持久化
- 持久化会保存他们所求出的分区数据

| 级别   |      使用空间      |  CPU时间 |  是否在内存中 | 是否在磁盘上| 
|----------|:-------------:|------:|------:|------:|
|  MEMORY_ONLY |  高  | 低 | 是 | 否  |
| MEMORY_ONLY_SER | 低  | 高 | 是 | 否  |
| MEMORY_AND_DISK | 高  | 中等  | 部分  | 如果数据中内存放不下，会溢出写到磁盘里  |
| MEMORY_AND_DISK_SER | 低   | 高 | 部分 |  如果数据中内存放不下，会溢出写到磁盘里，内存中存放序列化数据 |
| DISK_ONLY |  低 | 高 | 否 | 是  |

## Pair RDD转化操作

`以键值对集合{(1,2),(3,4),(3,6)}为例`

| 函数名   |      目的      |  示例 |  结果 |
|----------|:-------------:|------:|------:|
| `reduceByKey(func)` |  合并具有相同键的值 | `rdd.reduceByKey((x,y)=>x+y)` | `{(1,2),(3,10)}` |
| `groupByKey()` | 对具有相同键的值进行分组     | `rdd.groupByKey()`   |  `{(1, [2]) , (3,[4 , 6])}`  |
| `combineByKey(createCombiner,mergeValue,mergeCombiner,partitioner)` |   【最常用】使用不同的返回类型合并具有相同键的值   | ``   |  ``  |
| `mapValues` |  对pairRDD中每个值应用一个函数而不改变建    | `rdd.mapValues(x=>x+1)`   |  `{(1,3), (3,5), (3,7)}`  |
| `flatMapValues(func)` |      | ``   |  ``  |
| `keys` |      | `rdd.keys()`   |  `{1,3,3}`  |
| `values` |      | `rdd.values()`   |  `{2,4,6}`  |
| `sortByKey` |      | `rdd.sortByKey()`   |  `{(1,2),(3,4),(3,6)}`  |

`以键值对集合{(1,2),(3,4),(3,6)} Other = {(3,9)}为例`

| 函数名   |      目的      |  示例 |  结果 |
|----------|:-------------:|------:|------:|
| `subtractByKey` | | `rdd.subtractByKey(other)` | `{(1,2)}` |
| `join` |  两个RDD进行内连接    | `rdd.join(other)`   |  `{(3, [4,9]) , (3,[6 , 9])}`  |
| `rightOuterjoin` |  右外连接   | `rdd.rightOuterjoin(other)`   |  `{(3, [Some(4),9]) , (3,[some(6) , 9])}`  |
| `leftOuterjoin` |  左外连接    | `rdd.leftOuterjoin(other)`   |  `{(1,(2, None)),(3, [4,some(9)]) , (3,[6 , some(9)])}`  |
| `cogroup` |  两个RDD中拥有相同键的数据分组    | `rdd.cogroup(other)`   |  `{(1, ([2], [])) , (3, ([4,6],[9]))}`  |

- 左外连接：两个表在连接过程中除了返回满足连接条件的行以外，还要返回左表中不满足条件的行，这种连接称为左外连接
- 右外连接：两个表在连接的过程中除了返回满足连接条件的行以外，还要返回右表中不满足条件的行，这种连接称为右外连接

## Pair RDD 聚合操作

combineByKey 详解

```python
# combineByKey 有多个参数，分别对应聚合操作的各个阶段
new_rdd = combineByKey(
    createCombiner,  # 遇到新键的时候，创建键和对应累加器的初始值。 
    mergeValue,  # 遇到一个在当前分区的的旧键。
    mergeCombiner,  # 因为每一个分区都是独立处理的。此函数对于同一个键的多个累加器合并
    partitioner
)

# example
# 求每一个键的平均值
sumCount = nums.combineBykey(
    (lambda x: (x, 1)),
    (lambda x, y: (x[0] + y, x[1] + 1)),
    (lambda x, y: (x[0] + y[0], x[1] + y[1])),
)
sumCount.map(lambda key, xy: (key, xy[0] / xy[1])).collectAsMap()
```

## 并行度调优

- 分区数量等于并行数量
- rdd.getNumPartitions 查看rdd的分区数

```python
data = [("a", 3), ("b", 4), ("a", 1)]
sc.parallelize(data).reduceByKey(lambda x, y: x + y)  # 默认并行度
sc.parallelize(data).reduceByKey(lambda x, y: x + y, 10)  # 自定义并行度
```
