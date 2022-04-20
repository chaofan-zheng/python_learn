from pyspark import SparkConf, SparkContext

# 初始化Spark
conf = SparkConf().setMaster("local").setAppName("My App")  # 集群url 使用local表示用单机
sc = SparkContext(conf=conf)

# wordcount
lines = sc.textFile("README.md")  # 创建一个名为lines的RDD
print(lines.count())  # 统计RDD中元素的个数
print(lines.first())  # RDD中第一个元素，也就是第一行

# 用户可以使用两种方法创建RDD
#     1. 读取外部数据集
#     2. 程序中的对象集合，比如list和set

# RDD可以进行两个操作：转化和行动

# 转化
pythonlines = lines.filter(lambda line: "Python" in line)
# 行动
pythonlines.first()

# RDD持久化
# RDD.presist()
"""
    - 转化和行动的差别：可以在任何时候定义新的RDD，但只有在第一次的一个行动操作中用到的时候才会进行真正的计算。
    - RDD在每次进行行动操作的时候会进行重新计算。如果想在多个行动操作中同用一个RDD，需要进行持久化（一般是内存）
    - 总的来说，每个spark程序都按如下方式进行工作
        1. 从外部数据创建出输入RDD
        2. 转化操作（filter）进行转化RDD
        3. 需要被重用的中间结果RDD进行persist操作
        4. 再用行动（count()和first）触发一次并行计算
"""

# 创建RDD
lines = sc.parallelize(["pandas", "i like pandas"])  # 通过已有集合创建RDD
lines = sc.textFile("README.md")  # 外部读取

# RDD操作
# 判断是转化操作还是行动操作的方法是，看他的返回值是RDD 还是其他数据类型

# 转化
inputRDD = sc.textFile("README.md")
errorsRDD = inputRDD.filter(lambda x: "error" in x)
# 转化并不会改变原有的RDD，会生成一个全新的RDD
warningsRDD = inputRDD.filter(lambda x: "warning" in x)
badlinesRDD = errorsRDD.union(warningsRDD)


# 向spark传递函数
def containsError(s):
    return "error" in s


word = lines.filter(containsError)


# 注意，如果传递某个类的对象的成员时，spark会把整个对象都发送到工作节点上，导致可能会比你想传递的东西大的多。比如
# 错误示范
class SearchFunctions():
    def __init__(self, query):
        self.query = query

    def isMatch(self, s):
        return self.query in s

    def getMatchesFunction(self, rdd):
        # 问题，在引用self.isMatch时专递了整个self
        return rdd.filter(self.isMatch)


# 正确的方法是要创建一个局部变量
class SearchFunctions2():
    def __init__(self, query):
        self.query = query

    def isMatch(self, s):
        return self.query in s

    def getMatchesFunction(self, rdd):
        # 问题，在引用self.isMatch时专递了整个self
        query = self.query
        return rdd.filter(lambda x: query in x)


# 计算RDD中各值的平方
nums = sc.parallelize([1, 2, 3, 4])
squared = nums.map(lambda x: x * x)
for num in squared:
    print(num)

# flatmap 和 map的区别
rdd1 = sc.parallelize(["python pandas", "numpy scrapy", "pytorch celery"])
rdd1_map = rdd1.map(lambda x: x.split(" "))
rdd2_map = rdd1.flatMap(lambda x: x.split(" "))
print(rdd1_map)
print(rdd2_map)  # ["python", "pandas" ....]

# 伪集合操作
rdd2 = sc.parallelize(["python", "python", "scrapy", "celery"])
rdd3 = sc.parallelize(["python", "pytorch", "celery", "pandas"])
print(rdd2.distinct)  # 去重，开销比较大
print(rdd2.union(rdd3))  # 合并
print(rdd2.intersection(rdd3))  # 交集
print(rdd2.subtract(rdd3))  # 减去

# 求用户相似度，计算两个RDD的笛卡尔积
print(rdd2.cartesian(rdd3))

# 行动操作
# 聚合
sum = lines.reduce(lambda x, y: x + y)  # RDD中所有元素的总和
print(sum)

# 计算平均值
# RDD为一个元组，0位是sum，1位是count
sumCount = nums.aggregate(
    (0, 0),
    (lambda acc, val: (acc[0] + val, acc[1] + 1)),
    (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])),
)

# 持久化（缓存）
# 为避免多次计算同一个RDD，使用持久化
# 持久化会保存他们所求出的分区数据
#



























