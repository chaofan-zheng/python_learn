from pyspark import SparkConf, SparkContext

# 初始化Spark
conf = SparkConf().setMaster("local").setAppName("My App")  # 集群url 使用local表示用单机
sc = SparkContext(conf=conf)

# wordcount
lines = sc.textFile("README.md")  # 创建一个名为lines的RDD

# 创建pair RDD
pairs = lines.map(lambda x: (x.split(" ")[0], x))  # 使用第一个单次为键创建出一个二元组键值对

# 用python实现单词计数
rdd = sc.textFile("s3://...")
words = rdd.flatMap(lambda x: x.split(" "))
result = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)

# example
# 求每一个键的平均值
sumCount = nums.combineBykey(
    (lambda x: (x,1)),
    (lambda x,y:(x[0] + y , x[1] + 1)),
    (lambda x,y:(x[0] + y[0] , x[1] + y[1])),
)
sumCount.map(lambda key, xy: (key,xy[0]/xy[1])).collectAsMap()