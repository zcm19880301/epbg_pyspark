# -*- coding:utf-8 -*-
#1-初始案例
from pyspark.sql import SparkSession
import operator
from functools import reduce
spark = SparkSession.builder.getOrCreate()
print('本案例为graphframes的初始案例')
v = spark.createDataFrame([("a",  "foo"), ("b", "bar"),], ["id", "attr"])
e = spark.createDataFrame([("a", "b", "foobar")], ["src", "dst", "rel"])
from graphframes import *
g = GraphFrame(v, e)
g.inDegrees.show()
spark.stop()

#2-hive读取数据后的demo-1
print("本案例与案例3具有相同的结果，但是本案例是不进行tube转化的这个过程")
spark = SparkSession.builder.master('local').appName("Python Spark demo").enableHiveSupport().getOrCreate()#创建SparkSession对象spark
data_frame1 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W101' limit 10")
data1=data_frame1.select("qtf_qcno","qtf_x1","qtf_x2","qtf_x3","qtf_x4")
data1_info=data1.toPandas()
data_frame2 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W102' limit 10")
data2=data_frame2.select("qtf_qcno","qtf_x1","qtf_x2","qtf_x3","qtf_x4")
data2_info=data2.toPandas()
data_frame3 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W103' limit 10")
data3=data_frame3.select("qtf_qcno","qtf_x1","qtf_x2","qtf_x3","qtf_x4")
data3_info=data3.toPandas()
data_info=data1_info.append(data2_info.append(data3_info))#将数据合并
data_info=data_info.rename(columns={'qtf_qcno':'id'})#将qtf_qcno改成id，因为GraphFrame中需要v的一个列为id
print(data_info)#查看顶点数据
v=spark.createDataFrame(data_info)
#创建一个e的graphframes
data_e=[("W101","W102","same"),("W101","W103","none"),("W102","W103","same"),]
e = spark.createDataFrame(data_e, ["src", "dst", "rel"])
print(e.collect())#查看边数据
print(v.collect())#查看顶点数据
from graphframes import *
g = GraphFrame(v, e)
# 查询每个顶点的in-degree，即入度
g.inDegrees.show()
# 查询图中同事关系的数目
g.edges.filter("rel = 'same'").count()
# 运行PageRank 算法并给出结果
results = g.pageRank(resetProbability=0.01, maxIter=20)
results.vertices.select("id", "pagerank").show()

#3-hive读取数据后的demo-2
spark = SparkSession.builder.master('local').appName("Python Spark demo").enableHiveSupport().getOrCreate()#创建SparkSession对象spark
data_frame1 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W101' limit 10")
#选择类别标号，即机械号和选择维度数据qtf_x1-qtf_x4
data1=data_frame1.select("qtf_qcno","qtf_x1","qtf_x2","qtf_x3","qtf_x4")
#将pyspark的dataframe转为pandas的dataframe，即list
data1_info=data1.toPandas()
data_frame2 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W102' limit 10")
#选择类别标号，即机械号和选择维度数据qtf_x1-qtf_x4
data2=data_frame2.select("qtf_qcno","qtf_x1","qtf_x2","qtf_x3","qtf_x4")
#将pyspark的dataframe转为pandas的dataframe，即list
data2_info=data2.toPandas()
data_frame3 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W103' limit 10")
#选择类别标号，即机械号和选择维度数据qtf_x1-qtf_x4
data3=data_frame3.select("qtf_qcno","qtf_x1","qtf_x2","qtf_x3","qtf_x4")
#将pyspark的dataframe转为pandas的dataframe，即list
data3_info=data3.toPandas()
data_info=data1_info.append(data2_info.append(data3_info))#将数据合并
print(data_info)
#创建一个v的graphframes
data_v=[]
print('graphframes需要的数据是以元组为元素的列表，所以需要转化,data_labeled是一个以元组为元素的列表')
for rows in range(len(data_info)):#graphframes需要的数据是以元组为元素的列表，所以需要转化
    #point_value=reduce(operator.add,data_info[rows:rows+1].values)
    point_value=tuple(reduce(operator.add,data_info[rows:rows+1].values))
    data_v.append(point_value)
    print(type(point_value))
print(data_v)
v = spark.createDataFrame(data_v, ["id","qtf_x1","qtf_x2","qtf_x3","qtf_x4"])
#创建一个e的graphframes
data_e=[("W101","W102","same"),("W101","W103","none"),("W102","W103","same"),]
e = spark.createDataFrame(data_e, ["src", "dst", "rel"])
print(e.collect())
print(v.collect())
from graphframes import *
g = GraphFrame(v, e)
# 查询每个顶点的in-degree，即入度
g.inDegrees.show()
# 查询图中同事关系的数目
g.edges.filter("rel = 'same'").count()
# 运行PageRank 算法并给出结果
results = g.pageRank(resetProbability=0.01, maxIter=20)
results.vertices.select("id", "pagerank").show()



#3-初始案例，创建一个grampframe，并且调用pagerank算法
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
print('创建一个graphframe，并且调用pagerank算法')
# 创建一个具有独立id列的顶点dataframe
v = spark.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
], ["id", "name", "age"])
# 创建一个具有src和dst目标的边DataFrame
e = spark.createDataFrame([
  ("a", "b", "friend"),
  ("b", "c", "follow"),
  ("c", "b", "follow"),
], ["src", "dst", "relationship"])
# 创建一个GraphFrame
from graphframes import *
g = GraphFrame(v, e)
# 查询每个顶点的in-degree，即入度
g.inDegrees.show()
# 查询图中同事关系的数目
g.edges.filter("relationship = 'follow'").count()
# 运行PageRank 算法并给出结果
results = g.pageRank(resetProbability=0.01, maxIter=20)
results.vertices.select("id", "pagerank").show()