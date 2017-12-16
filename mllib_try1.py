# -*- coding:utf-8 -*-
from os.path import abspath
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pandas as pd
import numpy as np
import scipy.sparse as sps
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vector,Vectors
from pyspark.mllib.regression import LabeledPoint
import operator
from functools import reduce
#获得数据源
print('读取tdabc_qc_time_funcation表格，并导入from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel,from pyspark.mllib.util import MLUtils，进行朴素贝叶斯算法')
spark = SparkSession.builder.master('local').appName("Python Spark demo").enableHiveSupport().getOrCreate()#创建SparkSession对象spark
#读取W101的数据
#data_frame1 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W101' order by qtf_x3 limit 20")#选取前20条数据
data_frame1 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W101' order by qtf_x3")
#选择类别标号，即机械号和选择维度数据qtf_x1-qtf_x4
data1_value=data_frame1.select("qtf_x1","qtf_x2","qtf_x3","qtf_x4")
data1_label=data_frame1.select("qtf_qcno")
#将pyspark的dataframe转为pandas的dataframe，即list
data1_value_info=data1_value.toPandas()
data1_label_info=data1_label.toPandas()
#标签设定
data1_label_info["qtf_qcno"]=0
#读取W102的数据
data_frame2 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W102' order by qtf_x3")
#选择类别标号，即机械号和选择维度数据qtf_x1-qtf_x4
data2_value=data_frame2.select("qtf_x1","qtf_x2","qtf_x3","qtf_x4")
data2_label=data_frame2.select("qtf_qcno")
#将pyspark的dataframe转为pandas的dataframe，即list
data2_value_info=data2_value.toPandas()
data2_label_info=data2_label.toPandas()
#标签设定
data2_label_info["qtf_qcno"]=1
#将data1和data2合并
data_value=data1_value_info.append(data2_value_info)
data_label=data1_label_info.append(data2_label_info)
#print(data_value)#打印样本数据
#print(data_label)#打印样本标签
#下面将data分成两部分，一部分是label，另外一部分是feature，是一个Labeledpoint的RDD，为此需要导入from pyspark.mllib.linalg import Vector,Vectors
#from pyspark.mllib.regression import LabeledPoint
data_labeled=[]#将pandas中可以操作的dataframe逐步读取，变成list
"""
print(LabeledPoint(data_label[0:1].values,reduce(operator.add,data_value[0:1].values)))
data_labeled.append(LabeledPoint(data_label[0:1].values,reduce(operator.add,data_value[0:1].values)))
data_labeled.append(LabeledPoint(data_label[0:1].values,reduce(operator.add,data_value[0:1].values)))
print(data_labeled)
"""
for rows in range(len(data_value)):
    point_label=reduce(operator.add,reduce(operator.add,data_label[rows:rows+1].values))
    point_value=reduce(operator.add,data_value[rows:rows+1].values)
    point=LabeledPoint(point_label, point_value)
    data_labeled.append(point)
print("样本信息如下(list形式)：")

print(data_labeled)
#list完成转化为Spark的dataframe
data=spark.createDataFrame(data_labeled)
print("样本信息如下(dataframe形式)：")
data.show()
#将dataframe再转成rdd，从而方便mllib库要求的rdd of labeledpoint
data1=data.rdd.map(lambda row: LabeledPoint(row.label, row.features))
print(data1)
# 将样本分成60%的训练样本和40%的测试样本
training, test = data1.randomSplit([0.6, 0.4])
print(training)#打印训练点
print(test)#打印测试点
# 训练一个原生的贝叶斯模型.
model = NaiveBayes.train(training, 1.0)
#进行预测和测试识别
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('model accuracy {}'.format(accuracy))

""""""





