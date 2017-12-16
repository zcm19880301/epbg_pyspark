# -*- coding:utf-8 -*-

#用hive读取我们自己的数据，进行神经网络计算
from os.path import abspath
from pyspark.sql import SparkSession
from pyspark import SparkContext
import tensorflow as tf
import numpy as np
import pandas as pd
import operator
from functools import reduce
#用TensorFlow对数据进行神经网络计算
def make_layer(inputs, in_size, out_size, activate=None):#定义神经网络的层初始化
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    basis = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    result = tf.matmul(inputs, weights) + basis
    if activate is None:
        return result
    else:
        return activate(result)


class BPNeuralNetwork:#定义BP神经网络的类
    def __init__(self):
        self.session = tf.Session()
        self.input_layer = None
        self.label_layer = None
        self.loss = None
        self.optimizer = None
        self.layers = []

    def __del__(self):
        self.session.close()

    def train(self, cases, labels, limit=100, learn_rate=0.05):
        # 构建网络
        self.input_layer = tf.placeholder(tf.float32, [None, 2])
        self.label_layer = tf.placeholder(tf.float32, [None, 1])
        self.layers.append(make_layer(self.input_layer, 2, 10, activate=tf.nn.relu))
        self.layers.append(make_layer(self.layers[0], 10, 2, activate=None))
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.label_layer - self.layers[1])), reduction_indices=[1]))
        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        initer = tf.initialize_all_variables()
        # 做训练
        self.session.run(initer)
        for i in range(limit):
            self.session.run(self.optimizer, feed_dict={self.input_layer: cases, self.label_layer: labels})

    def predict(self, case):
        return self.session.run(self.layers[-1], feed_dict={self.input_layer: case})

    def test(self,data_label_list,data_list):
        print("数据信息为：")
        print(data_list)
        print("类别信息为：")
        label_info=[data_label_list]
        print(label_info)
        x_data = np.array(data_list)#训练样本
        y_data = np.array(label_info).transpose()#训练样本类别标号
        test_data = np.array([[0, 1]])#测试样本
        self.train(x_data, y_data)
        print("测试结果为：")
        print(self.predict(test_data))

#先要导入数据
#获得数据源
print('读取tdabc_qc_time_funcation表格,获得相关数据')
spark = SparkSession.builder.master('local').appName("Python Spark demo").enableHiveSupport().getOrCreate()#创建SparkSession对象spark
#读取W101的数据
data_frame1 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W101' limit 10")
#选择类别标号，即机械号和选择维度数据qtf_x1-qtf_x4
data1_value=data_frame1.select("qtf_x1","qtf_x2")
data1_label=data_frame1.select("qtf_qcno")
#将pyspark的dataframe转为pandas的dataframe，即list
data1_value_info=data1_value.toPandas()
data1_label_info=data1_label.toPandas()
#标签设定
data1_label_info["qtf_qcno"]=0
#读取W102的数据
data_frame2 = spark.sql("select * from muku.tdabc_qc_time_funcation where qtf_qcno='W102' limit 10")
#选择类别标号，即机械号和选择维度数据qtf_x1-qtf_x4
data2_value=data_frame2.select("qtf_x1","qtf_x2")
data2_label=data_frame2.select("qtf_qcno")
#将pyspark的dataframe转为pandas的dataframe，即list
data2_value_info=data2_value.toPandas()
data2_label_info=data2_label.toPandas()
#标签设定
data2_label_info["qtf_qcno"]=1
#将data1和data2合并
data_value=data1_value_info.append(data2_value_info)
data_label=data1_label_info.append(data2_label_info)
print(data_value)#打印样本数据
print(data_label)#打印样本标签
#下面将data分成两部分，一部分是label，另外一部分是feature，是一个Labeledpoint的RDD，为此需要导入from pyspark.mllib.linalg import Vector,Vectors
#from pyspark.mllib.regression import LabeledPoint
data_list=[]#将pandas中可以操作的dataframe逐步读取，变成list
data_label_list=[]
for rows in range(len(data_value)):
    point_label=reduce(operator.add,reduce(operator.add,np.array(data_label[rows:rows+1]).tolist()))#将dataframe数据转成list形式，再去掉list外面的[]
    data_value_in_list=reduce(operator.add,np.array(data_value[rows:rows+1]).tolist())#将dataframe数据转成list形式，再去掉list外面的[]
    #print(data_value_in_list)
    #print(type(data_value_in_list))
    data_list.append(data_value_in_list)
    data_label_list.append(point_label)
print("样本信息如下(list形式)：")
print(data_list)
print(data_label_list)

nn = BPNeuralNetwork()#创建一个对象赋给nn
nn.test(data_label_list,data_list)#调用测试方法