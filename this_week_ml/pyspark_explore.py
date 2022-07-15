import os
import traceback
import matplotlib.pyplot as plt
from pyspark.ml.feature import PCA
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

schema = StructType([ \
    StructField("id",StringType(),True), \
    StructField("f_00", DoubleType(),True), \
    StructField("f_01", DoubleType(),True), \
    StructField("f_02", DoubleType(),True), \
    StructField("f_03", DoubleType(),True), \
    StructField("f_04", DoubleType(),True), \
    StructField("f_05", DoubleType(),True), \
    StructField("f_06", DoubleType(),True), \
    StructField("f_07", DoubleType(),True), \
    StructField("f_08", DoubleType(),True), \
    StructField("f_09", DoubleType(),True), \
    StructField("f_10", DoubleType(),True), \
    StructField("f_11", DoubleType(),True), \
    StructField("f_12", DoubleType(),True), \
    StructField("f_13", DoubleType(),True), \
    StructField("f_14", DoubleType(),True), \
    StructField("f_15", DoubleType(),True), \
    StructField("f_16", DoubleType(),True), \
    StructField("f_17", DoubleType(),True), \
    StructField("f_18", DoubleType(),True), \
    StructField("f_19", DoubleType(),True), \
    StructField("f_20", DoubleType(),True), \
    StructField("f_21", DoubleType(),True), \
    StructField("f_22", DoubleType(),True), \
    StructField("f_23", DoubleType(),True), \
    StructField("f_24", DoubleType(),True), \
    StructField("f_25", DoubleType(),True), \
    StructField("f_26", DoubleType(),True), \
    StructField("f_27", DoubleType(),True), \
    StructField("f_28", DoubleType(),True)])

class KaggleDemo:
    def __init__(self, spark):
        self.spark = spark
        self.km = KMeans(k=8, seed=10)
        self.pc = PCA(k=8, inputCol='features')

    def get_data(self):
        return self.spark.read.options(header=True).schema(schema).csv('../data.csv')

    def create_cluster(self, va2):
        model = self.km.fit(va2)
        count_dct = {k: v for k,v in enumerate(model.summary.clusterSizes)}
        plt.bar(count_dct.keys(), count_dct.values())
        plt.show()
        return model

    def get_pca_plot(self, va2):
        model = self.pc.fit(va2)
        count_dct = {k: v for k,v in enumerate(model.explainedVariance)}
        plt.bar(count_dct.keys(), count_dct.values())
        plt.show()
        return model


    def start(self):
        df = self.get_data()
        df.printSchema()
        va = VectorAssembler(outputCol="features").setInputCols(df.columns[1:])
        va2 = va.transform(df.drop('id'))
        self.get_pca_plot(va2)
        self.create_cluster(va2)
        print("Done")

if __name__ == "__main__":
    spark = SparkSession.builder.appName('kaggle_demo')\
        .getOrCreate()
    kd = KaggleDemo(spark)
    kd.start()


