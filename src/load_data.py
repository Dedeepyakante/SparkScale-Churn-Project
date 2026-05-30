from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("SparkScaleChurn")
    .master("local[*]")
    .getOrCreate()
)

df = spark.read.csv(
    "data/telecom_customers.csv",
    header=True,
    inferSchema=True
)

print("CSV loaded")

df.show()

df.printSchema()

print("Rows:", df.count())

spark.stop()