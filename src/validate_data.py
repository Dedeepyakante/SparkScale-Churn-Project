from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

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

print("Schema:")
df.printSchema()

print("Total rows:")
print(df.count())

print("Duplicate rows:")
print(df.count() - df.dropDuplicates().count())

print("Missing values:")

df.select([
    count(
        when(col(c).isNull(), c)
    ).alias(c)

    for c in df.columns

]).show()

spark.stop()