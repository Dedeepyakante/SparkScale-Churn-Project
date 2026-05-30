from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

# Start Spark
spark = (
    SparkSession.builder
    .appName("SparkScaleChurn")
    .master("local[*]")
    .getOrCreate()
)

# Load telecom data
df = spark.read.csv(
    "data/telecom_customers.csv",
    header=True,
    inferSchema=True
)

# Register as SQL table
df.createOrReplaceTempView("telecom")

# Feature Engineering using Spark SQL
result = spark.sql("""
SELECT
    customer_id,

    AVG(data_usage_gb) AS avg_usage,

    SUM(complaints_count) AS total_complaints,

    AVG(payment_delay_days) AS avg_delay,

    MAX(monthly_bill) AS monthly_bill,

    MAX(tenure_months) AS tenure_months,

    MAX(churn) AS churn

FROM telecom

GROUP BY customer_id
""")

print("Feature table:")
result.show()

# Convert features to ML vector
assembler = VectorAssembler(

    inputCols=[
        "avg_usage",
        "total_complaints",
        "avg_delay",
        "monthly_bill",
        "tenure_months"
    ],

    outputCol="features"
)

final_df = assembler.transform(result)

print("Final ML Features:")

final_df.select(
    "customer_id",
    "features",
    "churn"
).show()

spark.stop()