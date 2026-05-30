from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel

spark = (
    SparkSession.builder
    .appName("SparkScaleBatchPrediction")
    .master("local[*]")
    .getOrCreate()
)

model = PipelineModel.load(
    "models/churn_lr_model"
)

df = spark.read.csv(
    "data/telecom_customers.csv",
    header=True,
    inferSchema=True
)

predictions = model.transform(df)

predictions.select(
    "customer_id",
    "prediction",
    "probability"
).show(truncate=False)

spark.stop()