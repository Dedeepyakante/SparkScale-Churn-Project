from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    OneHotEncoder
)
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# -------------------------------
# Start Spark
# -------------------------------
spark = (
    SparkSession.builder
    .appName("Customer Churn Prediction")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# -------------------------------
# Load CSV
# -------------------------------
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/telecom_customers.csv")
)

print("Loaded rows:", df.count())
print("Columns:")
print(df.columns)


# -------------------------------
# Convert churn -> label
# -------------------------------
df = df.withColumn(
    "label",
    col("churn").cast(DoubleType())
)


# -------------------------------
# Handle customer_value (string)
# -------------------------------
indexer = StringIndexer(
    inputCol="customer_value",
    outputCol="customer_value_index",
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=["customer_value_index"],
    outputCols=["customer_value_vec"]
)


# -------------------------------
# Numeric columns
# -------------------------------
numeric_features = [
    "monthly_bill",
    "payment_delay_days",
    "complaints_count",
    "avg_call_duration",
    "data_usage_gb",
    "international_calls",
    "tenure_months"
]


# -------------------------------
# Assemble features
# -------------------------------
assembler = VectorAssembler(
    inputCols=numeric_features + ["customer_value_vec"],
    outputCol="features"
)


# -------------------------------
# Logistic Regression
# -------------------------------
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label"
)


# -------------------------------
# Pipeline
# -------------------------------
pipeline = Pipeline(
    stages=[
        indexer,
        encoder,
        assembler,
        lr
    ]
)


# -------------------------------
# Train / Test split
# -------------------------------
train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)


# -------------------------------
# Train model
# -------------------------------
model = pipeline.fit(train_df)


# -------------------------------
# Predictions
# -------------------------------
predictions = model.transform(test_df)

print("\nPredictions:")

predictions.select(
    "customer_id",
    "label",
    "prediction",
    "probability"
).show(truncate=False)


# -------------------------------
# Evaluate
# -------------------------------
evaluator = BinaryClassificationEvaluator(
    labelCol="label"
)

print(
    "AUC Score:",
    evaluator.evaluate(predictions)
)

print("\nTraining completed successfully.")


# -------------------------------
# Stop Spark
# -------------------------------
spark.stop()