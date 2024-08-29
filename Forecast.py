# Import Required Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col, to_date, to_timestamp
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark Session
my_spark = SparkSession.builder.appName("SalesForecast").getOrCreate()

# Import Sales Data
sales_data = my_spark.read.csv("Online Retail.csv", header=True, inferSchema=True, sep=",")

# Convert InvoiceDate to Datetime
sales_data = sales_data.withColumn("InvoiceDate", to_date(to_timestamp(col("InvoiceDate"), "d/M/yyyy H:mm")))

# Aggregate Sales Data
agg = sales_data.groupBy("Country", "StockCode", "InvoiceDate", "Year", "Month", "Day", "Week").agg(
    F.sum("Quantity").alias("Quantity"),
    F.avg("UnitPrice").alias("AvgUnitPrice")
)

agg.show()

# Split Data into Train and Test Sets
split_date = "2011-09-25"
train_data = agg.filter(F.col("InvoiceDate") <= split_date)
test_data = agg.filter(F.col("InvoiceDate") > split_date)

pd_daily_train_data = train_data.toPandas()

print(pd_daily_train_data.head())

# Encode Categorical Columns
country_indexer = StringIndexer(inputCol="Country", outputCol="CountryIndex").setHandleInvalid("keep")
stock_code_indexer = StringIndexer(inputCol="StockCode", outputCol="StockCodeIndex").setHandleInvalid("keep")

# Assemble Features into a Vector
assembler = VectorAssembler(
    inputCols=["CountryIndex", "StockCodeIndex", "Year", "Month", "Day", "Week", "AvgUnitPrice"],
    outputCol="features"
)

# Initialize Random Forest Regressor
rf = RandomForestRegressor(featuresCol="features", labelCol="Quantity", maxBins=4000)

# Create Pipeline
pipeline = Pipeline(stages=[country_indexer, stock_code_indexer, assembler, rf])

# Fit Model to Training Data
model = pipeline.fit(train_data)

# Make Predictions on Test Data
test_predictions = model.transform(test_data)

# Convert Predictions to Double
test_predictions = test_predictions.withColumn("prediction", col("prediction").cast("double"))

# Display Predictions
test_predictions.show()

# Evaluate Model with MAE
mae_evaluator = RegressionEvaluator(labelCol="Quantity", predictionCol="prediction", metricName="mae")
mae = mae_evaluator.evaluate(test_predictions)
print("Mean Absolute Error (MAE) on test data =", mae)

# Calculate Weekly Sales for Week 39
weekly_test_predictions = test_predictions.groupBy("Year", "Week").agg({"prediction": "sum"})
promotion_week = weekly_test_predictions.filter(col('Week') == 39)
quantity_sold_w39 = int(promotion_week.select("sum(prediction)").collect()[0][0])

# Stop Spark Session
my_spark.stop()
