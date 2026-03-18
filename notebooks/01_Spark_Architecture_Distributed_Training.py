# Databricks notebook source

# MAGIC %md
# MAGIC # Module 1: Spark Architecture & Distributed Training
# MAGIC
# MAGIC **Training Event:** ServiceNow x Databricks — AI/ML on Lakehouse
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this module, you will be able to:
# MAGIC
# MAGIC 1. Explain Spark's distributed architecture (driver, workers, partitions, shuffle) and how it applies to ML workloads
# MAGIC 2. Build a feature engineering pipeline using Spark SQL and DataFrame joins
# MAGIC 3. Train classification models using `pyspark.ml` (Logistic Regression, Random Forest)
# MAGIC 4. Evaluate binary classifiers with AUC, accuracy, and F1 score
# MAGIC 5. Use the Pandas API on Spark and Pandas UDFs for distributed inference
# MAGIC 6. Understand the newest Spark ML capabilities (`pyspark.ml.connect`, `TorchDistributor`)
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Run **Module 0** first to create the `servicenow_training` schema and all GTM tables
# MAGIC - Compute: Serverless (no cluster configuration needed)
# MAGIC
# MAGIC **Estimated Runtime:** ~15 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")
print(f"Active catalog/schema: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 1: Spark Architecture for ML
# MAGIC
# MAGIC ## 1.1 — How Spark Distributes Work
# MAGIC
# MAGIC Apache Spark uses a **driver-worker architecture** for distributed computation:
# MAGIC
# MAGIC | Component | Role |
# MAGIC |-----------|------|
# MAGIC | **Driver** | Coordinates the job. Parses your code, builds a logical plan, optimizes it (Catalyst), and schedules tasks. |
# MAGIC | **Workers (Executors)** | Execute tasks in parallel. Each executor runs on a separate machine and processes a subset of the data. |
# MAGIC | **Partitions** | The unit of parallelism. A DataFrame is split into partitions, and each task processes one partition. |
# MAGIC | **Shuffle** | Redistribution of data across executors — required for joins, aggregations, and sorts. Shuffles are expensive (disk + network I/O). |
# MAGIC
# MAGIC For **ML workloads**, this means:
# MAGIC - Feature engineering (joins, aggregations) runs in parallel across workers
# MAGIC - `pyspark.ml` algorithms are implemented as distributed operations (e.g., gradient descent sends partial gradients from each partition to the driver)
# MAGIC - The **driver** is the bottleneck if you `.collect()` too much data or use single-node libraries
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 — Load Data from Unity Catalog
# MAGIC
# MAGIC Let's start by loading the tables we created in Module 0.

# COMMAND ----------

df_lead_scores = spark.table(f"{catalog}.{schema}.gtm_lead_scores")
df_contacts    = spark.table(f"{catalog}.{schema}.gtm_contacts")
df_activities  = spark.table(f"{catalog}.{schema}.gtm_activities")
df_campaigns   = spark.table(f"{catalog}.{schema}.gtm_campaign_members")
df_accounts    = spark.table(f"{catalog}.{schema}.gtm_accounts")

print("Tables loaded from Unity Catalog:")
print(f"  lead_scores     : {df_lead_scores.count():>10,} rows")
print(f"  contacts        : {df_contacts.count():>10,} rows")
print(f"  activities      : {df_activities.count():>10,} rows")
print(f"  campaign_members: {df_campaigns.count():>10,} rows")
print(f"  accounts        : {df_accounts.count():>10,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 — Build a Rich Feature Table
# MAGIC
# MAGIC A good ML model needs rich, informative features. We will join multiple tables and engineer
# MAGIC features that capture:
# MAGIC
# MAGIC - **Activity signals:** total activities, average sentiment, activity type breakdown
# MAGIC - **Campaign engagement:** total email opens, clicks, form fills, response rate
# MAGIC - **Firmographic fit:** employee count, annual revenue, account tier
# MAGIC - **Contact attributes:** seniority level, lead source, department
# MAGIC - **Lead scores:** engagement, fit, behavior, recency scores
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# --- Step 1: Aggregate activities per contact ---
df_activity_agg = (
    df_activities
    .groupBy("contact_id")
    .agg(
        F.count("*").alias("total_activities"),
        F.avg("sentiment_score").alias("avg_sentiment"),
        F.sum(F.when(F.col("outcome") == "Positive", 1).otherwise(0)).alias("positive_outcomes"),
        F.sum(F.when(F.col("outcome") == "Negative", 1).otherwise(0)).alias("negative_outcomes"),
        F.sum(F.when(F.col("activity_type") == "Email", 1).otherwise(0)).alias("email_count"),
        F.sum(F.when(F.col("activity_type") == "Meeting", 1).otherwise(0)).alias("meeting_count"),
        F.sum(F.when(F.col("activity_type") == "Demo", 1).otherwise(0)).alias("demo_count"),
        F.sum(F.when(F.col("activity_type") == "Call", 1).otherwise(0)).alias("call_count"),
        F.avg("duration_minutes").alias("avg_duration_minutes"),
        F.max("activity_date").alias("last_activity_date"),
    )
)

print("Activity aggregates per contact:")
df_activity_agg.show(5, truncate=False)

# COMMAND ----------

# --- Step 2: Aggregate campaign engagement per contact ---
df_campaign_agg = (
    df_campaigns
    .groupBy("contact_id")
    .agg(
        F.count("*").alias("campaigns_participated"),
        F.sum("email_opens").alias("email_opens_total"),
        F.sum("email_clicks").alias("email_clicks_total"),
        F.sum("form_fills").alias("form_fills_total"),
        F.sum(F.when(F.col("status") == "Responded", 1).otherwise(0)).alias("campaign_responses"),
        F.sum(F.when(F.col("status") == "Converted", 1).otherwise(0)).alias("campaign_conversions"),
    )
)

print("Campaign engagement aggregates per contact:")
df_campaign_agg.show(5, truncate=False)

# COMMAND ----------

# --- Step 3: Join everything into a single feature table ---

# Start with lead scores (one row per contact — this is our label table)
df_features = (
    df_lead_scores
    .join(df_contacts, on="contact_id", how="inner")
    .join(df_accounts, on="account_id", how="left")
    .join(df_activity_agg, on="contact_id", how="left")
    .join(df_campaign_agg, on="contact_id", how="left")
)

# --- Step 4: Add computed features ---
df_features = (
    df_features
    # Days since contact creation (relative to a fixed reference date for reproducibility)
    .withColumn(
        "days_since_creation",
        F.datediff(F.lit("2026-03-18"), F.col("created_date"))
    )
    # Days since last activity
    .withColumn(
        "days_since_last_activity",
        F.datediff(F.lit("2026-03-18"), F.col("last_activity_date"))
    )
    # Fill nulls for contacts with no activities or campaigns
    .fillna(0, subset=[
        "total_activities", "positive_outcomes", "negative_outcomes",
        "email_count", "meeting_count", "demo_count", "call_count",
        "campaigns_participated", "email_opens_total", "email_clicks_total",
        "form_fills_total", "campaign_responses", "campaign_conversions",
    ])
    .fillna(0.5, subset=["avg_sentiment"])
    .fillna(0.0, subset=["avg_duration_minutes"])
    .fillna(365, subset=["days_since_last_activity"])
)

# Select final feature columns
feature_columns = [
    # Label
    "contact_id", "converted",
    # Numeric features
    "engagement_score", "fit_score", "behavior_score", "recency_score",
    "total_activities", "avg_sentiment", "positive_outcomes", "negative_outcomes",
    "email_count", "meeting_count", "demo_count", "call_count",
    "avg_duration_minutes", "days_since_last_activity",
    "campaigns_participated", "email_opens_total", "email_clicks_total",
    "form_fills_total", "campaign_responses", "campaign_conversions",
    "employee_count", "annual_revenue", "days_since_creation",
    # Categorical features (to be encoded)
    "seniority_level", "lead_source", "industry", "account_tier",
]

df_features = df_features.select(*feature_columns)

print(f"Feature table: {df_features.count()} rows x {len(df_features.columns)} columns")
df_features.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 — Inspect the Execution Plan
# MAGIC
# MAGIC Spark's **Catalyst optimizer** rewrites your query into an efficient physical plan. The `.explain(True)` method
# MAGIC shows all four plan stages:
# MAGIC
# MAGIC 1. **Parsed Logical Plan** — raw translation of your code
# MAGIC 2. **Analyzed Logical Plan** — schema resolution, column binding
# MAGIC 3. **Optimized Logical Plan** — predicate pushdown, projection pruning, join reordering
# MAGIC 4. **Physical Plan** — concrete execution strategy (broadcast hash join vs. sort-merge join, scan type, etc.)
# MAGIC

# COMMAND ----------

# Show the full execution plan for our feature table
df_features.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5 — Partitions and Parallelism
# MAGIC
# MAGIC The number of **partitions** determines how many tasks run in parallel. Too few partitions
# MAGIC means under-utilized workers; too many means excessive overhead.
# MAGIC
# MAGIC **Rule of thumb:** 2-4 partitions per available CPU core. For serverless, Spark auto-tunes this.

# COMMAND ----------

print(f"Current number of partitions: {df_features.rdd.getNumPartitions()}")

# Repartition to a specific number (useful when writing output for downstream consumers)
df_repartitioned = df_features.repartition(8)
print(f"After repartition(8):         {df_repartitioned.rdd.getNumPartitions()}")

# Coalesce reduces partitions WITHOUT a full shuffle (more efficient for reducing)
df_coalesced = df_features.coalesce(4)
print(f"After coalesce(4):            {df_coalesced.rdd.getNumPartitions()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 2: Distributed Training with Spark ML
# MAGIC
# MAGIC ## 2.1 — The `pyspark.ml` Pipeline
# MAGIC
# MAGIC Spark ML uses a **Pipeline** abstraction inspired by scikit-learn:
# MAGIC
# MAGIC | Concept | Description |
# MAGIC |---------|-------------|
# MAGIC | **Transformer** | Takes a DataFrame, returns a DataFrame (e.g., `VectorAssembler`, `StringIndexer.fit().transform()`) |
# MAGIC | **Estimator** | Takes a DataFrame, fits a model, returns a Transformer (e.g., `LogisticRegression`, `RandomForestClassifier`) |
# MAGIC | **Pipeline** | Chains multiple stages (Estimators + Transformers) into a single workflow |
# MAGIC | **ParamGrid** | Defines hyperparameter search space for cross-validation |
# MAGIC
# MAGIC All of these run **distributed** — the heavy computation happens on the workers, not the driver.
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml import Pipeline

print("Spark ML imports loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 — Encode Categorical Features
# MAGIC
# MAGIC `StringIndexer` converts categorical strings (e.g., "Enterprise", "Mid-Market", "SMB") into
# MAGIC numeric indices (0, 1, 2). We do this for all four categorical columns.

# COMMAND ----------

categorical_cols = ["seniority_level", "lead_source", "industry", "account_tier"]

indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_idx",
        handleInvalid="keep"  # Assign new index for unseen categories at inference time
    )
    for col in categorical_cols
]

print("StringIndexer stages created for:", categorical_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 — Assemble Feature Vector
# MAGIC
# MAGIC `VectorAssembler` combines all numeric and indexed columns into a single `features` vector
# MAGIC column — the format required by all Spark ML algorithms.

# COMMAND ----------

numeric_cols = [
    "engagement_score", "fit_score", "behavior_score", "recency_score",
    "total_activities", "avg_sentiment", "positive_outcomes", "negative_outcomes",
    "email_count", "meeting_count", "demo_count", "call_count",
    "avg_duration_minutes", "days_since_last_activity",
    "campaigns_participated", "email_opens_total", "email_clicks_total",
    "form_fills_total", "campaign_responses", "campaign_conversions",
    "employee_count", "annual_revenue", "days_since_creation",
]

indexed_cols = [f"{col}_idx" for col in categorical_cols]

all_feature_cols = numeric_cols + indexed_cols

assembler = VectorAssembler(
    inputCols=all_feature_cols,
    outputCol="features_raw",
    handleInvalid="skip",
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True,
)

print(f"Feature vector will contain {len(all_feature_cols)} features:")
for i, col in enumerate(all_feature_cols, 1):
    print(f"  {i:>2}. {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 — Train/Test Split
# MAGIC
# MAGIC We use an 80/20 split with a fixed seed for reproducibility.

# COMMAND ----------

# Cast 'converted' to double (Spark ML expects a numeric label column)
df_ml = df_features.withColumn("label", F.col("converted").cast("double"))

train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)

print(f"Training set : {train_df.count():>6,} rows")
print(f"Test set     : {test_df.count():>6,} rows")

# Check class balance
print("\nClass distribution (training):")
train_df.groupBy("label").count().orderBy("label").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 — Train Logistic Regression
# MAGIC
# MAGIC Logistic Regression is a fast, interpretable baseline for binary classification.
# MAGIC In Spark ML, the optimization (L-BFGS) runs **distributed** across partitions.

# COMMAND ----------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.5,  # L1/L2 mix (0.5 = Elastic Net)
)

# Build the full pipeline: indexers → assembler → scaler → model
lr_pipeline = Pipeline(stages=indexers + [assembler, scaler, lr])

# Fit the pipeline
print("Training Logistic Regression pipeline...")
lr_model = lr_pipeline.fit(train_df)
print("Training complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.6 — Train Random Forest
# MAGIC
# MAGIC Random Forest is an ensemble method that trains many decision trees in parallel —
# MAGIC a natural fit for Spark's distributed architecture. Each tree can be trained on a
# MAGIC different partition of the data.

# COMMAND ----------

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=10,
    seed=42,
)

rf_pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])

print("Training Random Forest pipeline (100 trees)...")
rf_model = rf_pipeline.fit(train_df)
print("Training complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.7 — Evaluate Both Models
# MAGIC
# MAGIC We evaluate on the held-out test set using:
# MAGIC - **AUC-ROC** — Area Under the ROC Curve (ranking quality; 0.5 = random, 1.0 = perfect)
# MAGIC - **Accuracy** — Fraction of correct predictions
# MAGIC - **F1 Score** — Harmonic mean of precision and recall (important for imbalanced classes)

# COMMAND ----------

# Evaluators
binary_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
accuracy_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

results = {}

for name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
    predictions = model.transform(test_df)
    auc = binary_eval.evaluate(predictions)
    acc = accuracy_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)

    results[name] = {"AUC": auc, "Accuracy": acc, "F1": f1}

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.8 — Feature Importance (Random Forest)
# MAGIC
# MAGIC Random Forest provides built-in feature importance based on how much each feature
# MAGIC reduces impurity (Gini) across all trees. This helps the sales team understand
# MAGIC **which signals matter most** for lead conversion.
# MAGIC

# COMMAND ----------

import pandas as pd

# Extract the Random Forest model from the pipeline (last stage)
rf_classifier = rf_model.stages[-1]
importances = rf_classifier.featureImportances.toArray()

# Map to feature names
importance_df = pd.DataFrame({
    "feature": all_feature_cols,
    "importance": importances,
}).sort_values("importance", ascending=False)

print("Top 15 Features by Importance:\n")
print(f"{'Rank':<6} {'Feature':<35} {'Importance':>12}")
print("-" * 55)
for rank, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
    bar = "#" * int(row["importance"] * 200)
    print(f"{rank:<6} {row['feature']:<35} {row['importance']:>10.4f}  {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.9 — View Predictions
# MAGIC
# MAGIC Let's look at some example predictions to sanity-check the model output.

# COMMAND ----------

predictions_rf = rf_model.transform(test_df)

(
    predictions_rf
    .select(
        "contact_id",
        "label",
        "prediction",
        "probability",
        "engagement_score",
        "fit_score",
        "total_activities",
        "demo_count",
    )
    .show(10, truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 3: Pandas API on Spark & Pandas UDFs
# MAGIC
# MAGIC ## 3.1 — Why Pandas on Spark?
# MAGIC
# MAGIC Many data scientists prefer the **pandas** API for its expressiveness. The **Pandas API on Spark**
# MAGIC (`pyspark.pandas`) lets you write pandas-style code that executes on Spark under the hood —
# MAGIC getting distributed processing without rewriting your code.
# MAGIC
# MAGIC | Approach | Runs On | Scale | API |
# MAGIC |----------|---------|-------|-----|
# MAGIC | `pandas` | Single node (driver) | GBs | pandas |
# MAGIC | `pyspark.pandas` | Spark cluster (workers) | TBs | pandas-like |
# MAGIC | `pyspark.sql` | Spark cluster (workers) | TBs | DataFrame |
# MAGIC

# COMMAND ----------

import pyspark.pandas as ps

# Convert our Spark DataFrame to a pandas-on-Spark DataFrame
psdf = df_features.pandas_api()

print(f"Type: {type(psdf)}")
print(f"Shape: {psdf.shape}")
psdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 — Train with scikit-learn on Pandas-on-Spark
# MAGIC
# MAGIC We can use familiar scikit-learn code by converting to a (local) pandas DataFrame.
# MAGIC This works well when the dataset fits in driver memory (typically under ~10 GB).
# MAGIC

# COMMAND ----------

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Convert to pandas (fits in memory for our 10K-row dataset)
pdf = df_features.toPandas()

# Encode categoricals with LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    pdf[col + "_encoded"] = le.fit_transform(pdf[col].fillna("Unknown"))
    label_encoders[col] = le

# Prepare feature matrix
sklearn_feature_cols = numeric_cols + [f"{c}_encoded" for c in categorical_cols]
X = pdf[sklearn_feature_cols].fillna(0).values
y = pdf["converted"].values

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Gradient Boosting
gbt = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
)
gbt.fit(X_train, y_train)

# Evaluate
y_pred = gbt.predict(X_test)
y_prob = gbt.predict_proba(X_test)[:, 1]

print("Gradient Boosting (scikit-learn) — Test Results:")
print(f"  AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 — Pandas UDFs for Distributed Inference
# MAGIC
# MAGIC **Pandas UDFs** (also called Vectorized UDFs) let you apply a Python function to each
# MAGIC **partition** of a Spark DataFrame. The data arrives as a pandas Series or DataFrame,
# MAGIC and you return a pandas Series. Spark handles the distribution automatically.
# MAGIC
# MAGIC This is the **best pattern for distributed inference** with scikit-learn models:
# MAGIC 1. Train a model on the driver (as above)
# MAGIC 2. Broadcast the model to all workers
# MAGIC 3. Apply it via a Pandas UDF across all partitions
# MAGIC

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pickle

# Broadcast the trained sklearn model to all workers
model_broadcast = spark.sparkContext.broadcast(gbt)

# Broadcast the feature column list (order matters for the model)
feature_cols_broadcast = spark.sparkContext.broadcast(sklearn_feature_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Pandas UDF
# MAGIC
# MAGIC The UDF receives a `pandas.DataFrame` (one partition at a time) and returns a `pandas.Series`
# MAGIC of predicted probabilities.

# COMMAND ----------

@pandas_udf(DoubleType())
def predict_conversion_probability(*cols: pd.Series) -> pd.Series:
    """
    Distributed inference using a broadcasted scikit-learn model.
    Each argument is a pandas Series corresponding to one feature column.
    Returns a pandas Series of predicted conversion probabilities.
    """
    # Stack all feature columns into a 2D numpy array
    features = pd.concat(cols, axis=1).fillna(0).values
    # Get the model from the broadcast variable
    model = model_broadcast.value
    # Predict probability of class 1 (converted)
    probabilities = model.predict_proba(features)[:, 1]
    return pd.Series(probabilities)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply the UDF to the Full DataFrame
# MAGIC
# MAGIC The key insight: this runs **on the workers**, not the driver. Even if the DataFrame had
# MAGIC billions of rows, the inference would be distributed.

# COMMAND ----------

# We need to encode categoricals in Spark before passing to the UDF.
# Use the StringIndexer outputs from Section 2 for consistency.

# Re-transform using our fitted pipeline's indexers (just the indexer+assembler stages)
from pyspark.ml import Pipeline

# Fit indexers on full dataset
encoding_pipeline = Pipeline(stages=indexers)
encoding_model = encoding_pipeline.fit(df_ml)
df_encoded = encoding_model.transform(df_ml)

# Build the list of columns to pass to the UDF (same order as sklearn training)
udf_input_cols = numeric_cols + [f"{c}_idx" for c in categorical_cols]

# Apply the pandas UDF for distributed scoring
df_scored = df_encoded.withColumn(
    "conversion_probability",
    predict_conversion_probability(*[F.col(c) for c in udf_input_cols])
)

print("Distributed inference complete. Sample predictions:")
(
    df_scored
    .select("contact_id", "label", "conversion_probability", "engagement_score", "fit_score")
    .orderBy(F.col("conversion_probability").desc())
    .show(10, truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Predictions at Scale
# MAGIC
# MAGIC Let's confirm the UDF-based predictions are consistent with the driver-side model.

# COMMAND ----------

# Distribution of predicted probabilities
df_scored.select("conversion_probability").describe().show()

# Bucket predictions and compare to actual conversion rate
df_scored.createOrReplaceTempView("scored_leads")

spark.sql("""
    SELECT
        CASE
            WHEN conversion_probability < 0.2 THEN '0.0 - 0.2'
            WHEN conversion_probability < 0.4 THEN '0.2 - 0.4'
            WHEN conversion_probability < 0.6 THEN '0.4 - 0.6'
            WHEN conversion_probability < 0.8 THEN '0.6 - 0.8'
            ELSE '0.8 - 1.0'
        END AS probability_bucket,
        COUNT(*) AS total_leads,
        SUM(CAST(label AS INT)) AS actual_conversions,
        ROUND(AVG(label), 3) AS actual_conversion_rate
    FROM scored_leads
    GROUP BY 1
    ORDER BY 1
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 4: Latest Features (Brief Intro)
# MAGIC
# MAGIC ## 4.1 — `pyspark.ml.connect` — Spark ML via Databricks Connect
# MAGIC
# MAGIC Starting with **DBR 15.0+**, the `pyspark.ml.connect` module allows you to run Spark ML
# MAGIC pipelines from your **local IDE** (VS Code, PyCharm) while the computation executes on
# MAGIC a remote Databricks cluster via Spark Connect.
# MAGIC
# MAGIC This is a game-changer for ML teams who want to:
# MAGIC - Develop and debug locally with familiar tools
# MAGIC - Run unit tests in CI/CD pipelines
# MAGIC - Use version control workflows (Git) natively
# MAGIC - Still leverage distributed Spark compute for training
# MAGIC

# COMMAND ----------

# NOTE: The code below illustrates the pattern but is designed to run
# from a LOCAL environment with Databricks Connect, not from a notebook.
# It is included here for educational purposes.

# --- Conceptual Example (run locally with databricks-connect installed) ---

# from databricks.connect import DatabricksSession
# from pyspark.ml.connect.classification import LogisticRegression as LR_Connect
# from pyspark.ml.connect.feature import VectorAssembler as VA_Connect
#
# # Establish remote Spark session
# spark_remote = DatabricksSession.builder.remote(
#     host="https://<workspace-url>",
#     token="<your-token>",
#     cluster_id="<cluster-id>"
# ).getOrCreate()
#
# # Load data from Unity Catalog (executes on the remote cluster)
# df = spark_remote.table("ankit_yadav.servicenow_training.gtm_lead_scores")
#
# # Build and train pipeline (distributed on the cluster)
# assembler = VA_Connect(inputCols=["engagement_score", "fit_score"], outputCol="features")
# lr = LR_Connect(featuresCol="features", labelCol="converted", maxIter=50)
# model = Pipeline(stages=[assembler, lr]).fit(df)
#
# # Predictions come back to local Python session
# predictions = model.transform(df)
# predictions.show(5)

print("pyspark.ml.connect — conceptual example shown above (commented out).")
print("This runs from a local IDE with Databricks Connect, not from a notebook.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 — `TorchDistributor` — Distributed Deep Learning
# MAGIC
# MAGIC For deep learning workloads (PyTorch), Databricks provides the **`TorchDistributor`** utility
# MAGIC that distributes training across multiple GPUs on a Spark cluster. It handles:
# MAGIC
# MAGIC - Multi-node, multi-GPU communication (via `torch.distributed`)
# MAGIC - Process spawning and coordination
# MAGIC - Fault tolerance and log collection
# MAGIC
# MAGIC This is ideal for:
# MAGIC - Fine-tuning large language models (LLMs) on domain-specific data
# MAGIC - Training custom neural networks on tabular data at scale
# MAGIC - Image classification and NLP tasks with large datasets
# MAGIC

# COMMAND ----------

# NOTE: Requires a GPU-enabled cluster (not serverless).
# This is a conceptual illustration of the TorchDistributor pattern.

# --- Conceptual Example ---

# from pyspark.ml.torch.distributor import TorchDistributor
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
#
# # Define a simple model
# class LeadScoringNet(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# # Define the training function (runs on EACH worker)
# def train_fn():
#     import torch
#     import torch.distributed as dist
#     from torch.nn.parallel import DistributedDataParallel as DDP
#
#     dist.init_process_group("nccl")  # NCCL backend for GPU
#     rank = dist.get_rank()
#     device = torch.device(f"cuda:{rank}")
#
#     model = LeadScoringNet(input_dim=27).to(device)
#     model = DDP(model, device_ids=[rank])
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.BCELoss()
#
#     # Load data (each rank gets a shard via DistributedSampler)
#     # ... data loading code ...
#
#     for epoch in range(10):
#         model.train()
#         # ... training loop ...
#
#     if rank == 0:
#         torch.save(model.state_dict(), "/dbfs/tmp/lead_scoring_model.pt")
#
#     dist.destroy_process_group()
#
# # Launch distributed training across the cluster
# distributor = TorchDistributor(
#     num_processes=4,        # Total GPU count across all workers
#     local_mode=False,       # False = distributed across workers
#     use_gpu=True,
# )
# distributor.run(train_fn)

print("TorchDistributor — conceptual example shown above (commented out).")
print("Requires a GPU-enabled cluster for execution.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Summary
# MAGIC
# MAGIC In this module, we covered:
# MAGIC
# MAGIC | Topic | Key Takeaway |
# MAGIC |-------|-------------|
# MAGIC | **Spark Architecture** | Driver coordinates, workers execute. Data is split into partitions for parallel processing. |
# MAGIC | **Feature Engineering** | Join and aggregate multiple tables using Spark DataFrames. Catalyst optimizes the execution plan. |
# MAGIC | **Spark ML Pipelines** | `StringIndexer` + `VectorAssembler` + `StandardScaler` + Classifier = reproducible, distributed pipeline. |
# MAGIC | **Model Evaluation** | AUC-ROC for ranking quality, F1 for class-imbalanced classification. |
# MAGIC | **Feature Importance** | Random Forest reveals which signals drive conversion (actionable for the business). |
# MAGIC | **Pandas on Spark** | Familiar pandas API backed by distributed Spark execution. Great for prototyping. |
# MAGIC | **Pandas UDFs** | The go-to pattern for distributed inference with sklearn models. Runs on workers, not driver. |
# MAGIC | **pyspark.ml.connect** | Train Spark ML models from your local IDE via Databricks Connect. |
# MAGIC | **TorchDistributor** | Distribute PyTorch training across multiple GPUs on a Spark cluster. |
# MAGIC
# MAGIC ## Next Module
# MAGIC
# MAGIC **Module 2** will cover MLflow Experiment Tracking, Model Registry, and Feature Store —
# MAGIC taking our trained models from experimentation to production-ready artifacts.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *Module 1 complete. Proceed to Module 2: MLflow & Model Management.*
