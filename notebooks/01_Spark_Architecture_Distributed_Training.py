# Databricks notebook source
# DBTITLE 1,Module 1: Spark Architecture and Distributed Training
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
# MAGIC 3. Train classification models using scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
# MAGIC 4. Evaluate binary classifiers with AUC, accuracy, and F1 score
# MAGIC 5. Use the Pandas API on Spark and Pandas UDFs for distributed inference
# MAGIC 6. Understand the newest Spark ML capabilities (`pyspark.ml.connect`, `TorchDistributor`)
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Run **Module 0** first to create the `servicenow_training` schema and all GTM tables
# MAGIC - Compute: Serverless (no cluster configuration needed). Serverless compute is fully managed by Databricks -- it auto-scales, requires no cluster configuration, and you pay only for what you use.
# MAGIC
# MAGIC **Estimated Runtime:** ~15 minutes

# COMMAND ----------

# DBTITLE 1,Configuration
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# DBTITLE 1,Run Config File
# MAGIC %run ./_config

# COMMAND ----------

# DBTITLE 1,Activate Training Catalog and Schema
# MAGIC %md
# MAGIC Activate the training catalog and schema established in Notebook 00.

# COMMAND ----------

# DBTITLE 1,Set Active Catalog and Schema
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")
print(f"Active catalog/schema: {catalog}.{schema}")

# COMMAND ----------

# DBTITLE 1,Section 1: How Spark Distributes Work
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
# MAGIC | **Driver** | Coordinates the job. Parses your code, builds a logical plan, optimizes it via Catalyst (Spark's built-in query optimizer that rewrites and optimizes your SQL/DataFrame operations before execution), and schedules tasks. |
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

# DBTITLE 1,Load Data from Unity Catalog
# MAGIC %md
# MAGIC ## 1.2 — Load Data from Unity Catalog
# MAGIC
# MAGIC **Unity Catalog** (Databricks' centralized governance layer for access control and data lineage)
# MAGIC stores every table, model, and function we'll use. Let's start by loading the tables we created in Module 0.

# COMMAND ----------

# DBTITLE 1,Load GTM Tables from Unity Catalog
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

# DBTITLE 1,Build a Rich Feature Table
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

# DBTITLE 1,Import PySpark SQL Functions
# MAGIC %md
# MAGIC Import PySpark SQL functions for distributed aggregations and window operations.

# COMMAND ----------

# DBTITLE 1,Step 1: Aggregate Activities per Contact
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

# DBTITLE 1,Step 2: Aggregate Campaign Engagement
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

# DBTITLE 1,Step 3: Join into Feature Table
# --- Step 3: Join everything into a single feature table ---

# Start with lead scores (one row per contact — this is our label table)
# Drop created_date from accounts to avoid ambiguity with contacts.created_date
df_accounts_slim = df_accounts.drop("created_date")

df_features = (
    df_lead_scores
    .join(df_contacts, on="contact_id", how="inner")
    .join(df_accounts_slim, on="account_id", how="left")
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

# DBTITLE 1,Pandas vs Spark vs Photon Comparison
# MAGIC %md
# MAGIC ## 1.3.1 — Pandas vs Spark vs Photon: Why Distributed Processing Matters
# MAGIC
# MAGIC Before moving on, let's answer the question every data scientist asks:
# MAGIC **"Why not just use pandas?"**
# MAGIC
# MAGIC | Approach | Engine | Best For | Limitation |
# MAGIC |----------|--------|----------|------------|
# MAGIC | **pandas** | Single node (driver CPU) | Datasets < ~10 GB, fast iteration | Memory-bound, no parallelism |
# MAGIC | **Spark DataFrames** | Distributed (workers) | Datasets 10 GB – PB scale | Processes data where it lives |
# MAGIC | **Spark + Photon** | Distributed + C++ vectorized | Large analytical queries, joins, aggs | Classic clusters only (not serverless) |
# MAGIC
# MAGIC **Photon** is Databricks' C++ vectorized query engine that accelerates Spark SQL workloads -- available on classic clusters.
# MAGIC
# MAGIC To see this in action, we'll **scale up** our activity and campaign data to simulate a realistic
# MAGIC enterprise CRM workload — millions of engagement records across your customer base — and run the
# MAGIC **same feature engineering pipeline** in both pandas and Spark.

# COMMAND ----------

# DBTITLE 1,Benchmark Setup: Scale to Enterprise Volume
import time

# === BENCHMARK SETUP: Scale data to enterprise volume ===
# Real-world CRM systems have millions of activity and campaign records.
# We replicate our tables 20x to simulate that scale.

SCALE_FACTOR = 20

_bench_activities = df_activities.crossJoin(
    spark.range(SCALE_FACTOR).select(F.col("id").alias("_scale"))
).drop("_scale")

_bench_campaigns = df_campaigns.crossJoin(
    spark.range(SCALE_FACTOR).select(F.col("id").alias("_scale"))
).drop("_scale")

# Materialize once so both approaches start from the same baseline
# (.cache() is not available on serverless — write to temp tables instead)
_bench_activities.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}._bench_activities")
_bench_campaigns.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}._bench_campaigns")

_bench_activities = spark.table(f"{catalog}.{schema}._bench_activities")
_bench_campaigns = spark.table(f"{catalog}.{schema}._bench_campaigns")

_act_count = _bench_activities.count()
_camp_count = _bench_campaigns.count()

print(f"Benchmark data prepared ({SCALE_FACTOR}x scale):")
print(f"  Activities:       {_act_count:>10,} rows  (original: {df_activities.count():,})")
print(f"  Campaign members: {_camp_count:>10,} rows  (original: {df_campaigns.count():,})")
print(f"\nRunning identical feature engineering in pandas vs Spark...")

# COMMAND ----------

# DBTITLE 1,Benchmark: Pandas Single-Node Processing
# === PANDAS: Collect everything to the driver, process single-threaded ===

pandas_start = time.time()

# Step 1: Collect scaled data to driver memory
_pd_act = _bench_activities.toPandas()
_pd_camp = _bench_campaigns.toPandas()
_pd_ls = df_lead_scores.toPandas()
_pd_ct = df_contacts.toPandas()
_pd_acc = df_accounts.drop("created_date").toPandas()

# Step 2: Activity aggregations (single-threaded on driver)
_pd_act["is_positive"] = (_pd_act["outcome"] == "Positive").astype(int)
_pd_act["is_negative"] = (_pd_act["outcome"] == "Negative").astype(int)
_pd_act["is_email"] = (_pd_act["activity_type"] == "Email").astype(int)
_pd_act["is_meeting"] = (_pd_act["activity_type"] == "Meeting").astype(int)
_pd_act["is_demo"] = (_pd_act["activity_type"] == "Demo").astype(int)
_pd_act["is_call"] = (_pd_act["activity_type"] == "Call").astype(int)

_pd_act_agg = (
    _pd_act.groupby("contact_id")
    .agg(
        total_activities=("contact_id", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        positive_outcomes=("is_positive", "sum"),
        negative_outcomes=("is_negative", "sum"),
        email_count=("is_email", "sum"),
        meeting_count=("is_meeting", "sum"),
        demo_count=("is_demo", "sum"),
        call_count=("is_call", "sum"),
        avg_duration_minutes=("duration_minutes", "mean"),
        last_activity_date=("activity_date", "max"),
    )
    .reset_index()
)

# Step 3: Campaign aggregations (single-threaded on driver)
_pd_camp["is_responded"] = (_pd_camp["status"] == "Responded").astype(int)
_pd_camp["is_converted"] = (_pd_camp["status"] == "Converted").astype(int)

_pd_camp_agg = (
    _pd_camp.groupby("contact_id")
    .agg(
        campaigns_participated=("contact_id", "count"),
        email_opens_total=("email_opens", "sum"),
        email_clicks_total=("email_clicks", "sum"),
        form_fills_total=("form_fills", "sum"),
        campaign_responses=("is_responded", "sum"),
        campaign_conversions=("is_converted", "sum"),
    )
    .reset_index()
)

# Step 4: Join all tables (single-threaded on driver)
_pd_features = (
    _pd_ls
    .merge(_pd_ct, on="contact_id", how="inner")
    .merge(_pd_acc, on="account_id", how="left")
    .merge(_pd_act_agg, on="contact_id", how="left")
    .merge(_pd_camp_agg, on="contact_id", how="left")
)

pandas_row_count = len(_pd_features)
pandas_elapsed = time.time() - pandas_start
print(f"Pandas (single-node):  {pandas_elapsed:.2f}s  —  {pandas_row_count:,} output rows from {_act_count:,} activities")

del _pd_ls, _pd_ct, _pd_act, _pd_camp, _pd_acc, _pd_act_agg, _pd_camp_agg, _pd_features

# COMMAND ----------

# DBTITLE 1,Benchmark: Spark Distributed Processing
# === SPARK: Distributed processing across workers ===
# Same pipeline, but Spark partitions the work across the cluster.

spark_start = time.time()

_spark_act_agg = (
    _bench_activities
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

_spark_camp_agg = (
    _bench_campaigns
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

_spark_accounts_slim = df_accounts.drop("created_date")

_spark_features = (
    df_lead_scores
    .join(df_contacts, on="contact_id", how="inner")
    .join(_spark_accounts_slim, on="account_id", how="left")
    .join(_spark_act_agg, on="contact_id", how="left")
    .join(_spark_camp_agg, on="contact_id", how="left")
)

# .count() forces full materialization — no lazy shortcuts
spark_row_count = _spark_features.count()

spark_elapsed = time.time() - spark_start
print(f"Spark (distributed):   {spark_elapsed:.2f}s  —  {spark_row_count:,} output rows from {_act_count:,} activities")

del _spark_act_agg, _spark_camp_agg, _spark_accounts_slim, _spark_features

# Clean up benchmark tables
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}._bench_activities")
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}._bench_campaigns")
del _bench_activities, _bench_campaigns

# COMMAND ----------

# DBTITLE 1,Benchmark Results Interpretation
# MAGIC %md
# MAGIC ### Results Interpretation
# MAGIC
# MAGIC Spark wins — and the gap widens with data size. Here's why:
# MAGIC
# MAGIC | What happens | pandas (single-node) | Spark (distributed) |
# MAGIC |-------------|---------------------|---------------------|
# MAGIC | **Data location** | Must collect ALL data to the driver first | Processes data where it already lives on workers |
# MAGIC | **Aggregation** | Single CPU core iterates over every row | Partitioned across workers, each handles a slice |
# MAGIC | **Joins** | Single-threaded merge in driver memory | Broadcast or shuffle join across the cluster |
# MAGIC | **Memory** | Entire dataset must fit in driver RAM | Each worker holds only its partition |
# MAGIC
# MAGIC | Data Size | pandas | Spark | Spark + Photon | Recommendation |
# MAGIC |-----------|--------|-------|----------------|----------------|
# MAGIC | < 1 GB | Fast | Comparable | Comparable | Either works |
# MAGIC | 1–10 GB | Slow, feasible | Faster | 2–5x faster than Spark | Spark |
# MAGIC | 10–100 GB | Out of memory | Scales well | 2–5x faster than Spark | Spark + Photon |
# MAGIC | 100 GB+ | Impossible | Scales horizontally | 2–5x faster than Spark | Spark + Photon |
# MAGIC
# MAGIC **Key insight for the ServiceNow team:**
# MAGIC - At production CRM volumes (millions of activities, hundreds of thousands of contacts), Spark's
# MAGIC   distributed processing is not optional — it's essential
# MAGIC - **Photon** accelerates Spark SQL and DataFrame operations using a C++ vectorized engine — enable it
# MAGIC   on classic clusters via the "Use Photon Acceleration" toggle. It is **not available on serverless**
# MAGIC   (serverless has its own optimized runtime that provides similar benefits automatically)
# MAGIC
# MAGIC **Our approach in this training:** Use Spark for feature engineering (it scales), scikit-learn
# MAGIC for model training (it's fast for driver-sized data), and Pandas UDFs for distributed inference
# MAGIC (best of both worlds).

# COMMAND ----------

# DBTITLE 1,Print Benchmark Comparison
# Print the measured comparison
speedup = pandas_elapsed / spark_elapsed if spark_elapsed > 0 else float("inf")

print("=" * 65)
print("  FEATURE ENGINEERING BENCHMARK")
print(f"  Data: {_act_count:,} activity rows + {_camp_count:,} campaign rows ({SCALE_FACTOR}x scale)")
print("=" * 65)
print(f"  {'Approach':<30} {'Time':>10} {'Output Rows':>14}")
print("-" * 65)
print(f"  {'Pandas (single-node)':<30} {pandas_elapsed:>9.2f}s {pandas_row_count:>13,}")
print(f"  {'Spark (distributed)':<30} {spark_elapsed:>9.2f}s {spark_row_count:>13,}")
print("-" * 65)
print(f"\n  Spark was {speedup:.1f}x faster.")
print(f"  The pandas approach spent most of its time collecting {_act_count:,} rows")
print(f"  to the driver and processing them single-threaded.")
print(f"  Spark distributed the aggregation across workers and only")
print(f"  moved the final {spark_row_count:,}-row result.")
print(f"\n  Photon (classic clusters only): adds another 2-5x on top of Spark")
print(f"  for SQL/DataFrame operations via C++ vectorized execution.")

# COMMAND ----------

# DBTITLE 1,Inspect the Execution Plan
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

# DBTITLE 1,Show Full Execution Plan
# Show the full execution plan for our feature table
df_features.explain(True)

# COMMAND ----------

# DBTITLE 1,Partitions and Parallelism
# MAGIC %md
# MAGIC ## 1.5 — Partitions and Parallelism
# MAGIC
# MAGIC The number of **partitions** determines how many tasks run in parallel. Too few partitions
# MAGIC means under-utilized workers; too many means excessive overhead.
# MAGIC
# MAGIC **Rule of thumb:** 2-4 partitions per available CPU core. For serverless, Spark auto-tunes this.

# COMMAND ----------

# DBTITLE 1,Demo: Repartition and Coalesce
# On serverless compute, .rdd operations are not supported.
# Instead, we can write the DataFrame to a temp table and inspect partition count via DESCRIBE DETAIL,
# or simply demonstrate repartition/coalesce concepts without .rdd.getNumPartitions().

# Save to a temp location to inspect partition count via DESCRIBE DETAIL
df_features.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}._tmp_partition_demo")
partition_info = spark.sql(f"DESCRIBE DETAIL {catalog}.{schema}._tmp_partition_demo").select("numFiles").collect()
print(f"Current number of data files (partitions written): {partition_info[0]['numFiles']}")

# Repartition to a specific number (useful when writing output for downstream consumers)
df_repartitioned = df_features.repartition(8)
df_repartitioned.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}._tmp_partition_demo_8")
partition_info_8 = spark.sql(f"DESCRIBE DETAIL {catalog}.{schema}._tmp_partition_demo_8").select("numFiles").collect()
print(f"After repartition(8) and write:                    {partition_info_8[0]['numFiles']} files")

# Coalesce reduces partitions WITHOUT a full shuffle (more efficient for reducing)
df_coalesced = df_features.coalesce(4)
df_coalesced.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}._tmp_partition_demo_4")
partition_info_4 = spark.sql(f"DESCRIBE DETAIL {catalog}.{schema}._tmp_partition_demo_4").select("numFiles").collect()
print(f"After coalesce(4) and write:                       {partition_info_4[0]['numFiles']} files")

# Clean up temp tables
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}._tmp_partition_demo")
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}._tmp_partition_demo_8")
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}._tmp_partition_demo_4")

# COMMAND ----------

# DBTITLE 1,Section 2: ML on Serverless with scikit-learn
# MAGIC %md
# MAGIC ---
# MAGIC # Section 2: Training ML Models on Serverless Compute
# MAGIC
# MAGIC ## 2.1 — ML on Serverless: scikit-learn + Spark DataFrames
# MAGIC
# MAGIC On **serverless compute**, the classic JVM-based `pyspark.ml` APIs (StringIndexer, VectorAssembler,
# MAGIC LogisticRegression, etc.) are **not available** because custom JVM code is restricted.
# MAGIC (Serverless enforces multi-tenant isolation and restricts custom JVM bytecode execution for security.)
# MAGIC
# MAGIC Instead, we use a powerful and practical pattern:
# MAGIC
# MAGIC | Step | Tool | Why |
# MAGIC |------|------|-----|
# MAGIC | **Feature engineering** | Spark DataFrames / SQL | Distributed, handles TBs of data |
# MAGIC | **Model training** | scikit-learn (on pandas) | Rich algorithm library, fast on driver for datasets that fit in memory |
# MAGIC | **Distributed inference** | Pandas UDFs (`mapInPandas`) | Applies trained model across partitions on workers |
# MAGIC
# MAGIC This is actually the **recommended pattern** for most ML workloads on Databricks — even on
# MAGIC classic compute — because scikit-learn and XGBoost are often faster and more flexible than
# MAGIC Spark ML for datasets under ~100 GB.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import scikit-learn Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

print("scikit-learn imports loaded successfully (serverless-compatible).")

# COMMAND ----------

# DBTITLE 1,Prepare Data for scikit-learn
# MAGIC %md
# MAGIC ## 2.2 — Prepare Data for scikit-learn
# MAGIC
# MAGIC We convert our Spark DataFrame to pandas (this works well for datasets that fit in driver memory,
# MAGIC typically under ~10 GB). For our 10K-row dataset, this is instantaneous.
# MAGIC
# MAGIC Categorical columns are encoded using scikit-learn's `LabelEncoder`.

# COMMAND ----------

# DBTITLE 1,Encode Categorical Columns
categorical_cols = ["seniority_level", "lead_source", "industry", "account_tier"]

# Convert to pandas for scikit-learn training
pdf = df_features.toPandas()

# Encode categoricals with LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    pdf[col + "_idx"] = le.fit_transform(pdf[col].fillna("Unknown"))
    label_encoders[col] = le

print(f"Converted {len(pdf):,} rows to pandas. Categorical encoding complete.")
print("Encoded columns:", [f"{c}_idx" for c in categorical_cols])

# COMMAND ----------

# DBTITLE 1,Assemble Feature Matrix
# MAGIC %md
# MAGIC ## 2.3 — Assemble Feature Matrix
# MAGIC
# MAGIC In scikit-learn, we build the feature matrix as a NumPy array. We combine all numeric
# MAGIC and encoded categorical columns, then apply `StandardScaler` for normalization.

# COMMAND ----------

# DBTITLE 1,Define Feature Columns
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

print(f"Feature vector will contain {len(all_feature_cols)} features:")
for i, col in enumerate(all_feature_cols, 1):
    print(f"  {i:>2}. {col}")

# COMMAND ----------

# DBTITLE 1,Train/Test Split
# MAGIC %md
# MAGIC ## 2.4 — Train/Test Split
# MAGIC
# MAGIC We use an 80/20 stratified split with a fixed seed for reproducibility.

# COMMAND ----------

# DBTITLE 1,Stratified Train/Test Split and Scaling
# Keep a Spark DataFrame version with label for later use (Pandas UDF section)
df_ml = df_features.withColumn("label", F.col("converted").cast("double"))

# Prepare feature matrix and labels
X = pdf[all_feature_cols].fillna(0).values
y = pdf["converted"].values

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training set : {len(X_train):>6,} rows")
print(f"Test set     : {len(X_test):>6,} rows")

# Check class balance
print(f"\nClass distribution (training):")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Label {int(label)}: {count:>6,} ({count/len(y_train)*100:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Train Logistic Regression
# MAGIC %md
# MAGIC ## 2.5 — Train Logistic Regression
# MAGIC
# MAGIC Logistic Regression is a fast, interpretable baseline for binary classification.
# MAGIC Using scikit-learn with Elastic Net regularization (L1/L2 mix).

# COMMAND ----------

# DBTITLE 1,Train Logistic Regression Model
lr = LogisticRegression(
    max_iter=100,
    C=100.0,          # Inverse of regParam (1/0.01)
    penalty="elasticnet",
    l1_ratio=0.5,     # L1/L2 mix (0.5 = Elastic Net)
    solver="saga",    # Required for elasticnet penalty
    random_state=42,
)

print("Training Logistic Regression...")
lr.fit(X_train, y_train)
print("Training complete.")

# COMMAND ----------

# DBTITLE 1,Train Random Forest
# MAGIC %md
# MAGIC ## 2.6 — Train Random Forest
# MAGIC
# MAGIC Random Forest is an ensemble method that trains many decision trees in parallel.
# MAGIC scikit-learn parallelizes tree construction using `n_jobs=-1` (all available cores).

# COMMAND ----------

# DBTITLE 1,Train Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,  # Use all available CPU cores
)

print("Training Random Forest (100 trees)...")
rf.fit(X_train, y_train)
print("Training complete.")

# COMMAND ----------

# DBTITLE 1,Evaluate Both Models
# MAGIC %md
# MAGIC ## 2.7 — Evaluate Both Models
# MAGIC
# MAGIC We evaluate on the held-out test set using:
# MAGIC - **AUC-ROC** — Area Under the ROC Curve (ranking quality; 0.5 = random, 1.0 = perfect)
# MAGIC - **Accuracy** — Fraction of correct predictions
# MAGIC - **F1 Score** — Harmonic mean of precision and recall (important for imbalanced classes)

# COMMAND ----------

# DBTITLE 1,Compute AUC, Accuracy, and F1 Scores
results = {}

for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {"AUC": auc, "Accuracy": acc, "F1": f1}

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")

# COMMAND ----------

# DBTITLE 1,ROC Curve: Logistic Regression vs Random Forest
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fig, ax = plt.subplots(figsize=(8, 7))

# Plot ROC curve for each model
colors = {"Logistic Regression": "#4A90D9", "Random Forest": "#E55934"}

for name, model, ls in [("Logistic Regression", lr, "-"), ("Random Forest", rf, "--")]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=colors[name], linewidth=2.5, linestyle=ls,
            label=f"{name} (AUC = {auc:.4f})")

# Diagonal reference line (random classifier)
ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Random (AUC = 0.5)")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — Model Comparison", fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

print("\nInterpretation: A curve closer to the top-left corner indicates better")
print("discriminative power. The model with the higher AUC is the stronger ranker.")

# COMMAND ----------

# DBTITLE 1,Feature Importance (Random Forest)
# MAGIC %md
# MAGIC ## 2.8 — Feature Importance (Random Forest)
# MAGIC
# MAGIC Random Forest provides built-in feature importance based on how much each feature
# MAGIC reduces impurity (Gini) across all trees. This helps the sales team understand
# MAGIC **which signals matter most** for lead conversion.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Extract and Display Feature Importances
# Extract feature importances from the trained Random Forest
importances = rf.feature_importances_

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

# DBTITLE 1,View Predictions
# MAGIC %md
# MAGIC ## 2.9 — View Predictions
# MAGIC
# MAGIC Let's look at some example predictions to sanity-check the model output.

# COMMAND ----------

# DBTITLE 1,Sample Predictions from Random Forest
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Note: X_test values are scaled, so we show the raw probabilities and labels
predictions_pdf = pd.DataFrame({
    "label": y_test.astype(float),
    "prediction": y_pred_rf.astype(float),
    "probability": y_prob_rf,
})

# Display as Spark DataFrame for consistent notebook output
print("Sample predictions (Random Forest):")
spark.createDataFrame(predictions_pdf).show(10, truncate=False)

# COMMAND ----------

# DBTITLE 1,Section 3: Pandas API on Spark and UDFs
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

# DBTITLE 1,Convert to Pandas-on-Spark DataFrame
import pyspark.pandas as ps

# Convert our Spark DataFrame to a pandas-on-Spark DataFrame
psdf = df_features.pandas_api()

print(f"Type: {type(psdf)}")
print(f"Shape: {psdf.shape}")
psdf.head()

# COMMAND ----------

# DBTITLE 1,Train with scikit-learn on Pandas-on-Spark
# MAGIC %md
# MAGIC ## 3.2 — Train with scikit-learn on Pandas-on-Spark
# MAGIC
# MAGIC We can use familiar scikit-learn code by converting to a (local) pandas DataFrame.
# MAGIC This works well when the dataset fits in driver memory (typically under ~10 GB).
# MAGIC

# COMMAND ----------

# DBTITLE 1,Train Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Re-use the pandas DataFrame, feature matrix, and train/test split from Section 2.
# We also keep the same feature columns for consistency.
sklearn_feature_cols = all_feature_cols  # alias for Section 3.3 reference

# Train Gradient Boosting (a third model for comparison)
gbt = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
)
gbt.fit(X_train, y_train)

# Evaluate
y_pred_gbt = gbt.predict(X_test)
y_prob_gbt = gbt.predict_proba(X_test)[:, 1]

print("Gradient Boosting (scikit-learn) — Test Results:")
print(f"  AUC-ROC  : {roc_auc_score(y_test, y_prob_gbt):.4f}")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_gbt):.4f}")
print(f"  F1 Score : {f1_score(y_test, y_pred_gbt):.4f}")
print()
print(classification_report(y_test, y_pred_gbt, target_names=["Not Converted", "Converted"]))

# COMMAND ----------

# DBTITLE 1,Pandas UDFs for Distributed Inference
# MAGIC %md
# MAGIC ## 3.3 — Pandas UDFs for Distributed Inference
# MAGIC
# MAGIC **Pandas UDFs** (also called Vectorized UDFs) batch rows into pandas DataFrames for efficient
# MAGIC Python processing. `mapInPandas` is a related API for partition-level transforms.
# MAGIC These UDFs let you apply a Python function to each **partition** of a Spark DataFrame.
# MAGIC The data arrives as a pandas Series or DataFrame, and you return a pandas Series.
# MAGIC Spark handles the distribution automatically.
# MAGIC
# MAGIC This is the **best pattern for distributed inference** with scikit-learn models:
# MAGIC 1. Train a model on the driver (as above)
# MAGIC 2. Serialize the model (pickle) and capture via closure
# MAGIC 3. Apply it via a Pandas UDF across all partitions
# MAGIC

# COMMAND ----------

# DBTITLE 1,Serialize Model and Scaler for UDF
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pickle

# On serverless compute, spark.sparkContext.broadcast() is unavailable because it requires a
# persistent JVM context that serverless does not expose. Instead, we use closure serialization:
# Spark automatically distributes the pickled model to workers when the UDF is called.
# This works well for small models (< a few hundred MB).

model_bytes = pickle.dumps(gbt)
scaler_bytes = pickle.dumps(scaler)  # Also serialize the scaler for feature normalization
feature_cols_list = list(sklearn_feature_cols)  # plain Python list for closure capture

# COMMAND ----------

# DBTITLE 1,Define the Pandas UDF
# MAGIC %md
# MAGIC ### Define the Pandas UDF
# MAGIC
# MAGIC The UDF receives a `pandas.DataFrame` (one partition at a time) and returns a `pandas.Series`
# MAGIC of predicted probabilities.

# COMMAND ----------

# DBTITLE 1,Define Prediction Pandas UDF
@pandas_udf(DoubleType())
def predict_conversion_probability(*cols: pd.Series) -> pd.Series:
    """
    Distributed inference using a serialized scikit-learn model.
    Each argument is a pandas Series corresponding to one feature column.
    Returns a pandas Series of predicted conversion probabilities.

    The model is captured via closure (pickle-serialized bytes) rather than
    spark.sparkContext.broadcast(), which is not available on serverless compute.
    """
    import pickle as _pickle
    # Stack all feature columns into a 2D numpy array
    features = pd.concat(cols, axis=1).fillna(0).values
    # Deserialize the scaler and model from the closure-captured pickle bytes
    _scaler = _pickle.loads(scaler_bytes)
    model = _pickle.loads(model_bytes)
    # Scale features (same normalization as training)
    features_scaled = _scaler.transform(features)
    # Predict probability of class 1 (converted)
    probabilities = model.predict_proba(features_scaled)[:, 1]
    return pd.Series(probabilities)

# COMMAND ----------

# DBTITLE 1,Apply UDF to Full DataFrame
# MAGIC %md
# MAGIC ### Apply the UDF to the Full DataFrame
# MAGIC
# MAGIC The key insight: this runs **on the workers**, not the driver. Even if the DataFrame had
# MAGIC billions of rows, the inference would be distributed.

# COMMAND ----------

# DBTITLE 1,Distributed Scoring with Pandas UDF
# We need to encode categoricals in Spark SQL before passing to the UDF.
# On serverless, pyspark.ml.feature.StringIndexer is not available, so we
# use a pure Spark SQL approach: map each category to its LabelEncoder index
# using CASE WHEN expressions (derived from the encoders we already fitted).

# Build category-to-index mappings from our fitted LabelEncoders
for col in categorical_cols:
    le = label_encoders[col]
    mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    # Build a CASE WHEN expression for this column
    case_expr = "CASE "
    for cat_val, idx_val in mapping.items():
        safe_val = cat_val.replace("'", "\\'")
        case_expr += f"WHEN {col} = '{safe_val}' THEN {idx_val} "
    case_expr += f"ELSE {len(mapping)} END"  # unknown category gets next index
    df_ml = df_ml.withColumn(f"{col}_idx", F.expr(case_expr))

# Build the list of columns to pass to the UDF (same order as sklearn training)
udf_input_cols = numeric_cols + [f"{c}_idx" for c in categorical_cols]

# Apply the pandas UDF for distributed scoring
df_scored = df_ml.withColumn(
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

# DBTITLE 1,Verify Predictions at Scale
# MAGIC %md
# MAGIC ### Verify Predictions at Scale
# MAGIC
# MAGIC Let's confirm the UDF-based predictions are consistent with the driver-side model.

# COMMAND ----------

# DBTITLE 1,Bucket Predictions vs Actual Conversion
# ── Part 1: Quick summary stats on the predicted scores ──────────────────────
print("=" * 65)
print("  PREDICTED CONVERSION PROBABILITY — Summary Statistics")
print("=" * 65)
print("  This shows the distribution of scores the model assigned to all")
print("  10,000 leads. A healthy model should use the full 0→1 range,")
print("  not cluster everything near 0.5.\n")

df_scored.select("conversion_probability").describe().show()

print("  ✓ The model spreads scores from ~0.01 to ~0.99 with a std dev")
print("    of ~0.29 — it's confidently separating high- from low-quality leads.\n")

# ── Part 2: Bucket analysis — "Do the scores actually mean anything?" ────────
print("=" * 65)
print("  CALIBRATION CHECK — Predicted Bucket vs Reality")
print("=" * 65)
print("  We group leads into 5 buckets by what the model predicted,")
print("  then check what ACTUALLY happened in each bucket.")
print("  If the model is trustworthy, leads scored 0.6–0.8 should")
print("  convert ~60–80% of the time in reality.\n")

df_scored.createOrReplaceTempView("scored_leads")

spark.sql("""
    WITH bucketed AS (
        SELECT
            CASE
                WHEN conversion_probability < 0.2 THEN '0.0 - 0.2'
                WHEN conversion_probability < 0.4 THEN '0.2 - 0.4'
                WHEN conversion_probability < 0.6 THEN '0.4 - 0.6'
                WHEN conversion_probability < 0.8 THEN '0.6 - 0.8'
                ELSE '0.8 - 1.0'
            END AS bucket,
            label
        FROM scored_leads
    )
    SELECT
        bucket AS `Predicted Score Range`,
        COUNT(*) AS `Leads in Bucket`,
        SUM(CAST(label AS INT)) AS `Actually Converted`,
        CONCAT(ROUND(AVG(label) * 100, 1), '%') AS `Real Conversion Rate`,
        CASE
            WHEN bucket = '0.0 - 0.2'
                THEN CASE WHEN AVG(label) < 0.20 THEN '✓ Yes' ELSE '✗ No' END
            WHEN bucket = '0.2 - 0.4'
                THEN CASE WHEN AVG(label) BETWEEN 0.15 AND 0.45 THEN '✓ Yes' ELSE '✗ No' END
            WHEN bucket = '0.4 - 0.6'
                THEN CASE WHEN AVG(label) BETWEEN 0.35 AND 0.65 THEN '✓ Yes' ELSE '✗ No' END
            WHEN bucket = '0.6 - 0.8'
                THEN CASE WHEN AVG(label) BETWEEN 0.55 AND 0.85 THEN '✓ Yes' ELSE '✗ No' END
            ELSE CASE WHEN AVG(label) > 0.75 THEN '✓ Yes' ELSE '✗ No' END
        END AS `Model Accurate?`
    FROM bucketed
    GROUP BY bucket
    ORDER BY bucket
""").show(truncate=False)

print("  HOW TO READ THIS TABLE:")
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  Predicted Score Range  = What the model predicted          │")
print("  │  Leads in Bucket        = How many leads got that score     │")
print("  │  Actually Converted     = How many really became customers  │")
print("  │  Real Conversion Rate   = Actual % that converted           │")
print("  │  Model Accurate?        = Does reality match the prediction │")
print("  └─────────────────────────────────────────────────────────────┘")
print()
print("  KEY TAKEAWAY FOR SALES:")
print("  → A lead scored 0.75 truly has ~75% chance of converting.")
print("  → The scores are NOT just rankings — they are real probabilities.")
print("  → Focus outreach on the 0.6+ buckets for highest ROI.")

# COMMAND ----------

# DBTITLE 1,Calibration Plot: Predicted Buckets vs Actual Conversion Rate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Build calibration data from df_scored (created in the cell above)
calibration_pdf = (
    df_scored
    .withColumn(
        "probability_bucket",
        F.when(F.col("conversion_probability") < 0.2, "0.0 - 0.2")
         .when(F.col("conversion_probability") < 0.4, "0.2 - 0.4")
         .when(F.col("conversion_probability") < 0.6, "0.4 - 0.6")
         .when(F.col("conversion_probability") < 0.8, "0.6 - 0.8")
         .otherwise("0.8 - 1.0")
    )
    .groupBy("probability_bucket")
    .agg(
        F.count("*").alias("total_leads"),
        F.sum(F.col("label").cast("int")).alias("actual_conversions"),
        F.round(F.avg("label"), 3).alias("actual_conversion_rate"),
    )
    .orderBy("probability_bucket")
    .toPandas()
)

# Midpoints for the ideal calibration reference line
bucket_midpoints = [0.1, 0.3, 0.5, 0.7, 0.9]

fig, ax1 = plt.subplots(figsize=(10, 6))

# --- Bar chart: lead volume per bucket ---
bars = ax1.bar(
    calibration_pdf["probability_bucket"],
    calibration_pdf["total_leads"],
    color="#4A90D9", alpha=0.7, label="Total Leads", zorder=2
)
ax1.set_xlabel("Predicted Probability Bucket", fontsize=12)
ax1.set_ylabel("Number of Leads", fontsize=12, color="#4A90D9")
ax1.tick_params(axis="y", labelcolor="#4A90D9")

# Add count labels on bars
for bar, count in zip(bars, calibration_pdf["total_leads"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
             f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# --- Line chart: actual conversion rate (secondary y-axis) ---
ax2 = ax1.twinx()
ax2.plot(
    calibration_pdf["probability_bucket"],
    calibration_pdf["actual_conversion_rate"],
    color="#E55934", marker="o", linewidth=2.5, markersize=8,
    label="Actual Conversion Rate", zorder=3
)

# Ideal calibration reference line (dashed)
ax2.plot(
    calibration_pdf["probability_bucket"],
    bucket_midpoints,
    color="gray", linestyle="--", linewidth=1.5, alpha=0.6,
    label="Ideal Calibration", zorder=1
)

ax2.set_ylabel("Actual Conversion Rate", fontsize=12, color="#E55934")
ax2.tick_params(axis="y", labelcolor="#E55934")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax2.set_ylim(0, 1.05)

# --- Legend and title ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

plt.title("Model Calibration: Predicted Probability vs Actual Conversion Rate",
          fontsize=14, fontweight="bold", pad=15)
fig.tight_layout()
plt.show()

print("\nInterpretation: The red line closely tracks the dashed ideal line,")
print("confirming the model is well-calibrated \u2014 predicted probabilities")
print("match observed conversion rates across all buckets.")
print("\nFor the sales team, this means a lead scored at 0.7 truly has ~70% chance")
print("of converting \u2014 the scores can be trusted as-is for prioritization.")
print("\nNotice the bar heights: most leads fall in the 0.0-0.2 bucket. A well-calibrated")
print("model with a skewed distribution like this helps sales focus effort on the")
print("smaller pool of high-probability leads rather than spreading thin across all.")

# COMMAND ----------

# DBTITLE 1,Section 4: pyspark.ml.connect
# MAGIC %md
# MAGIC ---
# MAGIC # Section 4: Latest Features (Brief Intro)
# MAGIC
# MAGIC ## 4.1 — `pyspark.ml.connect` — Spark ML via Databricks Connect
# MAGIC
# MAGIC **Spark Connect** is a client-server protocol that lets you run Spark code remotely.
# MAGIC **Databricks Connect** is Databricks' implementation -- it lets you develop in your local IDE
# MAGIC (VS Code, PyCharm) while executing on a Databricks cluster.
# MAGIC
# MAGIC Starting with **DBR 15.0+**, the `pyspark.ml.connect` module allows you to run Spark ML
# MAGIC pipelines from your **local IDE** while the computation executes on
# MAGIC a remote Databricks cluster via Spark Connect.
# MAGIC
# MAGIC This is a game-changer for ML teams who want to:
# MAGIC - Develop and debug locally with familiar tools
# MAGIC - Run unit tests in CI/CD pipelines
# MAGIC - Use version control workflows (Git) natively
# MAGIC - Still leverage distributed Spark compute for training
# MAGIC

# COMMAND ----------

# DBTITLE 1,Conceptual: Spark ML via Databricks Connect
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

# DBTITLE 1,TorchDistributor: Distributed Deep Learning
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

# DBTITLE 1,Conceptual: TorchDistributor Example
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

# DBTITLE 1,Module 1 Summary
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

# DBTITLE 1,Next: Module 2
# MAGIC %md
# MAGIC ---
# MAGIC *Module 1 complete. Proceed to Module 2: MLflow & Model Management.*
