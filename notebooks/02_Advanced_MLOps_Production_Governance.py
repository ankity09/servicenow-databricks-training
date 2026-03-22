# Databricks notebook source

# MAGIC %md
# MAGIC # Module 2: Advanced MLOps & Production Governance
# MAGIC
# MAGIC **ServiceNow x Databricks Training Workshop**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What You Will Learn
# MAGIC
# MAGIC | Section | Topic | Key Concepts |
# MAGIC |---------|-------|-------------|
# MAGIC | 1 | Hyperparameter Tuning at Scale | Hyperopt + Trials, Optuna, hyperparameter search |
# MAGIC | 2 | MLflow & Unity Catalog Model Registry | Experiment tracking, model versioning, aliases |
# MAGIC | 3 | Model Serving Deployment | Real-time endpoints, auto-scaling, A/B testing |
# MAGIC | 4 | Inference Tables & Monitoring | Drift detection, latency monitoring, alerting |
# MAGIC | 5 | Workflows & DABs Overview | CI/CD, orchestration, production pipelines |
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - Completed **Module 1** (notebook 01) which trained Spark ML models for lead scoring
# MAGIC - GTM data tables created in **Module 0** (notebook 00)
# MAGIC
# MAGIC ### Compute
# MAGIC This notebook is designed for **Serverless** compute. No cluster configuration needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Configuration

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC Verify that the shared configuration loaded correctly and activate our training namespace.

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")
print(f"Catalog: {catalog} | Schema: {schema} | User: {username}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Libraries
# MAGIC
# MAGIC We need a few libraries that may not be pre-installed on serverless compute.

# COMMAND ----------

# MAGIC %pip install xgboost lightgbm optuna hyperopt mlflow scikit-learn matplotlib seaborn --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC After installing libraries (which restarts the Python process), reload the shared configuration.

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")
print(f"Config reloaded: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 1: Hyperparameter Tuning at Scale
# MAGIC
# MAGIC In Notebook 01 we used scikit-learn on the Spark driver. Here we step up to **XGBoost** -- a gradient-boosted tree algorithm that is the industry standard for tabular ML. We pair it with **Hyperopt** for intelligent hyperparameter search.
# MAGIC
# MAGIC Now we level up:
# MAGIC
# MAGIC 1. **Rebuild the feature table** using the same joins from notebook 01
# MAGIC 2. **Train XGBoost** with scikit-learn API (runs on the driver, but tuning is distributed)
# MAGIC 3. **Use Hyperopt + Trials** to run hyperparameter search (serverless-compatible)
# MAGIC 4. **Compare with Optuna** as an alternative tuning framework
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Rebuild the Feature Table
# MAGIC
# MAGIC We replicate the feature engineering from notebook 01: join accounts, contacts,
# MAGIC opportunities, activities, and lead scores to create a rich feature set for
# MAGIC predicting the `converted` target column.

# COMMAND ----------

from pyspark.sql import functions as F

# ── Load raw tables ──────────────────────────────────────────────
accounts     = spark.table(f"{catalog}.{schema}.gtm_accounts")
contacts     = spark.table(f"{catalog}.{schema}.gtm_contacts")
opportunities = spark.table(f"{catalog}.{schema}.gtm_opportunities")
activities   = spark.table(f"{catalog}.{schema}.gtm_activities")
lead_scores  = spark.table(f"{catalog}.{schema}.gtm_lead_scores")

# COMMAND ----------

# MAGIC %md
# MAGIC Validate that all training tables from Notebook 00 loaded successfully.

# COMMAND ----------

print("Table row counts:")
print(f"  accounts      : {accounts.count():,}")
print(f"  contacts      : {contacts.count():,}")
print(f"  opportunities : {opportunities.count():,}")
print(f"  activities    : {activities.count():,}")
print(f"  lead_scores   : {lead_scores.count():,}")

# COMMAND ----------

# ── Aggregate activity features per contact ──────────────────────
activity_features = (
    activities
    .groupBy("contact_id")
    .agg(
        F.count("*").alias("total_activities"),
        F.countDistinct("activity_type").alias("distinct_activity_types"),
        F.sum(F.when(F.col("activity_type") == "Email", 1).otherwise(0)).alias("email_count"),
        F.sum(F.when(F.col("activity_type") == "Call", 1).otherwise(0)).alias("call_count"),
        F.sum(F.when(F.col("activity_type") == "Meeting", 1).otherwise(0)).alias("meeting_count"),
        F.sum(F.when(F.col("activity_type") == "Demo", 1).otherwise(0)).alias("demo_count"),
        F.sum(F.when(F.col("activity_type") == "Webinar", 1).otherwise(0)).alias("webinar_count"),
    )
)

print(f"Activity features: {activity_features.count():,} contacts with activity data")

# COMMAND ----------

# ── Aggregate opportunity features per account ────────────────────
opp_features = (
    opportunities
    .groupBy("account_id")
    .agg(
        F.count("*").alias("num_opportunities"),
        F.avg("amount").alias("avg_opp_amount"),
        F.max("amount").alias("max_opp_amount"),
        F.sum("amount").alias("total_opp_amount"),
        F.countDistinct("stage").alias("distinct_stages"),
        F.sum(F.when(F.col("stage") == "Closed Won", 1).otherwise(0)).alias("closed_won_count"),
        F.sum(F.when(F.col("stage") == "Closed Lost", 1).otherwise(0)).alias("closed_lost_count"),
    )
)

print(f"Opportunity features: {opp_features.count():,} accounts with opportunity data")

# COMMAND ----------

# ── Join everything into a single feature table ───────────────────
feature_df = (
    lead_scores
    .join(contacts.select("contact_id", "account_id", "lead_source", "title", "department"),
          on="contact_id", how="left")
    .join(accounts.select("account_id", "industry", "employee_count", "annual_revenue"),
          on="account_id", how="left")
    .join(activity_features, on="contact_id", how="left")
    .join(opp_features, on="account_id", how="left")
    .fillna(0)
)

print(f"Feature table rows: {feature_df.count():,}")
print(f"Feature table cols: {len(feature_df.columns)}")
feature_df.printSchema()

# COMMAND ----------

display(feature_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Prepare Data for Scikit-Learn / XGBoost
# MAGIC
# MAGIC We convert the Spark DataFrame to Pandas and encode categorical columns.
# MAGIC For large datasets you would use Spark ML or pandas-on-Spark, but our GTM
# MAGIC dataset fits comfortably in driver memory.

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)

# Convert to Pandas
pdf = feature_df.toPandas()

# ── Define target and feature columns ────────────────────────────
target_col = "converted"

# Identify numeric and categorical columns
exclude_cols = [target_col, "contact_id", "account_id", "score_id"]
categorical_cols = ["lead_source", "title", "department", "industry"]
numeric_cols = [c for c in pdf.columns
                if c not in exclude_cols
                and c not in categorical_cols
                and pd.api.types.is_numeric_dtype(pdf[c])]

print(f"Target          : {target_col}")
print(f"Numeric features: {len(numeric_cols)}")
print(f"  {numeric_cols}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"  {categorical_cols}")

# COMMAND ----------

# ── Encode categorical features ──────────────────────────────────
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    pdf[col] = pdf[col].astype(str)
    pdf[col] = le.fit_transform(pdf[col])
    label_encoders[col] = le

feature_cols = numeric_cols + categorical_cols

X = pdf[feature_cols].values
y = pdf[target_col].values

# ── Train / test split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set : {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
print(f"Test set     : {X_test.shape[0]:,} rows, {X_test.shape[1]} features")
print(f"Target rate  : {y.mean():.2%} converted")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Hyperparameter Tuning with Hyperopt (Serverless-Compatible)
# MAGIC
# MAGIC **Hyperopt** is a Bayesian optimization library -- instead of trying every combination (grid search), it learns from previous trials to focus on promising parameter regions.
# MAGIC
# MAGIC **How it works:**
# MAGIC
# MAGIC ```
# MAGIC ┌───────────────────────────────────────────────────┐
# MAGIC │              Serverless Compute Driver             │
# MAGIC │   Hyperopt TPE Suggest  ──>  fmin() coordinator  │
# MAGIC │                                                   │
# MAGIC │   Trial 1 ──> Trial 2 ──> Trial 3 ──> ...       │
# MAGIC │   XGBoost     XGBoost     XGBoost                │
# MAGIC │   (sequential on driver)                          │
# MAGIC └───────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC - `Trials()` runs sequentially on the driver -- compatible with serverless. `SparkTrials()` distributes trials across worker nodes -- requires a classic multi-node cluster.
# MAGIC - Each trial trains a full XGBoost model with a different hyperparameter combo
# MAGIC - All trials are automatically logged to MLflow
# MAGIC
# MAGIC > **Note:** `SparkTrials` is not supported on serverless compute because it
# MAGIC > requires `sparkContext`. We use `Trials` instead, which works on all compute types.
# MAGIC

# COMMAND ----------

import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Set the MLflow experiment for this module
mlflow.set_experiment(f"/Users/{username}/servicenow_lead_scoring")

# COMMAND ----------

# ── Define the search space ──────────────────────────────────────
# hp.choice passes the *actual chosen value* to the objective function,
# but fmin() returns *indices* in best_params. Keep lookup lists for decoding.
MAX_DEPTH_OPTIONS = [3, 4, 5, 6, 7, 8, 10]
N_ESTIMATORS_OPTIONS = [50, 100, 150, 200, 300]
MIN_CHILD_WEIGHT_OPTIONS = [1, 3, 5, 7, 10]

search_space = {
    "learning_rate":    hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "max_depth":        hp.choice("max_depth", MAX_DEPTH_OPTIONS),
    "n_estimators":     hp.choice("n_estimators", N_ESTIMATORS_OPTIONS),
    "min_child_weight": hp.choice("min_child_weight", MIN_CHILD_WEIGHT_OPTIONS),
    "subsample":        hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    "gamma":            hp.uniform("gamma", 0, 0.5),
}

# ── Define the objective function ────────────────────────────────
def objective(params):
    """
    Train an XGBoost model with the given hyperparameters.
    Returns the negative AUC (because Hyperopt minimizes).

    Note: hp.choice passes the actual chosen value (not the index)
    to the objective function. No index mapping needed here.
    """
    params = dict(params)  # shallow copy to avoid mutating trial data
    # Ensure integer types for XGBoost params that require int
    params["max_depth"]        = int(params["max_depth"])
    params["n_estimators"]     = int(params["n_estimators"])
    params["min_child_weight"] = int(params["min_child_weight"])

    with mlflow.start_run(nested=True):
        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred       = model.predict(X_test)

        auc       = roc_auc_score(y_test, y_pred_proba)
        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_params(params)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

    # Hyperopt minimizes, so return negative AUC
    return {"loss": -auc, "status": STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC **Running 20 trials with Hyperopt Trials...**
# MAGIC
# MAGIC Each trial trains a separate XGBoost model sequentially on the driver.
# MAGIC This is compatible with serverless compute. The search uses TPE (Tree-structured Parzen Estimator),
# MAGIC a Bayesian algorithm that models "good" and "bad" parameter regions separately to guide the search.

# COMMAND ----------

# ── Run Hyperopt ─────────────────────────────────────────────────
# Trials runs sequentially on the driver (serverless-compatible)
# Note: SparkTrials is NOT supported on serverless compute
trials = Trials()

with mlflow.start_run(run_name="hyperopt_xgboost_tuning") as parent_run:
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    parent_run_id = parent_run.info.run_id

print("Hyperopt search complete!")
print(f"Parent run ID: {parent_run_id}")

# COMMAND ----------

# ── Decode best parameters ───────────────────────────────────────
# hp.choice returns indices, so we map them back to actual values
best_decoded = {
    "learning_rate":    best_params["learning_rate"],
    "max_depth":        MAX_DEPTH_OPTIONS[int(best_params["max_depth"])],
    "n_estimators":     N_ESTIMATORS_OPTIONS[int(best_params["n_estimators"])],
    "min_child_weight": MIN_CHILD_WEIGHT_OPTIONS[int(best_params["min_child_weight"])],
    "subsample":        best_params["subsample"],
    "colsample_bytree": best_params["colsample_bytree"],
    "gamma":            best_params["gamma"],
}

print("Best Hyperparameters Found:")
print("-" * 40)
for k, v in best_decoded.items():
    if isinstance(v, float):
        print(f"  {k:20s}: {v:.4f}")
    else:
        print(f"  {k:20s}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Train the Best Model and Evaluate
# MAGIC
# MAGIC Now we retrain with the best hyperparameters and do a thorough evaluation.

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from sklearn.metrics import roc_curve

# ── Train the champion model ─────────────────────────────────────
best_model = XGBClassifier(
    **best_decoded,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
)
best_model.fit(X_train, y_train)

y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred       = best_model.predict(X_test)

# ── Compute all metrics ──────────────────────────────────────────
metrics = {
    "auc":       roc_auc_score(y_test, y_pred_proba),
    "accuracy":  accuracy_score(y_test, y_pred),
    "f1_score":  f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall":    recall_score(y_test, y_pred),
}

print("Best Model Performance:")
print("=" * 40)
for metric_name, metric_val in metrics.items():
    print(f"  {metric_name:12s}: {metric_val:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))

# COMMAND ----------

# ── Visualize: ROC Curve and Confusion Matrix ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color="#10B981", linewidth=2,
             label=f'XGBoost (AUC = {metrics["auc"]:.3f})')
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve -- Best XGBoost Model")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=axes[1],
            xticklabels=["Not Converted", "Converted"],
            yticklabels=["Not Converted", "Converted"])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join(tempfile.gettempdir(), "model_evaluation.png"), dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Bonus: Hyperparameter Tuning with Optuna
# MAGIC
# MAGIC [Optuna](https://optuna.org/) is another popular hyperparameter tuning framework.
# MAGIC While Hyperopt integrates natively with Databricks, Optuna offers
# MAGIC a clean API with powerful features like pruning (early stopping of bad trials)
# MAGIC and a built-in dashboard.
# MAGIC

# COMMAND ----------

import optuna
from optuna.samplers import TPESampler

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optuna_objective(trial):
    """Optuna objective function -- note the cleaner API compared to Hyperopt."""
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300, step=50),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
    }

    model = XGBClassifier(
        **params,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    return auc  # Optuna maximizes by default when direction="maximize"


# ── Run Optuna study ─────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    study_name="xgboost_lead_scoring_optuna",
)

study.optimize(optuna_objective, n_trials=10, show_progress_bar=True)

print(f"\nOptuna Best AUC  : {study.best_value:.4f}")
print(f"Optuna Best Params:")
for k, v in study.best_params.items():
    if isinstance(v, float):
        print(f"  {k:20s}: {v:.4f}")
    else:
        print(f"  {k:20s}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperopt vs. Optuna -- Quick Comparison
# MAGIC
# MAGIC | Feature | Hyperopt (Trials) | Optuna |
# MAGIC |---------|-------------------|--------|
# MAGIC | **Serverless support** | Yes (with `Trials`) | Yes |
# MAGIC | **Search algorithms** | TPE, Random | TPE, CMA-ES, Grid, Random, and more |
# MAGIC | **Pruning** | Not built-in | Built-in (MedianPruner, etc.) |
# MAGIC | **MLflow integration** | Manual logging | Manual (but straightforward) |
# MAGIC | **API style** | Functional (`fmin`) | Object-oriented (`study.optimize`) |
# MAGIC | **Databricks native** | Yes | Community-supported |
# MAGIC
# MAGIC **Recommendation:** Both **Hyperopt** and **Optuna** work on serverless compute.
# MAGIC Use Hyperopt for its native Databricks integration. Use Optuna when you need
# MAGIC advanced pruning or more flexible search algorithms.
# MAGIC
# MAGIC > **Note:** `SparkTrials` (distributed Hyperopt) requires `sparkContext` and is only
# MAGIC > available on classic (non-serverless) compute. On serverless, use `Trials` instead.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 2: MLflow Experiment Tracking & Unity Catalog Model Registry
# MAGIC
# MAGIC **MLflow** is Databricks' open-source platform for ML experiment tracking. It logs hyperparameters, metrics, and model artifacts so every experiment is reproducible and comparable. MLflow is the backbone of MLOps on Databricks. In this section we:
# MAGIC
# MAGIC 1. Log a **production-quality run** with all metrics, params, and artifacts
# MAGIC 2. Register the model in **Unity Catalog** (not the legacy Workspace registry)
# MAGIC 3. Manage model **versions and aliases** (champion / challenger pattern). **Champion/Challenger pattern**: the champion is the current production model; challengers are new candidates tested in parallel. Only promote a challenger to champion when it demonstrably outperforms.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Log a Production Run with Full Artifacts

# COMMAND ----------

import mlflow
import mlflow.xgboost
import json
import tempfile
import os

# Point the model registry at Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{username}/servicenow_lead_scoring")

model_name = f"{catalog}.{schema}.lead_scoring_model"

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_champion_candidate") as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # ── Log hyperparameters ──────────────────────────────────────
    if "best_decoded" not in dir():
        best_decoded = {"note": "hyperopt_skipped"}
    mlflow.log_params(best_decoded)
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("feature_count", len(feature_cols))
    mlflow.log_param("training_rows", X_train.shape[0])
    mlflow.log_param("test_rows", X_test.shape[0])

    # ── Log metrics ──────────────────────────────────────────────
    for metric_name, metric_val in metrics.items():
        mlflow.log_metric(metric_name, metric_val)

    # ── Log the model ────────────────────────────────────────────
    # MLflow's XGBoost flavor creates a reusable model artifact
    mlflow.xgboost.log_model(
        xgb_model=best_model,
        artifact_path="model",
        input_example=X_test[:5],
    )

    # ── Log feature importance plot ──────────────────────────────
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    importance = best_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    top_n = min(15, len(feature_cols))
    ax_imp.barh(
        range(top_n),
        importance[sorted_idx[:top_n]][::-1],
        color="#10B981",
    )
    ax_imp.set_yticks(range(top_n))
    ax_imp.set_yticklabels([feature_cols[i] for i in sorted_idx[:top_n]][::-1])
    ax_imp.set_xlabel("Feature Importance (Gain)")
    ax_imp.set_title("Top Feature Importances -- XGBoost Lead Scoring")
    plt.tight_layout()
    fig_imp.savefig(os.path.join(tempfile.gettempdir(), "feature_importance.png"), dpi=150, bbox_inches="tight")
    mlflow.log_artifact(os.path.join(tempfile.gettempdir(), "feature_importance.png"))
    plt.show()

    # ── Log confusion matrix plot ────────────────────────────────
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax_cm,
                xticklabels=["Not Converted", "Converted"],
                yticklabels=["Not Converted", "Converted"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    plt.tight_layout()
    fig_cm.savefig(os.path.join(tempfile.gettempdir(), "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    mlflow.log_artifact(os.path.join(tempfile.gettempdir(), "confusion_matrix.png"))
    plt.show()

    # ── Log feature list as JSON artifact ────────────────────────
    feature_meta = {
        "feature_columns": feature_cols,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "target_column": target_col,
    }
    with open(os.path.join(tempfile.gettempdir(), "feature_metadata.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)
    mlflow.log_artifact(os.path.join(tempfile.gettempdir(), "feature_metadata.json"))

    # ── Log tags for searchability ───────────────────────────────
    mlflow.set_tag("model_purpose", "lead_scoring")
    mlflow.set_tag("dataset", f"{catalog}.{schema}.gtm_lead_scores")
    mlflow.set_tag("tuning_method", "hyperopt_trials")
    mlflow.set_tag("training_event", "servicenow_workshop")

    print(f"\nRun logged successfully!")
    print(f"  Run ID    : {run_id}")
    print(f"  AUC       : {metrics['auc']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1        : {metrics['f1_score']:.4f}")
    print(f"  Artifacts : feature_importance.png, confusion_matrix.png, feature_metadata.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Register Model in Unity Catalog
# MAGIC
# MAGIC The **Unity Catalog Model Registry** is the modern approach -- it centralizes model governance across all workspaces with the same permissions model as your data tables. (The older Workspace-level registry is deprecated.) Unity Catalog Model Registry provides:
# MAGIC - **Centralized governance** across workspaces
# MAGIC - **Lineage tracking** from data to model to endpoint
# MAGIC - **Access control** via Unity Catalog permissions
# MAGIC - **Model versioning** with aliases (champion, challenger, etc.)
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────┐
# MAGIC │              Unity Catalog                       │
# MAGIC │  ┌──────────┐  ┌─────────┐  ┌───────────────┐  │
# MAGIC │  │ Catalog:  │──│ Schema: │──│ Model:        │  │
# MAGIC │  │ ankit_    │  │ sn_     │  │ lead_scoring_ │  │
# MAGIC │  │ yadav     │  │ training│  │ model         │  │
# MAGIC │  └──────────┘  └─────────┘  │               │  │
# MAGIC │                              │  Version 1 ◄──── champion
# MAGIC │                              │  Version 2 ◄──── challenger
# MAGIC │                              │  Version 3    │  │
# MAGIC │                              └───────────────┘  │
# MAGIC └─────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# ── Register the model ───────────────────────────────────────────
model_uri = f"runs:/{run_id}/model"

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
)

print(f"Model registered!")
print(f"  Name    : {model_version.name}")
print(f"  Version : {model_version.version}")
print(f"  Source  : {model_version.source}")

# COMMAND ----------

# ── Add description to the registered model ──────────────────────
client.update_registered_model(
    name=model_name,
    description=(
        "XGBoost lead scoring model for GTM/Salesforce pipeline. "
        "Predicts probability of lead conversion based on activity engagement, "
        "account attributes, and opportunity data. "
        "Trained during ServiceNow x Databricks workshop."
    ),
)

# ── Add description to this specific version ─────────────────────
client.update_model_version(
    name=model_name,
    version=model_version.version,
    description=(
        f"Tuned with Hyperopt (20 trials). "
        f"AUC={metrics['auc']:.4f}, F1={metrics['f1_score']:.4f}. "
        f"Features: {len(feature_cols)} columns from joined GTM tables."
    ),
)

print("Model and version descriptions updated.")

# COMMAND ----------

# ── Set the "champion" alias ─────────────────────────────────────
# Aliases replace the old Stage-based model promotion (Staging/Production).
# They are flexible labels you can assign to any version.
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=model_version.version,
)

print(f"Alias 'champion' set on version {model_version.version}")

# COMMAND ----------

# ── List all versions and aliases ────────────────────────────────
print(f"Model: {model_name}")
print("=" * 60)

# Search for all versions of this model
from mlflow.entities.model_registry import ModelVersionTag
versions = client.search_model_versions(f"name='{model_name}'")

for v in versions:
    raw_aliases = v.aliases if hasattr(v, "aliases") else []
    # Handle aliases as list, dict, or other types across MLflow versions
    if isinstance(raw_aliases, dict):
        alias_list = list(raw_aliases.keys())
    elif isinstance(raw_aliases, (list, tuple)):
        alias_list = list(raw_aliases)
    else:
        alias_list = []
    alias_str = f" [aliases: {', '.join(alias_list)}]" if alias_list else ""
    print(f"  Version {v.version}: status={v.status}{alias_str}")
    if v.description:
        print(f"    Description: {v.description[:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Load the Model Back from Unity Catalog
# MAGIC
# MAGIC You can load models by version number or alias. Using aliases is the
# MAGIC recommended pattern because it decouples deployment from version numbers.

# COMMAND ----------

# ── Load by alias (recommended for production) ───────────────────
champion_model = mlflow.xgboost.load_model(f"models:/{model_name}@champion")

# Quick sanity check: score a few test rows
sample_preds = champion_model.predict_proba(X_test[:5])[:, 1]
print("Sample predictions from champion model:")
for i, prob in enumerate(sample_preds):
    print(f"  Lead {i+1}: {prob:.3f} conversion probability")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 3: Model Serving Deployment
# MAGIC
# MAGIC Databricks Model Serving provides **real-time, serverless endpoints** for
# MAGIC registered models. Key features:
# MAGIC
# MAGIC - **Auto-scaling** from zero (pay only when traffic arrives)
# MAGIC - **Unity Catalog integration** for governance
# MAGIC - **Inference tables** for automatic payload logging
# MAGIC - **A/B testing** with traffic routing
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Create a Model Serving Endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
import time

w = WorkspaceClient()
endpoint_name = "servicenow-lead-scoring"

# COMMAND ----------

# MAGIC %md
# MAGIC > **Note:** `AutoCaptureConfigInput` is deprecated. Use **AI Gateway inference tables** for production monitoring instead. Configure them via the Databricks UI under Serving > Endpoint > AI Gateway settings.

# COMMAND ----------

# ── Define the endpoint configuration ────────────────────────────
# We create the endpoint without auto_capture_config (use AI Gateway instead).
served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        entity_version=model_version.version,
        workload_size="Small",            # Small / Medium / Large
        scale_to_zero_enabled=True,        # Scale down to 0 when idle
    )
]

endpoint_config = EndpointCoreConfigInput(
    served_entities=served_entities,
)

# ── Create or update the endpoint ────────────────────────────────
try:
    # Check if endpoint already exists
    existing = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists. Updating configuration...")
    try:
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=served_entities,
        )
        print("Endpoint configuration updated.")
    except Exception as update_err:
        if "ResourceConflict" in str(type(update_err).__name__) or "currently being updated" in str(update_err):
            print(f"Endpoint is currently being updated from a previous run. Skipping update.")
            print(f"Current endpoint state will be used.")
        else:
            raise
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e).lower() or "404" in str(e):
        print(f"Creating new endpoint '{endpoint_name}'...")
        try:
            w.serving_endpoints.create(
                name=endpoint_name,
                config=endpoint_config,
            )
            print("Endpoint creation initiated. This may take 5-15 minutes to become ready.")
        except Exception as create_err:
            if "already exists" in str(create_err).lower() or "ResourceConflict" in str(type(create_err).__name__):
                print(f"Endpoint was created concurrently. Proceeding.")
            else:
                raise
    elif "ResourceConflict" in str(type(e).__name__) or "currently being updated" in str(e):
        print(f"Endpoint is currently being updated. Skipping. Proceeding with existing endpoint.")
    else:
        print(f"Error managing endpoint: {e}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Wait for Endpoint to Be Ready
# MAGIC

# COMMAND ----------

# ── Poll endpoint status ─────────────────────────────────────────
def check_endpoint_status(endpoint_name):
    """Check the current status of a serving endpoint."""
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state
        print(f"Endpoint: {endpoint_name}")
        print(f"  Ready state  : {state.ready}")
        print(f"  Config update: {state.config_update}")
        return state.ready == "READY"
    except Exception as e:
        print(f"Could not retrieve endpoint status: {e}")
        return False

is_ready = check_endpoint_status(endpoint_name)
if not is_ready:
    print("\nEndpoint is not yet ready. You can re-run this cell to check again.")
    print("Alternatively, check the Serving tab in the Databricks UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Query the Serving Endpoint
# MAGIC
# MAGIC Once the endpoint is `READY`, we can send real-time scoring requests.
# MAGIC This is how your application would call the model in production.

# COMMAND ----------

# ── Build sample payload ─────────────────────────────────────────
# Create a small DataFrame of sample leads to score
sample_leads = pdf[feature_cols].head(5).to_dict(orient="split")

print("Sample payload (first 2 rows):")
for i, row in enumerate(sample_leads["data"][:2]):
    print(f"  Lead {i+1}: {dict(zip(sample_leads['columns'], row))}")

# COMMAND ----------

# ── Query the endpoint ───────────────────────────────────────────
try:
    response = w.serving_endpoints.query(
        name=endpoint_name,
        dataframe_split=sample_leads,
    )
    print("Real-time predictions:")
    print("-" * 40)
    if hasattr(response, "predictions"):
        for i, pred in enumerate(response.predictions):
            print(f"  Lead {i+1}: {pred}")
    else:
        print(f"  Response: {response}")
except Exception as e:
    error_msg = str(e)
    if "NOT_READY" in error_msg or "not ready" in error_msg.lower():
        print("Endpoint is still provisioning. Please wait a few minutes and re-run this cell.")
    else:
        print(f"Error querying endpoint: {e}")
        print("\nIf the endpoint is not yet ready, wait a few minutes and try again.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 A/B Testing and Traffic Routing
# MAGIC
# MAGIC Databricks Model Serving supports **traffic splitting** for safe deployments:
# MAGIC
# MAGIC ```
# MAGIC                    ┌───────────────────────────┐
# MAGIC                    │     Serving Endpoint      │
# MAGIC                    │  "lead-scoring-endpoint"  │
# MAGIC                    └────────────┬──────────────┘
# MAGIC                                 │
# MAGIC                    ┌────────────┴──────────────┐
# MAGIC                    │     Traffic Router         │
# MAGIC                    │                            │
# MAGIC                    │   90%            10%       │
# MAGIC                    └─────┬──────────────┬──────┘
# MAGIC                          │              │
# MAGIC               ┌─────────▼──────┐  ┌───▼─────────────┐
# MAGIC               │  Champion v1   │  │  Challenger v2   │
# MAGIC               │  XGBoost       │  │  LightGBM        │
# MAGIC               │  AUC = 0.87    │  │  AUC = 0.89      │
# MAGIC               └────────────────┘  └──────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **How to configure A/B testing:**
# MAGIC
# MAGIC ```python
# MAGIC # Example: Route 90% traffic to champion, 10% to challenger
# MAGIC served_entities = [
# MAGIC     ServedEntityInput(
# MAGIC         entity_name="catalog.schema.lead_scoring_model",
# MAGIC         entity_version="1",
# MAGIC         workload_size="Small",
# MAGIC         scale_to_zero_enabled=True,
# MAGIC         traffic_percentage=90,   # <-- 90% of traffic
# MAGIC     ),
# MAGIC     ServedEntityInput(
# MAGIC         entity_name="catalog.schema.lead_scoring_model",
# MAGIC         entity_version="2",
# MAGIC         workload_size="Small",
# MAGIC         scale_to_zero_enabled=True,
# MAGIC         traffic_percentage=10,   # <-- 10% of traffic
# MAGIC     ),
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC **Blue-Green Deployment Pattern:**
# MAGIC
# MAGIC ```
# MAGIC    Blue (Current)              Green (New)
# MAGIC   ┌───────────────┐          ┌───────────────┐
# MAGIC   │  Version 1    │          │  Version 2    │
# MAGIC   │  100% traffic │  ──────> │  100% traffic │
# MAGIC   │  (serving)    │  switch  │  (validated)  │
# MAGIC   └───────────────┘          └───────────────┘
# MAGIC
# MAGIC   Steps:
# MAGIC   1. Deploy v2 with 0% traffic
# MAGIC   2. Run validation tests against v2
# MAGIC   3. If passed: shift 100% traffic to v2
# MAGIC   4. If failed: remove v2, keep v1 at 100%
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 4: Inference Tables & Monitoring
# MAGIC
# MAGIC Production ML requires continuous monitoring. Databricks provides:
# MAGIC
# MAGIC 1. **Inference Tables** -- automatic payload logging for every serving request
# MAGIC 2. **Lakehouse Monitoring** -- drift detection, data quality, statistical tests
# MAGIC 3. **System Tables** -- operational metrics (latency, throughput, errors)
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │                    Monitoring Architecture                       │
# MAGIC │                                                                  │
# MAGIC │  Request ──> Serving ──> Inference Table ──> Lakehouse Monitor  │
# MAGIC │              Endpoint    (auto-captured)      (drift detection)  │
# MAGIC │                │                                    │            │
# MAGIC │                ▼                                    ▼            │
# MAGIC │          System Tables                        Alert / Retrain   │
# MAGIC │          (latency, errors)                    Pipeline Trigger  │
# MAGIC └─────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Inference Tables
# MAGIC
# MAGIC **Inference Tables** are Delta tables automatically populated by serving endpoints -- every request/response is logged with timestamps and latency, giving you a complete audit trail.
# MAGIC
# MAGIC When you enable **Auto Capture** on a serving endpoint, Databricks automatically
# MAGIC logs every request and response to a Delta table in Unity Catalog.
# MAGIC
# MAGIC **What gets captured:**
# MAGIC
# MAGIC | Column | Description |
# MAGIC |--------|-------------|
# MAGIC | `timestamp` | When the request was made |
# MAGIC | `date` | Date partition |
# MAGIC | `request` | Full JSON request payload |
# MAGIC | `response` | Full JSON response payload |
# MAGIC | `status_code` | HTTP status code |
# MAGIC | `execution_time_ms` | End-to-end latency |
# MAGIC | `served_entity_name` | Which model version served it |
# MAGIC | `client_request_id` | For request tracing |
# MAGIC
# MAGIC Inference tables can be enabled via **AI Gateway** (the recommended approach),
# MAGIC which replaces the legacy `AutoCaptureConfigInput`. Configure inference tables
# MAGIC through the Databricks UI under Serving > Endpoint > AI Gateway settings.
# MAGIC
# MAGIC The inference table will be created at:
# MAGIC `{catalog}.{schema}.lead_scoring_<endpoint_name>_payload`

# COMMAND ----------

# ── Check if inference table exists ──────────────────────────────
# Note: The table is created after the first request to the endpoint.
# It may not exist yet if no requests have been sent.
inference_table_name = f"{catalog}.{schema}.lead_scoring_{endpoint_name.replace('-', '_')}_payload"

try:
    inf_df = spark.table(inference_table_name)
    print(f"Inference table found: {inference_table_name}")
    print(f"  Row count: {inf_df.count()}")
    display(inf_df.limit(5))
except Exception as e:
    print(f"Inference table not yet available: {inference_table_name}")
    print("This table is created after the first request hits the serving endpoint.")
    print("Once the endpoint is ready and you've sent requests, re-run this cell.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Monitoring with System Tables
# MAGIC
# MAGIC Databricks system tables provide operational metrics for serving endpoints.
# MAGIC These are available in `system.serving` (if enabled for your workspace).
# MAGIC

# COMMAND ----------

# ── Query system tables for serving metrics ──────────────────────
# Note: system.serving tables may not be available in all workspaces.
# These queries are wrapped in try/except so the notebook continues
# even if system tables are not enabled.

try:
    # Request volume over time (last 24 hours)
    request_volume_df = spark.sql("""
        SELECT
          date_trunc('hour', request_time) AS hour,
          COUNT(*) AS request_count,
          AVG(execution_time_ms) AS avg_latency_ms,
          PERCENTILE(execution_time_ms, 0.95) AS p95_latency_ms,
          PERCENTILE(execution_time_ms, 0.99) AS p99_latency_ms
        FROM system.serving.served_model_requests
        WHERE
          served_entity_name LIKE '%lead_scoring%'
          AND request_time > current_timestamp() - INTERVAL 24 HOURS
        GROUP BY 1
        ORDER BY 1
    """)
    display(request_volume_df)
except Exception as e:
    print(f"System table query skipped: {type(e).__name__}")
    print("system.serving tables may not be enabled in this workspace.")
    print("This is expected -- these queries demonstrate monitoring patterns.")

# COMMAND ----------

try:
    # Error rate monitoring
    error_rate_df = spark.sql("""
        SELECT
          date_trunc('hour', request_time) AS hour,
          COUNT(*) AS total_requests,
          SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS error_count,
          ROUND(
            SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
            2
          ) AS error_rate_pct
        FROM system.serving.served_model_requests
        WHERE
          served_entity_name LIKE '%lead_scoring%'
          AND request_time > current_timestamp() - INTERVAL 24 HOURS
        GROUP BY 1
        ORDER BY 1
    """)
    display(error_rate_df)
except Exception as e:
    print(f"Error rate query skipped: {type(e).__name__}")

# COMMAND ----------

try:
    # Latency distribution by served entity (useful for A/B testing)
    latency_df = spark.sql("""
        SELECT
          served_entity_name,
          COUNT(*) AS request_count,
          ROUND(AVG(execution_time_ms), 1) AS avg_latency_ms,
          ROUND(PERCENTILE(execution_time_ms, 0.50), 1) AS p50_latency_ms,
          ROUND(PERCENTILE(execution_time_ms, 0.95), 1) AS p95_latency_ms,
          ROUND(PERCENTILE(execution_time_ms, 0.99), 1) AS p99_latency_ms,
          MIN(request_time) AS first_request,
          MAX(request_time) AS last_request
        FROM system.serving.served_model_requests
        WHERE
          served_entity_name LIKE '%lead_scoring%'
          AND request_time > current_timestamp() - INTERVAL 7 DAYS
        GROUP BY 1
        ORDER BY request_count DESC
    """)
    display(latency_df)
except Exception as e:
    print(f"Latency query skipped: {type(e).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Data Drift Detection
# MAGIC
# MAGIC **Data drift** occurs when the distribution of incoming features changes
# MAGIC compared to the training data. This is a leading indicator that model
# MAGIC performance may degrade.
# MAGIC
# MAGIC We demonstrate drift detection by comparing training data distributions
# MAGIC with simulated "production" data.

# COMMAND ----------

from scipy import stats

# ── Simulate production data with drift ──────────────────────────
# In practice, you would pull this from the inference table.
# Here we simulate a production batch where some features have shifted.
np.random.seed(42)

production_pdf = pdf[feature_cols].copy()

# Introduce artificial drift in a few features to demonstrate detection
# Shift 'total_activities' upward (more engaged leads coming in)
if "total_activities" in production_pdf.columns:
    production_pdf["total_activities"] = production_pdf["total_activities"] + np.random.normal(3, 1, len(production_pdf))

# Shift 'annual_revenue' (larger companies entering pipeline)
if "annual_revenue" in production_pdf.columns:
    production_pdf["annual_revenue"] = production_pdf["annual_revenue"] * 1.5

print("Simulated production data with drift:")
print(f"  Rows: {len(production_pdf):,}")
print(f"  Features: {len(production_pdf.columns)}")
print(f"  Drift injected: total_activities (shifted +3), annual_revenue (scaled 1.5x)")

# COMMAND ----------

# ── Run drift detection on numeric features ──────────────────────
# The Kolmogorov-Smirnov (KS) test compares two probability distributions.
# A p-value below 0.05 means the distributions are statistically different
# -- a signal that input data has drifted from what the model was trained on.

drift_results = []

for col in numeric_cols:
    if col in production_pdf.columns and col in pdf.columns:
        training_vals = pdf[col].dropna().values
        production_vals = production_pdf[col].dropna().values

        if len(training_vals) > 0 and len(production_vals) > 0:
            ks_stat, p_value = stats.ks_2samp(training_vals, production_vals)

            drift_results.append({
                "feature": col,
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "drift_detected": "YES" if p_value < 0.05 else "no",
                "training_mean": round(np.mean(training_vals), 2),
                "production_mean": round(np.mean(production_vals), 2),
                "mean_shift_pct": round(
                    abs(np.mean(production_vals) - np.mean(training_vals))
                    / (abs(np.mean(training_vals)) + 1e-10) * 100, 1
                ),
            })

drift_df = pd.DataFrame(drift_results).sort_values("p_value")
print("Drift Detection Results (KS Test)")
print("=" * 90)
print(drift_df.to_string(index=False))

# COMMAND ----------

# ── Visualize drift for top features ─────────────────────────────
drifted_features = drift_df[drift_df["drift_detected"] == "YES"]["feature"].tolist()

if drifted_features:
    n_plots = min(4, len(drifted_features))
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, feat in zip(axes, drifted_features[:n_plots]):
        ax.hist(pdf[feat].dropna(), bins=30, alpha=0.6, label="Training", color="#3B82F6", density=True)
        ax.hist(production_pdf[feat].dropna(), bins=30, alpha=0.6, label="Production", color="#EF4444", density=True)
        ax.set_title(f"{feat}\n(drift detected)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Feature Distribution Drift: Training vs. Production", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("No significant drift detected in numeric features.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Lakehouse Monitoring (Conceptual)
# MAGIC
# MAGIC **Lakehouse Monitoring** is Databricks' automated monitoring suite that continuously checks for data drift, profile changes, and model quality degradation. You can create a monitor on any Delta table, including inference tables.
# MAGIC
# MAGIC ```python
# MAGIC # Example: Create a monitor on the inference table
# MAGIC from databricks.sdk.service.catalog import MonitorInferenceLog
# MAGIC
# MAGIC w.quality_monitors.create(
# MAGIC     table_name=f"{catalog}.{schema}.lead_scoring_inference_payload",
# MAGIC     inference_log=MonitorInferenceLog(
# MAGIC         problem_type="classification",
# MAGIC         prediction_col="prediction",
# MAGIC         label_col="converted",           # if labels are available
# MAGIC         model_id_col="served_entity_name",
# MAGIC         timestamp_col="timestamp",
# MAGIC     ),
# MAGIC     output_schema_name=f"{catalog}.{schema}",
# MAGIC     schedule=MonitorCronSchedule(
# MAGIC         quartz_cron_expression="0 0 * * * ?",  # Every hour
# MAGIC         timezone_id="UTC",
# MAGIC     ),
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **What Lakehouse Monitoring provides:**
# MAGIC
# MAGIC | Metric | Description |
# MAGIC |--------|-------------|
# MAGIC | **Profile metrics** | Summary stats (mean, stddev, nulls, distinct counts) |
# MAGIC | **Drift metrics** | KS test, Chi-squared test, Jensen-Shannon divergence |
# MAGIC | **Model quality** | Accuracy, AUC, precision, recall (when labels available) |
# MAGIC | **Custom metrics** | Define your own metrics with SQL expressions |
# MAGIC
# MAGIC The monitor writes results to two tables:
# MAGIC - `{table_name}_profile_metrics` -- per-column statistics over time
# MAGIC - `{table_name}_drift_metrics` -- statistical drift tests over time

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Automated Alerting
# MAGIC
# MAGIC Once monitoring is in place, set up alerts to catch issues early:
# MAGIC
# MAGIC **Option 1: Databricks SQL Alerts**
# MAGIC ```sql
# MAGIC -- Alert when AUC drops below threshold
# MAGIC SELECT
# MAGIC   window_start,
# MAGIC   auc
# MAGIC FROM {catalog}.{schema}.lead_scoring_inference_drift_metrics
# MAGIC WHERE metric_name = 'auc'
# MAGIC   AND auc < 0.75
# MAGIC   AND window_start > current_timestamp() - INTERVAL 1 HOUR
# MAGIC ```
# MAGIC
# MAGIC **Option 2: Databricks SQL Alert on Drift**
# MAGIC ```sql
# MAGIC -- Alert when KS statistic exceeds threshold for any feature
# MAGIC SELECT
# MAGIC   column_name,
# MAGIC   ks_statistic,
# MAGIC   p_value,
# MAGIC   window_start
# MAGIC FROM {catalog}.{schema}.lead_scoring_inference_drift_metrics
# MAGIC WHERE ks_statistic > 0.15
# MAGIC   AND p_value < 0.01
# MAGIC   AND window_start > current_timestamp() - INTERVAL 1 HOUR
# MAGIC ```
# MAGIC
# MAGIC **Option 3: Integration with ServiceNow**
# MAGIC
# MAGIC With ServiceNow's ITSM capabilities, you can:
# MAGIC 1. Create an **incident** when drift is detected (webhook to ServiceNow API)
# MAGIC 2. Trigger a **change request** for model retraining
# MAGIC 3. Route alerts through ServiceNow **Event Management**
# MAGIC
# MAGIC ```python
# MAGIC # Example: Webhook notification on drift
# MAGIC import requests
# MAGIC
# MAGIC def send_drift_alert(feature_name, ks_stat, p_value):
# MAGIC     payload = {
# MAGIC         "short_description": f"ML Model Drift Detected: {feature_name}",
# MAGIC         "description": (
# MAGIC             f"Feature '{feature_name}' shows significant drift. "
# MAGIC             f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.6f}. "
# MAGIC             f"Model: lead_scoring_model. Action: evaluate retraining."
# MAGIC         ),
# MAGIC         "urgency": "2",
# MAGIC         "impact": "2",
# MAGIC     }
# MAGIC     # POST to ServiceNow incident API or Databricks SQL Alert webhook
# MAGIC     # requests.post(servicenow_url, json=payload, auth=(...))
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 5: Workflows & DABs Overview
# MAGIC
# MAGIC Now that we have trained, registered, deployed, and monitored a model, let us
# MAGIC look at how to **orchestrate and productionize** the entire pipeline.
# MAGIC
# MAGIC This section is **conceptual** -- we walk through the architecture and
# MAGIC configuration, but do not execute the workflow definitions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Databricks Workflows
# MAGIC
# MAGIC Databricks Workflows is the native orchestration engine for data and ML pipelines.
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────┐
# MAGIC │                   ML Pipeline Workflow                       │
# MAGIC │                                                              │
# MAGIC │  ┌───────────┐   ┌──────────────┐   ┌─────────────────┐    │
# MAGIC │  │  Task 1   │──>│   Task 2     │──>│    Task 3       │    │
# MAGIC │  │ Feature   │   │ Train &      │   │ Register &      │    │
# MAGIC │  │ Engineer  │   │ Tune Model   │   │ Deploy Model    │    │
# MAGIC │  └───────────┘   └──────┬───────┘   └────────┬────────┘    │
# MAGIC │                         │                     │             │
# MAGIC │                         ▼                     ▼             │
# MAGIC │                  ┌──────────────┐   ┌─────────────────┐    │
# MAGIC │                  │   Task 2a    │   │    Task 4       │    │
# MAGIC │                  │ Validation   │   │ Run Monitoring  │    │
# MAGIC │                  │ Tests        │   │ Checks          │    │
# MAGIC │                  └──────────────┘   └─────────────────┘    │
# MAGIC └──────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Key features:**
# MAGIC - **Task dependencies** -- DAG-based execution with conditional branches
# MAGIC - **Task types** -- Notebooks, Python scripts, SQL, dbt, JAR, Spark Submit
# MAGIC - **Triggers** -- Cron schedule, file arrival, API trigger, continuous
# MAGIC - **Alerting** -- Email, webhook, PagerDuty, Slack on success/failure
# MAGIC - **Repair & retry** -- Automatic retries with exponential backoff
# MAGIC - **Parameterization** -- Dynamic values passed between tasks via `dbutils.jobs.taskValues`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Databricks Asset Bundles (DABs) -- Infrastructure as Code
# MAGIC
# MAGIC **Databricks Asset Bundles** let you define your entire ML platform as code:
# MAGIC - Notebooks, jobs, pipelines, serving endpoints
# MAGIC - Permissions, clusters, secrets
# MAGIC - All version-controlled in Git, deployed via CI/CD
# MAGIC
# MAGIC Here is a sample `databricks.yml` bundle definition for our lead scoring pipeline:
# MAGIC
# MAGIC ```yaml
# MAGIC # databricks.yml -- Lead Scoring ML Pipeline Bundle
# MAGIC bundle:
# MAGIC   name: servicenow-lead-scoring
# MAGIC
# MAGIC include:
# MAGIC   - resources/*.yml
# MAGIC
# MAGIC workspace:
# MAGIC   root_path: /Workspace/Users/${workspace.current_user.userName}/.bundle/${bundle.name}/${bundle.target}
# MAGIC
# MAGIC targets:
# MAGIC   dev:
# MAGIC     mode: development
# MAGIC     default: true
# MAGIC     workspace:
# MAGIC       host: https://<workspace-url>
# MAGIC     variables:
# MAGIC       catalog: ankit_yadav
# MAGIC       schema: servicenow_training_dev
# MAGIC
# MAGIC   staging:
# MAGIC     workspace:
# MAGIC       host: https://<workspace-url>
# MAGIC     variables:
# MAGIC       catalog: ankit_yadav
# MAGIC       schema: servicenow_training_staging
# MAGIC
# MAGIC   production:
# MAGIC     workspace:
# MAGIC       host: https://<workspace-url>
# MAGIC     variables:
# MAGIC       catalog: ankit_yadav
# MAGIC       schema: servicenow_training
# MAGIC     run_as:
# MAGIC       service_principal_name: ml-pipeline-sp
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Multi-Task Job Definition
# MAGIC
# MAGIC The workflow job is defined in a separate YAML resource file:
# MAGIC
# MAGIC ```yaml
# MAGIC # resources/lead_scoring_job.yml
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     lead_scoring_pipeline:
# MAGIC       name: "[${bundle.target}] Lead Scoring ML Pipeline"
# MAGIC       schedule:
# MAGIC         quartz_cron_expression: "0 0 6 * * ?"   # Daily at 6 AM UTC
# MAGIC         timezone_id: "UTC"
# MAGIC       email_notifications:
# MAGIC         on_failure:
# MAGIC           - ml-team@company.com
# MAGIC       tags:
# MAGIC         project: servicenow-lead-scoring
# MAGIC         team: data-science
# MAGIC
# MAGIC       tasks:
# MAGIC         # Task 1: Feature Engineering
# MAGIC         - task_key: feature_engineering
# MAGIC           notebook_task:
# MAGIC             notebook_path: ../notebooks/01_Feature_Engineering.py
# MAGIC             base_parameters:
# MAGIC               catalog: ${var.catalog}
# MAGIC               schema: ${var.schema}
# MAGIC           environment_key: ml_env
# MAGIC
# MAGIC         # Task 2: Model Training & Tuning
# MAGIC         - task_key: model_training
# MAGIC           depends_on:
# MAGIC             - task_key: feature_engineering
# MAGIC           notebook_task:
# MAGIC             notebook_path: ../notebooks/02_Model_Training.py
# MAGIC             base_parameters:
# MAGIC               catalog: ${var.catalog}
# MAGIC               schema: ${var.schema}
# MAGIC               max_evals: "50"
# MAGIC           environment_key: ml_env
# MAGIC
# MAGIC         # Task 3: Model Validation
# MAGIC         - task_key: model_validation
# MAGIC           depends_on:
# MAGIC             - task_key: model_training
# MAGIC           notebook_task:
# MAGIC             notebook_path: ../notebooks/03_Model_Validation.py
# MAGIC             base_parameters:
# MAGIC               catalog: ${var.catalog}
# MAGIC               schema: ${var.schema}
# MAGIC               min_auc_threshold: "0.75"
# MAGIC           environment_key: ml_env
# MAGIC
# MAGIC         # Task 4: Deploy to Serving (only if validation passes)
# MAGIC         - task_key: model_deployment
# MAGIC           depends_on:
# MAGIC             - task_key: model_validation
# MAGIC           notebook_task:
# MAGIC             notebook_path: ../notebooks/04_Model_Deployment.py
# MAGIC             base_parameters:
# MAGIC               catalog: ${var.catalog}
# MAGIC               schema: ${var.schema}
# MAGIC               endpoint_name: servicenow-lead-scoring
# MAGIC           environment_key: ml_env
# MAGIC
# MAGIC         # Task 5: Run Monitoring Checks
# MAGIC         - task_key: monitoring_checks
# MAGIC           depends_on:
# MAGIC             - task_key: model_deployment
# MAGIC           notebook_task:
# MAGIC             notebook_path: ../notebooks/05_Monitoring_Checks.py
# MAGIC             base_parameters:
# MAGIC               catalog: ${var.catalog}
# MAGIC               schema: ${var.schema}
# MAGIC           environment_key: ml_env
# MAGIC
# MAGIC       environments:
# MAGIC         - environment_key: ml_env
# MAGIC           spec:
# MAGIC             client: "1"
# MAGIC             dependencies:
# MAGIC               - xgboost
# MAGIC               - lightgbm
# MAGIC               - scikit-learn
# MAGIC               - mlflow
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 CI/CD Integration with Git
# MAGIC
# MAGIC The full MLOps lifecycle with DABs and Git:
# MAGIC
# MAGIC ```
# MAGIC ┌─────────┐     ┌─────────┐     ┌──────────┐     ┌────────────┐
# MAGIC │  Dev    │────>│  PR /   │────>│  CI/CD   │────>│ Databricks │
# MAGIC │  writes │     │  Review │     │  Pipeline│     │ Workspace  │
# MAGIC │  code   │     │         │     │          │     │            │
# MAGIC └─────────┘     └─────────┘     └──────────┘     └────────────┘
# MAGIC      │                               │                 │
# MAGIC      ▼                               ▼                 ▼
# MAGIC  notebooks/            GitHub Actions /           deploy to
# MAGIC  databricks.yml        Azure DevOps:             target env:
# MAGIC  resources/            - lint & test              - dev
# MAGIC                        - databricks bundle       - staging
# MAGIC                          validate                - production
# MAGIC                        - databricks bundle
# MAGIC                          deploy --target X
# MAGIC ```
# MAGIC
# MAGIC **Example GitHub Actions workflow:**
# MAGIC
# MAGIC ```yaml
# MAGIC # .github/workflows/deploy.yml
# MAGIC name: Deploy ML Pipeline
# MAGIC
# MAGIC on:
# MAGIC   push:
# MAGIC     branches: [main]
# MAGIC   pull_request:
# MAGIC     branches: [main]
# MAGIC
# MAGIC jobs:
# MAGIC   validate:
# MAGIC     runs-on: ubuntu-latest
# MAGIC     steps:
# MAGIC       - uses: actions/checkout@v4
# MAGIC       - uses: databricks/setup-cli@main
# MAGIC       - run: databricks bundle validate --target staging
# MAGIC
# MAGIC   deploy-staging:
# MAGIC     needs: validate
# MAGIC     if: github.event_name == 'push'
# MAGIC     runs-on: ubuntu-latest
# MAGIC     environment: staging
# MAGIC     steps:
# MAGIC       - uses: actions/checkout@v4
# MAGIC       - uses: databricks/setup-cli@main
# MAGIC       - run: databricks bundle deploy --target staging
# MAGIC       - run: databricks bundle run lead_scoring_pipeline --target staging
# MAGIC
# MAGIC   deploy-production:
# MAGIC     needs: deploy-staging
# MAGIC     runs-on: ubuntu-latest
# MAGIC     environment: production
# MAGIC     steps:
# MAGIC       - uses: actions/checkout@v4
# MAGIC       - uses: databricks/setup-cli@main
# MAGIC       - run: databricks bundle deploy --target production
# MAGIC ```
# MAGIC
# MAGIC **Key principles:**
# MAGIC - **Everything in Git** -- notebooks, configs, pipeline definitions
# MAGIC - **Environment promotion** -- dev -> staging -> production
# MAGIC - **Service principals** -- production runs under a service principal, not a user
# MAGIC - **Automated testing** -- validation notebooks run as part of CI
# MAGIC - **Bundle variables** -- same code, different catalogs/schemas per environment

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Summary & Key Takeaways
# MAGIC
# MAGIC ## What We Covered in Module 2
# MAGIC
# MAGIC | Topic | Key Databricks Feature | Why It Matters |
# MAGIC |-------|----------------------|----------------|
# MAGIC | Hyperparameter tuning | Hyperopt + Trials | Serverless-compatible hyperparameter search |
# MAGIC | Experiment tracking | MLflow | Full reproducibility: params, metrics, artifacts |
# MAGIC | Model registry | Unity Catalog | Centralized governance, lineage, aliases |
# MAGIC | Model serving | Serverless endpoints | Real-time predictions with auto-scaling |
# MAGIC | A/B testing | Traffic routing | Safe model rollouts with measurable impact |
# MAGIC | Inference tables | Auto capture | Automatic payload logging for every prediction |
# MAGIC | Monitoring | Lakehouse Monitoring | Automated drift detection and quality checks |
# MAGIC | Orchestration | Workflows + DABs | Production pipelines as code with CI/CD |
# MAGIC
# MAGIC ## The MLOps Maturity Journey
# MAGIC
# MAGIC ```
# MAGIC Level 0          Level 1              Level 2              Level 3
# MAGIC Manual            Automated            CI/CD                Full MLOps
# MAGIC ─────────────────────────────────────────────────────────────────────
# MAGIC - Notebooks       - MLflow tracking    - DABs bundles       - A/B testing
# MAGIC - No versioning   - Model registry     - Git integration    - Auto-retraining
# MAGIC - Copy/paste      - Basic serving      - Multi-environment  - Drift alerts
# MAGIC   deployment      - Manual deploy      - Automated deploy   - Feedback loops
# MAGIC ```
# MAGIC
# MAGIC ## Next Steps
# MAGIC - **Module 3**: AI Agents & Compound AI Systems (GenAI + LangChain + RAG)
# MAGIC - **Module 4**: Governance, Lineage & Unity Catalog Deep Dive
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *ServiceNow x Databricks Training Workshop | Module 2 Complete*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (Optional)
# MAGIC
# MAGIC Uncomment and run the cell below to clean up the serving endpoint created
# MAGIC during this module. **Only do this after the workshop or if you need to free resources.**

# COMMAND ----------

# # ── Uncomment to delete the serving endpoint ──────────────────
# try:
#     w.serving_endpoints.delete(endpoint_name)
#     print(f"Serving endpoint '{endpoint_name}' deleted.")
# except Exception as e:
#     print(f"Could not delete endpoint: {e}")
