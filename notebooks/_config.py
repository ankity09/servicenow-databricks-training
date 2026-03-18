# Databricks notebook source

# MAGIC %md
# MAGIC # Training Configuration
# MAGIC
# MAGIC **This notebook defines all shared settings for the training.**
# MAGIC Every other notebook runs this automatically via `%run ./_config`.
# MAGIC
# MAGIC If you need to change the catalog, schema, or any endpoint — change it here once.

# COMMAND ----------

# --- Catalog & Schema ---
# Change these if your instructor provides different values
catalog = "ankit_yadav"
schema = "servicenow_training"

# --- Foundation Model Endpoints ---
llm_endpoint = "databricks-meta-llama-3-3-70b-instruct"
embedding_endpoint = "databricks-gte-large-en"

# --- Vector Search ---
vs_endpoint_name = "mas-3876475e-endpoint"
vs_index_name = f"{catalog}.{schema}.gtm_knowledge_index"

# --- Model Serving ---
serving_endpoint_name = "servicenow-training-lead-scoring"

# --- Convenience ---
username = spark.sql("SELECT current_user()").first()[0]
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Note: Each notebook calls USE CATALOG / USE SCHEMA after %run ./_config
# (Notebook 00 creates the schema first, so we don't USE SCHEMA here)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick Reference
# MAGIC | Setting | Value |
# MAGIC |---------|-------|
# MAGIC | Catalog | `ankit_yadav` |
# MAGIC | Schema | `servicenow_training` |
# MAGIC | LLM | `databricks-meta-llama-3-3-70b-instruct` |
# MAGIC | Embeddings | `databricks-gte-large-en` |
# MAGIC | Vector Search | `mas-3876475e-endpoint` |

# COMMAND ----------

print(f"Config loaded successfully")
print(f"   Catalog:    {catalog}")
print(f"   Schema:     {schema}")
print(f"   User:       {username}")
print(f"   Workspace:  {workspace_url}")
print(f"   LLM:        {llm_endpoint}")
print(f"   Embeddings: {embedding_endpoint}")
