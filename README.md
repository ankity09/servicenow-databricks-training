# Databricks Training: Machine Learning at Scale & Foundations of Gen AI

One-day hands-on training covering distributed ML, production MLOps, GenAI foundations, and agentic AI on the Databricks platform — all built around a realistic Go-To-Market (GTM) data scenario.

---

## Agenda

| Time | Module | Notebook | Topics |
|------|--------|----------|--------|
| 9:00 – 9:30 | **Setup** | `00_Setup_and_Data_Generation` | Environment setup, synthetic GTM data generation |
| 9:30 – 11:00 | **Module 1** | `01_Spark_Architecture_Distributed_Training` | Spark execution plans, Spark ML pipelines, Pandas API on Spark, Pandas UDFs |
| 11:00 – 11:15 | *Break* | | |
| 11:15 – 12:45 | **Module 2** | `02_Advanced_MLOps_Production_Governance` | Hyperopt tuning, MLflow tracking, Unity Catalog model registry, Model Serving, monitoring |
| 12:45 – 1:45 | *Lunch* | | |
| 1:45 – 3:15 | **Module 3** | `03_GenAI_Foundations_Agent_Design` | Foundation Model APIs, prompt engineering, Vector Search, agent tools, Agent Bricks & MCP |
| 3:15 – 3:30 | *Break* | | |
| 3:30 – 5:00 | **Module 4** | `04_Custom_Agents_Evaluation_Governance` | Custom agents (OpenAI SDK), MLflow ResponsesAgent, AI Gateway, tracing, LLM-as-judge evaluation |

---

## Prerequisites

Before the training, please ensure the following:

1. **Databricks workspace access** — You should be able to log into the provided workspace URL
2. **Python proficiency** — Comfortable with Python, pandas, and basic SQL
3. **ML fundamentals** — Familiarity with classification, train/test splits, and evaluation metrics
4. **LLM basics** — High-level understanding of what large language models are
5. **Databricks Runtime** — DBR ML 15.4+ (serverless) or DBR ML 14.3+ (classic clusters)
6. **Unity Catalog permissions** — The following grants on the training catalog/schema:
   - `USE CATALOG`, `USE SCHEMA`, `SELECT`, `CREATE TABLE`
   - `CREATE FUNCTION`, `CREATE MODEL` (required for Modules 2–4)

No prior Databricks experience is required.

---

## Compute & Environment

| Option | Runtime | When to Use |
|--------|---------|-------------|
| **Serverless** (recommended) | DBR ML 15.4+ | Zero config, auto-scales, best for training workshops |
| **Classic cluster + Photon** | DBR ML 14.3+ | Needed for `TorchDistributor` (Module 1) or Spark ML pipeline APIs |

### Sizing Guidance (Classic Clusters Only)

| Cluster Type | Node Config | Use Case |
|-------------|-------------|----------|
| Classic — CPU | `i3.xlarge` x 2–4 workers | Spark ML pipelines, Hyperopt |
| Classic — GPU | `g5.xlarge` x 1–2 workers | TorchDistributor, fine-tuning |

> **What is Photon?** A C++ vectorized engine that accelerates Spark SQL and DataFrame operations 2–5x. Enable it on the cluster creation page under "Use Photon Acceleration." Photon is **not** available on serverless — serverless has its own optimized runtime.

---

## Getting Started

### 1. Import Notebooks

1. Log into the Databricks workspace (URL provided by your instructor)
2. Navigate to **Workspace** → **Home** → your user folder
3. Click **Import** and select one of:
   - **From Git repo**: `https://github.com/ankity09/servicenow-databricks-training` → Path: `notebooks/`
   - **From file**: download this repo as ZIP and upload the `notebooks/` folder

### 2. Run Configuration

Every notebook automatically loads shared settings from `_config.py`. Open it to review the catalog, schema, and endpoint values your instructor has set up.

### 3. Start with Notebook 00

Run `00_Setup_and_Data_Generation` first — it creates the shared dataset used by all other notebooks. Your instructor may have already run this for you; check with them before executing.

### 4. Follow Along

Work through the notebooks in order (01 → 02 → 03 → 04). Each notebook is self-contained with explanations, code, and exercises.

---

## Use Case: GTM Lead Scoring & Knowledge Assistant

All notebooks use a unified **Go-To-Market** scenario with synthetic Salesforce-style data:

| Table | Description | Rows |
|-------|-------------|------|
| `gtm_accounts` | Company firmographic data | ~2,000 |
| `gtm_contacts` | Leads and contacts | ~10,000 |
| `gtm_opportunities` | Sales pipeline deals | ~5,000 |
| `gtm_activities` | Engagement events (calls, emails, meetings) | ~50,000 |
| `gtm_campaigns` | Marketing campaigns | ~100 |
| `gtm_campaign_members` | Campaign engagement tracking | ~20,000 |
| `gtm_lead_scores` | Lead scoring with conversion labels | ~10,000 |
| `gtm_knowledge_base` | Product docs, sales playbooks, competitive intel | ~50 |

**Morning (ML):** Build a lead scoring model that predicts which contacts will convert, then deploy it to production with full MLOps governance.

**Afternoon (GenAI):** Build a GTM knowledge assistant that answers questions using RAG over product docs, queries live CRM data, and analyzes the sales pipeline — all as an AI agent.

---

## Notebook Descriptions

### `_config.py` — Shared Configuration
Central settings file. All catalog names, schema names, model endpoints, and credentials are defined here. Other notebooks inherit these via `%run ./_config`.

### `00_Setup_and_Data_Generation` — Environment Setup
Creates the Unity Catalog schema and generates all synthetic GTM datasets. Run this first (or confirm your instructor has already run it).

### `01_Spark_Architecture_Distributed_Training` — Module 1
Covers Spark internals (execution plans, partitioning, shuffles), then builds distributed ML pipelines with Spark ML, Pandas API on Spark, and Pandas UDFs. Introduces `pyspark.ml.connect` and `TorchDistributor`.

### `02_Advanced_MLOps_Production_Governance` — Module 2
Scales hyperparameter search with Hyperopt + SparkTrials, tracks experiments in MLflow, registers models to Unity Catalog, deploys to Model Serving, enables Inference Tables, and sets up drift monitoring.

### `03_GenAI_Foundations_Agent_Design` — Module 3
Explores Foundation Model APIs and prompt engineering, builds a Vector Search index over the knowledge base, creates agent tools for structured and unstructured retrieval, and introduces Agent Bricks and MCP.

### `04_Custom_Agents_Evaluation_Governance` — Module 4
Builds a complete GTM assistant agent with tool calling, wraps it as an MLflow model, covers AI Gateway governance, enables MLflow tracing, and evaluates agent quality with LLM-as-judge.

---

## What You Will Learn

By the end of this training, you will be able to:

- Train and tune ML models at scale using Spark ML and Hyperopt
- Track experiments, register models, and deploy endpoints using MLflow and Unity Catalog
- Monitor model quality with Inference Tables and Lakehouse Monitoring
- Call Foundation Models via the Databricks API and engineer effective prompts
- Build a RAG pipeline with Vector Search for knowledge retrieval
- Create AI agents with tool calling capabilities
- Evaluate agent quality using MLflow tracing and LLM-as-judge
- Understand AI Gateway for governance, cost control, and safety

---

## Certification Path

This training covers topics from two Databricks certifications:

- **Databricks Machine Learning Professional** — Spark ML, Hyperopt, MLflow, model serving, monitoring
- **Databricks Generative AI Engineer Associate** — Foundation Models, Vector Search, RAG, agent evaluation

After this session, explore self-paced preparation at [Databricks Academy](https://www.databricks.com/learn/training/catalog).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Table not found` errors | Run notebook `00` first to create the data |
| Slow model serving startup | Endpoints can take 5-10 minutes to provision — this is normal |
| `Rate limit exceeded` on LLM calls | Wait 30 seconds and retry; Foundation Model endpoints have rate limits |
| Import errors | Ensure you are running on serverless compute (not a classic cluster) |
| Vector Search index not ready | Index sync can take a few minutes after creation — check status in the notebook |
| `AnalysisException: Permission denied` | Ensure your user has `USE CATALOG`, `USE SCHEMA`, `SELECT`, `CREATE TABLE` on the training catalog |
| Notebook fails on DBR < 14.3 | Upgrade to DBR ML 15.4+ (serverless) or DBR ML 14.3+ (classic) — older runtimes lack required APIs |
| `pyspark.ml` not available on serverless | Expected — serverless restricts JVM-based Spark ML. Use scikit-learn instead (as shown in Module 1) |

If you encounter any other issues, ask your instructor for help.
