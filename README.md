# Databricks Training: Machine Learning at Scale and Foundations of Generative AI

A one-day, hands-on workshop covering distributed ML, production MLOps, GenAI foundations, and agentic AI on the Databricks platform. All modules use a shared Go-To-Market (GTM) dataset so that each exercise builds on the last, from raw data through a fully deployed AI agent.

---

## Agenda

| Time | Module | Notebook | Topics |
|------|--------|----------|--------|
| 9:30 am – 10:00 am | **Setup** | `00_Setup_and_Data_Generation` | Environment setup, synthetic GTM data generation |
| 10:00 am – 11:30 am | **Module 1** | `01_Spark_Architecture_Distributed_Training` | Spark execution plans, Spark ML pipelines, Pandas API on Spark, Pandas UDFs |
| 11:30 am – 11:45 am | *Break* | | |
| 11:45 am – 1:15 pm | **Module 2** | `02_Advanced_MLOps_Production_Governance` | Hyperopt tuning, MLflow tracking, Unity Catalog model registry, Model Serving, monitoring |
| 1:15 pm – 2:00 pm | *Lunch* | | |
| 2:00 pm – 3:30 pm | **Module 3** | `03_GenAI_Foundations_Agent_Design` | Foundation Model APIs, prompt engineering, Vector Search, agent tools, UC Functions, Agent Bricks & MCP |
| 3:30 pm – 3:45 pm | *Break* | | |
| 3:45 pm – 5:00 pm | **Module 4** | `04_Custom_Agents_Evaluation_Governance` | Custom agents (OpenAI SDK), MCP tool discovery, MLflow ResponsesAgent, agents.deploy(), AI Gateway, tracing, mlflow.genai.evaluate |
| 5:00 pm – 5:15 pm | **Wrap Up** | | Review, Q&A, next steps |

---

## Prerequisites

Before the training, please ensure the following:

1. **Databricks workspace access** -- You must be able to log into the workspace URL provided by your instructor
2. **Python proficiency** -- Comfortable with Python, pandas, and basic SQL
3. **ML fundamentals** -- Familiarity with classification, train/test splits, and evaluation metrics
4. **LLM basics** -- High-level understanding of what large language models are

Your instructor will handle compute provisioning and Unity Catalog permissions ahead of time. The workshop uses **serverless compute** (DBR ML 15.4+), which requires no cluster configuration on your part.

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

## Repository Structure

```
servicenow-databricks-training/
  notebooks/
    _config.py                                  # Shared configuration (catalog, schema, endpoints)
    00_Setup_and_Data_Generation.py             # Data setup
    01_Spark_Architecture_Distributed_Training.py
    02_Advanced_MLOps_Production_Governance.py
    03_GenAI_Foundations_Agent_Design.py
    04_Custom_Agents_Evaluation_Governance.py
  README.md
```

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
Central settings file defining the catalog, schema, model endpoints, and Vector Search endpoint used across all notebooks. Other notebooks inherit these values via `%run ./_config`. Your instructor will confirm the correct settings before the session.

### `00_Setup_and_Data_Generation` — Environment Setup
Creates the Unity Catalog schema and generates all synthetic GTM datasets (accounts, contacts, opportunities, activities, campaigns, lead scores, and a knowledge base). Run this first, or confirm your instructor has already run it. Takes approximately 2-3 minutes.

### `01_Spark_Architecture_Distributed_Training` — Module 1
Covers Spark internals (execution plans, partitioning, shuffles, broadcast vs. sort-merge joins) and then builds distributed ML pipelines using Spark ML, the Pandas API on Spark, and Pandas UDFs. Includes a hands-on performance comparison between pandas, Spark, and Photon at realistic data volumes. Also introduces `pyspark.ml.connect` and `TorchDistributor` for deep learning workloads.

### `02_Advanced_MLOps_Production_Governance` — Module 2
Scales hyperparameter tuning with Hyperopt (and compares it to Optuna), tracks experiments in MLflow, registers models to Unity Catalog, deploys a real-time Model Serving endpoint, and configures A/B testing with Inference Tables. Covers the full path from experimentation to production monitoring.

### `03_GenAI_Foundations_Agent_Design` — Module 3
Introduces Foundation Model APIs and prompt engineering patterns (zero-shot, few-shot, chain-of-thought). Builds a Vector Search index over the GTM knowledge base for retrieval-augmented generation (RAG). Creates agent tools as both Python functions and Unity Catalog SQL functions (governed and MCP-discoverable). Introduces the Agent Bricks framework and demonstrates live MCP tool discovery from Unity Catalog.

### `04_Custom_Agents_Evaluation_Governance` — Module 4
Builds a custom GTM assistant agent with a manual tool-calling loop (educational), then upgrades to MCP-based tool discovery and the `MCPToolCallingAgent(ResponsesAgent)` pattern for production packaging. Deploys the agent to a serving endpoint via `agents.deploy()`. Covers AI Gateway for governance and guardrails, MLflow 3.0 tracing for observability, and evaluates agent quality with both `mlflow.genai.evaluate()` (built-in scorers) and custom LLM-as-Judge scoring.

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

After this session, explore self-paced preparation at [Databricks Academy](https://www.databricks.com/learn/certification).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Table not found` errors | Run notebook `00` first to create the data |
| Slow model serving startup | Endpoints can take 5-10 minutes to provision — this is normal |
| `Rate limit exceeded` on LLM calls | Wait 30 seconds and retry; Foundation Model endpoints have rate limits |
| `ModuleNotFoundError` for a library | Serverless includes most ML/AI libraries. If a library is missing, add `%pip install <package>` at the top of the notebook and restart the Python environment |
| Vector Search index not ready | Index sync can take a few minutes after creation — check status in the notebook |
| `AnalysisException: Permission denied` | Ensure your user has `USE CATALOG`, `USE SCHEMA`, `SELECT`, `CREATE TABLE` on the training catalog |
| Notebook fails on DBR < 14.3 | Upgrade to DBR ML 15.4+ (serverless) or DBR ML 14.3+ (classic) — older runtimes lack required APIs |
| `pyspark.ml` not available on serverless | Expected — serverless restricts JVM-based Spark ML. Use scikit-learn instead (as shown in Module 1) |
| `SparkTrials` fails on serverless | Serverless does not support `SparkTrials`. Use `hyperopt.Trials()` instead (single-node tuning) |

If you encounter any other issues, ask your instructor for help.
