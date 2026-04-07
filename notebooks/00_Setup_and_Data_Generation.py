# Databricks notebook source
# DBTITLE 1,Module Overview and Scenario
# MAGIC %md
# MAGIC # Module 0: Environment Setup & GTM Data Generation
# MAGIC
# MAGIC **Training Event:** ServiceNow x Databricks — AI/ML on Lakehouse
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Scenario
# MAGIC
# MAGIC You are a data science team at a B2B SaaS company. Your Go-To-Market (GTM) organization wants to
# MAGIC improve lead conversion rates by building an **AI-powered lead scoring system**. The sales team
# MAGIC currently relies on gut instinct and static rules — you will replace that with ML models trained
# MAGIC on historical CRM data and augment the workflow with Generative AI.
# MAGIC
# MAGIC This notebook sets up the **Unity Catalog** environment (**Unity Catalog** is Databricks' centralized governance layer -- it manages access control, data lineage, and audit logging across all data assets) and generates realistic Salesforce-style
# MAGIC GTM data that we will use throughout the remaining modules:
# MAGIC
# MAGIC | Table | Description | Approx Rows |
# MAGIC |-------|-------------|-------------|
# MAGIC | `gtm_accounts` | Company accounts | 2,000 |
# MAGIC | `gtm_contacts` | Individual contacts / leads | 10,000 |
# MAGIC | `gtm_opportunities` | Sales opportunities | 5,000 |
# MAGIC | `gtm_activities` | Emails, calls, meetings, demos | 50,000 |
# MAGIC | `gtm_campaigns` | Marketing campaigns | 100 |
# MAGIC | `gtm_campaign_members` | Campaign engagement records | 20,000 |
# MAGIC | `gtm_lead_scores` | Composite lead scores + conversion label | 10,000 |
# MAGIC | `gtm_knowledge_base` | Product docs, playbooks, competitive intel | 50 |
# MAGIC
# MAGIC **Compute:** Serverless (serverless compute auto-provisions and auto-scales infrastructure -- no cluster configuration needed)
# MAGIC
# MAGIC **Estimated Runtime:** ~3 minutes

# COMMAND ----------

# DBTITLE 1,Configuration Instructions
# MAGIC %md
# MAGIC ## 0.1 — Configuration
# MAGIC
# MAGIC The `_config` notebook sets shared variables used across all modules: **catalog** name, **schema** name,
# MAGIC API tokens, and your username. Running it first ensures every notebook operates in the same workspace context.

# COMMAND ----------

# DBTITLE 1,Load Shared Configuration
# MAGIC %run ./_config

# COMMAND ----------

# DBTITLE 1,Unity Catalog Namespace
# MAGIC %md
# MAGIC ## 0.2 — Create Your Training Schema
# MAGIC
# MAGIC Databricks organizes data in a **three-level namespace**: `catalog.schema.table`.
# MAGIC - **Catalog** -- top-level container (like a database server)
# MAGIC - **Schema** -- logical grouping within a catalog (like a database)
# MAGIC - **Table** -- the actual data asset
# MAGIC
# MAGIC We create a dedicated schema for this training so all generated tables are isolated from production data.

# COMMAND ----------

# DBTITLE 1,Create Training Schema
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE SCHEMA {schema}")
print(f"Active catalog/schema: {catalog}.{schema}")

# COMMAND ----------

# DBTITLE 1,Imports
# MAGIC %md
# MAGIC ## 0.3 — Imports

# COMMAND ----------

# DBTITLE 1,Import Libraries and Define Helpers
import numpy as np
import pandas as pd
from datetime import date, timedelta

np.random.seed(42)

def uid_array(n):
    """Generate n short unique IDs."""
    return [f"{x:012x}" for x in np.random.randint(0, 2**48, size=n)]

def random_dates(start, end, n):
    """Generate n random dates between start and end."""
    start_ord = start.toordinal()
    end_ord = end.toordinal()
    days = np.random.randint(start_ord, end_ord + 1, size=n)
    return [date.fromordinal(d) for d in days]

# COMMAND ----------

# DBTITLE 1,Reference Data Overview
# MAGIC %md
# MAGIC ## 0.4 — Reference Data
# MAGIC
# MAGIC Now we'll build the synthetic dataset. The order matters: **accounts** (companies) are the root entity,
# MAGIC **contacts** belong to accounts, **opportunities** (deals) link to accounts, and **activities** (emails, calls, meetings)
# MAGIC connect to contacts. This mirrors a real CRM like Salesforce.

# COMMAND ----------

# DBTITLE 1,Define Reference Data Constants
PREFIXES = [
    "Apex", "Nova", "Quantum", "Vertex", "Horizon", "Summit", "Atlas",
    "Pinnacle", "Catalyst", "Meridian", "Nexus", "Vanguard", "Stellar",
    "Zenith", "Forge", "Prism", "Axiom", "Titan", "Elevate", "Kinetic",
    "Stratos", "Helix", "Omni", "Radiant", "Beacon", "Crest", "Dynamo",
    "Fusion", "Insight", "Lumen", "Orbital", "Pulse", "Sapient", "Trident",
    "Unity", "Vector", "Wave", "Zephyr", "Agile", "Bold", "Clarity",
    "Delta", "Echo", "Falcon", "Granite", "Ivy", "Jade", "Keystone"
]

SUFFIXES = [
    "Digital Solutions", "Industries", "Technologies", "Systems", "Analytics",
    "Consulting", "Global", "Partners", "Corp", "Labs", "Networks",
    "Innovations", "Group", "Platforms", "Services", "Dynamics", "Ventures",
    "Logic", "Data", "Cloud", "Software", "AI", "Security", "Health",
    "Financial", "Media", "Energy", "Robotics", "Biotech", "Aerospace",
    "Logistics", "Commerce", "Intelligence", "Works", "Hub", "Bridge",
    "Matrix", "Realm", "Core", "Link"
]

INDUSTRIES = [
    "Technology", "Healthcare", "Financial Services", "Manufacturing",
    "Retail", "Media & Entertainment", "Energy", "Education",
    "Government", "Telecommunications"
]

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
REGION_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

COUNTRIES = {
    "North America": ["United States", "Canada", "Mexico"],
    "EMEA": ["United Kingdom", "Germany", "France", "Netherlands", "Sweden"],
    "APAC": ["India", "Japan", "Australia", "Singapore", "South Korea"],
    "LATAM": ["Brazil", "Argentina", "Chile", "Colombia"]
}

TIERS = ["Enterprise", "Mid-Market", "SMB"]
TIER_WEIGHTS = [0.20, 0.35, 0.45]

DEPARTMENTS = ["Sales", "Marketing", "Engineering", "Product", "Finance", "Operations", "IT", "HR", "Legal", "Executive"]
SENIORITY = ["C-Level", "VP", "Director", "Manager", "Individual Contributor"]
SENIORITY_WEIGHTS = [0.05, 0.10, 0.15, 0.30, 0.40]

LEAD_SOURCES = ["Webinar", "Content Download", "Demo Request", "Referral", "Event", "Organic", "Paid Search"]
STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
ACTIVITY_TYPES = ["Email", "Call", "Meeting", "Demo", "Webinar", "Content Download"]
OUTCOMES = ["Positive", "Neutral", "Negative"]

FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth",
               "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
               "Raj", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Sanjay", "Deepa", "Arjun", "Meera",
               "Wei", "Yuki", "Hans", "Sophie", "Carlos", "Maria", "Ahmed", "Fatima", "Liam", "Emma"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Martinez", "Anderson",
              "Taylor", "Thomas", "Jackson", "White", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young",
              "Kumar", "Sharma", "Patel", "Singh", "Das", "Gupta", "Mehta", "Shah", "Joshi", "Reddy",
              "Chen", "Wang", "Kim", "Tanaka", "Mueller", "Dubois", "Santos", "Ali", "O'Brien", "Wilson"]

PRODUCT_LINES = ["Platform Pro", "Data Analytics Suite", "AI Accelerator", "Cloud Migration", "Security Shield", "Integration Hub"]

# COMMAND ----------

# DBTITLE 1,Accounts Table Description
# MAGIC %md
# MAGIC ## 0.5 — Generate Accounts (2,000 rows)
# MAGIC
# MAGIC Accounts are the root entity in our data model -- every contact, opportunity, and activity traces back to one.
# MAGIC Fields like `industry`, `employee_count`, and `account_tier` will become key ML features for predicting deal outcomes.
# MAGIC
# MAGIC All tables are saved in **Delta Lake** format (Delta Lake is an open-source storage format that adds reliability features like ACID transactions and time travel to data lakes).

# COMMAND ----------

# DBTITLE 1,Generate 2,000 Account Records
n_accounts = 2000

# Generate unique company names (prefixes x suffixes > 2000 needed)
all_names = [f"{p} {s}" for p in PREFIXES for s in SUFFIXES]
# Add numbered variants if we need more
while len(all_names) < n_accounts:
    all_names.append(f"{np.random.choice(PREFIXES)} {np.random.choice(SUFFIXES)} {len(all_names)}")
np.random.shuffle(all_names)
company_names = all_names[:n_accounts]

tiers = np.random.choice(TIERS, size=n_accounts, p=TIER_WEIGHTS)
regions = np.random.choice(REGIONS, size=n_accounts, p=REGION_WEIGHTS)
countries = [np.random.choice(COUNTRIES[r]) for r in regions]
industries = np.random.choice(INDUSTRIES, size=n_accounts)

employees = np.where(tiers == "Enterprise", np.random.randint(5000, 150000, n_accounts),
            np.where(tiers == "Mid-Market", np.random.randint(500, 5000, n_accounts),
                     np.random.randint(20, 500, n_accounts)))

revenue = np.where(tiers == "Enterprise", np.random.uniform(5e8, 5e10, n_accounts),
          np.where(tiers == "Mid-Market", np.random.uniform(5e7, 5e8, n_accounts),
                   np.random.uniform(1e6, 5e7, n_accounts)))

df_accounts = pd.DataFrame({
    "account_id": uid_array(n_accounts),
    "company_name": company_names,
    "industry": industries,
    "employee_count": employees.astype(int),
    "annual_revenue": np.round(revenue, 2),
    "region": regions,
    "country": countries,
    "account_tier": tiers,
    "website": [f"https://www.{n.lower().replace(' ', '')}.com" for n in company_names],
    "created_date": random_dates(date(2024, 1, 1), date(2025, 12, 31), n_accounts),
})

spark.createDataFrame(df_accounts).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.gtm_accounts")
print(f"gtm_accounts: {n_accounts} rows written")

# COMMAND ----------

# DBTITLE 1,Contacts Table Description
# MAGIC %md
# MAGIC ## 0.6 — Generate Contacts (10,000 rows)
# MAGIC
# MAGIC Each contact has a `lead_source` and `seniority_level` -- these become important ML features later.
# MAGIC For example, a "Demo Request" from a "VP" converts at a much higher rate than an "Organic" visit from an "Individual Contributor."

# COMMAND ----------

# DBTITLE 1,Generate 10,000 Contact Records
n_contacts = 10000
account_ids = df_accounts["account_id"].values

df_contacts = pd.DataFrame({
    "contact_id": uid_array(n_contacts),
    "account_id": np.random.choice(account_ids, size=n_contacts),
    "first_name": np.random.choice(FIRST_NAMES, size=n_contacts),
    "last_name": np.random.choice(LAST_NAMES, size=n_contacts),
    "title": np.random.choice(["VP Sales", "Director Marketing", "Sr. Engineer", "Product Manager", "Data Scientist",
                                "CTO", "CMO", "CFO", "Account Executive", "Solutions Architect",
                                "ML Engineer", "DevOps Lead", "Analytics Manager", "IT Director", "Platform Lead"], size=n_contacts),
    "department": np.random.choice(DEPARTMENTS, size=n_contacts),
    "seniority_level": np.random.choice(SENIORITY, size=n_contacts, p=SENIORITY_WEIGHTS),
    "lead_source": np.random.choice(LEAD_SOURCES, size=n_contacts),
    "created_date": random_dates(date(2024, 1, 1), date(2026, 3, 1), n_contacts),
})
df_contacts["email"] = df_contacts["first_name"].str.lower() + "." + df_contacts["last_name"].str.lower() + "@" + np.random.choice(["company.com", "corp.io", "business.net", "org.co", "tech.dev"], size=n_contacts)

spark.createDataFrame(df_contacts).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.gtm_contacts")
print(f"gtm_contacts: {n_contacts} rows written")

# COMMAND ----------

# DBTITLE 1,Opportunities Section
# MAGIC %md
# MAGIC ## 0.7 — Generate Opportunities (5,000 rows)

# COMMAND ----------

# DBTITLE 1,Generate 5,000 Opportunity Records
n_opps = 5000
contact_ids = df_contacts["contact_id"].values

stages = np.random.choice(STAGES, size=n_opps, p=[0.20, 0.20, 0.15, 0.15, 0.18, 0.12])
probs = {"Prospecting": 0.10, "Qualification": 0.25, "Proposal": 0.50, "Negotiation": 0.70, "Closed Won": 1.0, "Closed Lost": 0.0}
amounts = np.where(stages == "Closed Lost", np.random.uniform(10000, 500000, n_opps),
                   np.random.uniform(25000, 2000000, n_opps))

df_opps = pd.DataFrame({
    "opportunity_id": uid_array(n_opps),
    "account_id": np.random.choice(account_ids, size=n_opps),
    "contact_id": np.random.choice(contact_ids, size=n_opps),
    "opportunity_name": [f"Opp-{i+1:04d}" for i in range(n_opps)],
    "stage": stages,
    "amount": np.round(amounts, 2),
    "probability": [probs[s] for s in stages],
    "close_date": random_dates(date(2025, 1, 1), date(2026, 12, 31), n_opps),
    "created_date": random_dates(date(2024, 6, 1), date(2026, 3, 1), n_opps),
    "product_line": np.random.choice(PRODUCT_LINES, size=n_opps),
    "lead_source": np.random.choice(LEAD_SOURCES, size=n_opps),
})

spark.createDataFrame(df_opps).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.gtm_opportunities")
print(f"gtm_opportunities: {n_opps} rows written")

# COMMAND ----------

# DBTITLE 1,Activities Section
# MAGIC %md
# MAGIC ## 0.8 — Generate Activities (50,000 rows)

# COMMAND ----------

# DBTITLE 1,Generate 50,000 Activity Records
n_activities = 50000
opp_ids = df_opps["opportunity_id"].values

act_types = np.random.choice(ACTIVITY_TYPES, size=n_activities, p=[0.30, 0.25, 0.15, 0.10, 0.10, 0.10])
outcomes = np.random.choice(OUTCOMES, size=n_activities, p=[0.40, 0.35, 0.25])
durations = np.where(act_types == "Email", np.random.randint(1, 10, n_activities),
            np.where(act_types == "Call", np.random.randint(5, 45, n_activities),
            np.where(act_types == "Meeting", np.random.randint(30, 120, n_activities),
            np.where(act_types == "Demo", np.random.randint(30, 90, n_activities),
            np.where(act_types == "Webinar", np.random.randint(45, 90, n_activities),
                     np.random.randint(1, 5, n_activities))))))

sentiment = np.where(outcomes == "Positive", np.random.uniform(0.6, 1.0, n_activities),
            np.where(outcomes == "Neutral", np.random.uniform(0.3, 0.6, n_activities),
                     np.random.uniform(0.0, 0.3, n_activities)))

df_activities = pd.DataFrame({
    "activity_id": uid_array(n_activities),
    "contact_id": np.random.choice(contact_ids, size=n_activities),
    "opportunity_id": np.random.choice(opp_ids, size=n_activities),
    "activity_type": act_types,
    "activity_date": random_dates(date(2024, 6, 1), date(2026, 3, 1), n_activities),
    "duration_minutes": durations.astype(int),
    "outcome": outcomes,
    "sentiment_score": np.round(sentiment, 3),
})

spark.createDataFrame(df_activities).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.gtm_activities")
print(f"gtm_activities: {n_activities} rows written")

# COMMAND ----------

# DBTITLE 1,Campaigns Section
# MAGIC %md
# MAGIC ## 0.9 — Generate Campaigns (100 rows)

# COMMAND ----------

# DBTITLE 1,Generate 100 Campaign Records
n_campaigns = 100
campaign_types = ["Email Nurture", "Webinar", "Event", "Content Syndication", "Paid Search", "Social"]
channels = ["Email", "Web", "Social", "Events", "Search", "Partner"]

df_campaigns = pd.DataFrame({
    "campaign_id": uid_array(n_campaigns),
    "campaign_name": [f"Campaign-{t}-Q{q}-{y}" for t, q, y in zip(
        np.random.choice(["Launch", "Nurture", "Awareness", "Conversion", "Retention"], n_campaigns),
        np.random.choice([1, 2, 3, 4], n_campaigns),
        np.random.choice([2024, 2025, 2026], n_campaigns))],
    "campaign_type": np.random.choice(campaign_types, size=n_campaigns),
    "channel": np.random.choice(channels, size=n_campaigns),
    "start_date": random_dates(date(2024, 1, 1), date(2026, 1, 1), n_campaigns),
    "end_date": random_dates(date(2024, 4, 1), date(2026, 6, 1), n_campaigns),
    "budget": np.round(np.random.uniform(5000, 250000, n_campaigns), 2),
    "actual_spend": np.round(np.random.uniform(3000, 200000, n_campaigns), 2),
    "status": np.random.choice(["Active", "Completed", "Planned"], size=n_campaigns, p=[0.30, 0.50, 0.20]),
})

spark.createDataFrame(df_campaigns).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.gtm_campaigns")
print(f"gtm_campaigns: {n_campaigns} rows written")

# COMMAND ----------

# DBTITLE 1,Campaign Members Section
# MAGIC %md
# MAGIC ## 0.10 — Generate Campaign Members (20,000 rows)

# COMMAND ----------

# DBTITLE 1,Generate 20,000 Campaign Member Records
n_members = 20000
campaign_ids = df_campaigns["campaign_id"].values
statuses = ["Sent", "Opened", "Clicked", "Responded", "Converted"]

member_status = np.random.choice(statuses, size=n_members, p=[0.30, 0.25, 0.20, 0.15, 0.10])
opens = np.where(member_status == "Sent", 0,
        np.where(member_status == "Opened", np.random.randint(1, 5, n_members),
                 np.random.randint(2, 10, n_members)))
clicks = np.where(np.isin(member_status, ["Sent", "Opened"]), 0, np.random.randint(1, 8, n_members))
forms = np.where(np.isin(member_status, ["Responded", "Converted"]), np.random.randint(1, 4, n_members), 0)

df_members = pd.DataFrame({
    "member_id": uid_array(n_members),
    "campaign_id": np.random.choice(campaign_ids, size=n_members),
    "contact_id": np.random.choice(contact_ids, size=n_members),
    "status": member_status,
    "first_interaction_date": random_dates(date(2024, 1, 1), date(2026, 2, 1), n_members),
    "last_interaction_date": random_dates(date(2024, 6, 1), date(2026, 3, 1), n_members),
    "email_opens": opens.astype(int),
    "email_clicks": clicks.astype(int),
    "form_fills": forms.astype(int),
})

spark.createDataFrame(df_members).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.gtm_campaign_members")
print(f"gtm_campaign_members: {n_members} rows written")

# COMMAND ----------

# DBTITLE 1,Lead Scores Table Description
# MAGIC %md
# MAGIC ## 0.11 — Generate Lead Scores (10,000 rows — one per contact)
# MAGIC
# MAGIC The `converted` column is the **target variable** for our ML lead scoring model.
# MAGIC Approximately 30% of contacts convert, with conversion correlated to engagement and fit scores.

# COMMAND ----------

# DBTITLE 1,Generate 10,000 Lead Score Records
n_leads = n_contacts  # one score per contact

engagement = np.random.uniform(0, 100, n_leads)
fit = np.random.uniform(0, 100, n_leads)
behavior = np.random.uniform(0, 100, n_leads)
recency = np.random.uniform(0, 100, n_leads)
total = 0.30 * engagement + 0.25 * fit + 0.25 * behavior + 0.20 * recency

# Conversion probability correlated with total score
conv_prob = 1 / (1 + np.exp(-(total - 55) / 10))  # sigmoid centered at 55
converted = (np.random.random(n_leads) < conv_prob).astype(int)

days_to_convert = np.where(converted == 1, np.random.randint(7, 180, n_leads), 0)
conv_dates = [date(2025, 6, 1) + timedelta(days=int(d)) if c == 1 else None
              for c, d in zip(converted, np.random.randint(0, 365, n_leads))]

df_leads = pd.DataFrame({
    "contact_id": df_contacts["contact_id"].values,
    "engagement_score": np.round(engagement, 2),
    "fit_score": np.round(fit, 2),
    "behavior_score": np.round(behavior, 2),
    "recency_score": np.round(recency, 2),
    "total_score": np.round(total, 2),
    "converted": converted,
    "conversion_date": conv_dates,
    "days_to_convert": days_to_convert.astype(int),
})

spark.createDataFrame(df_leads).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.gtm_lead_scores")
conv_rate = converted.mean()
print(f"gtm_lead_scores: {n_leads} rows written (conversion rate: {conv_rate:.1%})")

# COMMAND ----------

# DBTITLE 1,Knowledge Base Description
# MAGIC %md
# MAGIC ## 0.12 — Generate Knowledge Base (50 documents)
# MAGIC
# MAGIC These documents will be used for **RAG (Retrieval-Augmented Generation)** in the afternoon session.
# MAGIC RAG lets an AI agent answer questions using your company's own documents rather than relying solely on the LLM's training data.
# MAGIC We generate 50 representative docs (product sheets, playbooks, competitive intel, FAQs) so the agent has realistic content to retrieve from.

# COMMAND ----------

knowledge_docs = [
    # Products
    ("Platform Pro Overview", "Product", "Core Platform",
     "Platform Pro is our flagship enterprise data platform that unifies data engineering, analytics, and AI on a single lakehouse architecture. It supports batch and streaming workloads with Delta Lake as the storage layer, providing ACID transactions, time travel, and schema enforcement. Platform Pro integrates with Unity Catalog for centralized governance across all data assets including tables, ML models, and dashboards. Key capabilities include serverless SQL warehouses for instant-on analytics, automated ETL pipelines with declarative orchestration, and built-in collaboration through shared notebooks. Enterprise customers typically see a 40-60% reduction in data infrastructure costs after migrating to Platform Pro compared to legacy data warehouse solutions."),

    ("AI Accelerator Suite", "Product", "AI/ML",
     "The AI Accelerator Suite provides a complete toolkit for building, training, and deploying machine learning models at scale. It includes AutoML for rapid prototyping, distributed training with GPU clusters, hyperparameter tuning with Hyperopt, and MLflow integration for full experiment lifecycle management. The suite supports popular frameworks including TensorFlow, PyTorch, scikit-learn, and XGBoost. Model serving endpoints provide real-time and batch inference with automatic scaling. Unity Catalog integration enables model versioning, lineage tracking, and access control. Feature Store support allows teams to share and reuse engineered features across projects, reducing duplication and ensuring consistency."),

    ("Data Analytics Suite", "Product", "Analytics",
     "The Data Analytics Suite empowers business analysts and data teams with self-service analytics powered by a lakehouse architecture. It features serverless SQL warehouses that auto-scale to meet query demand, AI-powered dashboards with natural language query capabilities, and integration with popular BI tools like Tableau and Power BI. The suite includes Genie Spaces for conversational data exploration, allowing non-technical users to ask questions in plain English. Built-in data quality monitoring detects anomalies and schema drift automatically. Materialized views and query result caching ensure sub-second response times for frequently accessed reports."),

    ("Security Shield", "Product", "Security",
     "Security Shield is our enterprise security and compliance layer that protects data assets across the entire lakehouse. It provides attribute-based access control (ABAC), row-level and column-level security, dynamic data masking, and comprehensive audit logging. Security Shield integrates with enterprise identity providers via SCIM and SAML for automated user provisioning. Data loss prevention features include sensitivity classification, encryption at rest and in transit, and private link connectivity. The solution supports compliance frameworks including SOC 2 Type II, HIPAA, FedRAMP, PCI-DSS, and GDPR with built-in compliance dashboards and automated evidence collection."),

    ("Integration Hub", "Product", "Integration",
     "Integration Hub connects your lakehouse to hundreds of data sources and destinations with pre-built connectors and low-code ingestion pipelines. Supported sources include Salesforce, SAP, Oracle, Snowflake, MongoDB, Kafka, and major cloud storage services. Delta Live Tables provides declarative ETL with automatic data quality checks, retry logic, and pipeline observability. Change Data Capture (CDC) support enables real-time replication from operational databases. The hub includes data transformation capabilities with both SQL and Python, schema evolution handling, and automatic data type inference. Federation queries allow querying external databases without data movement."),

    ("Cloud Migration Toolkit", "Product", "Migration",
     "The Cloud Migration Toolkit accelerates data platform modernization with automated assessment, migration, and validation tools. It supports migration from legacy warehouses including Teradata, Oracle, Netezza, and Hadoop to the lakehouse architecture. The toolkit includes SQL translation engines that automatically convert proprietary SQL dialects, workload profiling to right-size compute resources, and parallel data transfer utilities that minimize migration downtime. Validation frameworks compare source and target query results to ensure data integrity. Post-migration optimization recommendations help customers achieve better performance at lower cost."),

    # Sales Playbooks
    ("Enterprise Sales Methodology", "Sales Playbook", "Methodology",
     "Our enterprise sales methodology follows a consultative approach focused on understanding the customer's data and AI maturity. The process begins with a Discovery Workshop where we assess current architecture, pain points, and strategic objectives. We then develop a joint success plan with clear milestones and measurable outcomes. Key selling motions include Total Cost of Ownership analysis comparing current stack costs, proof-of-value engagements limited to 2-4 weeks, and executive briefings that align our platform capabilities with their digital transformation roadmap. Champions are typically VP of Data Engineering or Chief Data Officer, while economic buyers are CIO or CTO. Average enterprise deal size is $500K-$2M ARR with 6-9 month sales cycles."),

    ("Objection Handling Guide", "Sales Playbook", "Objections",
     "Common objections and recommended responses: (1) 'We already have Snowflake' — Position as complementary for AI/ML workloads where Snowflake lacks native support; highlight unified governance and the cost savings of not moving data between systems. (2) 'Too expensive' — Focus on TCO analysis including hidden costs of their current multi-tool stack; demonstrate serverless auto-scaling that eliminates over-provisioning. (3) 'Our team doesn't know Spark' — Emphasize SQL-first experience with serverless SQL warehouses; highlight AI-assisted coding and the Pandas API on Spark for Python users. (4) 'Security concerns' — Reference SOC 2 Type II certification, private link, customer-managed keys, and Fortune 500 customer references in their industry. (5) 'Locked into cloud provider' — Highlight multi-cloud support and data portability with open formats (Delta Lake, Parquet)."),

    ("Competitive Positioning vs Snowflake", "Sales Playbook", "Competitive",
     "Key differentiators against Snowflake: (1) Unified platform for data engineering, analytics, AND AI/ML — Snowflake requires external tools for ML training and deployment. (2) Open source foundation with Delta Lake and MLflow — avoids vendor lock-in and leverages community innovation. (3) Native GPU cluster support for deep learning and LLM fine-tuning — Snowflake has no equivalent capability. (4) Real-time streaming with Structured Streaming — Snowflake's Snowpipe Streaming is limited in comparison. (5) Cost efficiency for mixed workloads — lakehouse architecture eliminates data copies between warehouse and lake. (6) AI/BI with Genie — natural language querying that's tightly integrated with governance. When competing, focus on AI/ML use cases where the gap is widest."),

    ("Proof of Value Framework", "Sales Playbook", "POV",
     "A successful Proof of Value (POV) engagement follows a structured 3-phase approach over 2-4 weeks. Phase 1 (Week 1): Environment setup, data ingestion of 2-3 representative datasets, baseline query performance benchmarks. Phase 2 (Weeks 2-3): Execute 3-5 priority use cases including ETL pipeline development, interactive analytics, and at least one ML model. Phase 3 (Week 4): Results presentation with performance comparisons, TCO projections, and production migration roadmap. Success criteria must be defined upfront with the customer: query performance targets, data freshness SLAs, model accuracy thresholds, and cost per workload hour. Always secure executive sponsor commitment before starting. Document quick wins early and share weekly progress reports."),

    ("Land and Expand Strategy", "Sales Playbook", "Growth",
     "The land and expand strategy focuses on establishing an initial foothold and growing through demonstrated value. Landing motions: target a single team with a well-defined pain point (slow queries, manual ETL, model deployment friction). Start with a 3-month pilot limited to $50K-$100K. Expansion triggers: (1) Additional teams requesting access after seeing initial results. (2) New use cases in AI/ML after establishing data engineering foundation. (3) Governance needs that require Unity Catalog enterprise features. (4) Real-time requirements that unlock streaming workloads. Typical expansion path: Data Engineering → Analytics → ML/AI → GenAI/Agents. Track consumption metrics monthly and identify expansion opportunities before renewal conversations."),

    # Competitive Intel
    ("Competitive Analysis: Cloud Data Platforms 2025", "Competitive", "Market Overview",
     "The cloud data platform market is consolidating around three main architectural approaches: (1) Cloud data warehouses (Snowflake, BigQuery, Redshift) focusing on SQL analytics and expanding into AI/ML. (2) Lakehouse platforms (Databricks) unifying data engineering, analytics, and AI on open formats. (3) Multi-modal databases (MongoDB Atlas, Cosmos DB) serving operational and analytical workloads. Key market trends: convergence of analytics and AI workloads driving demand for unified platforms, growing importance of data governance with AI regulation, increasing adoption of streaming for real-time decision making, and emergence of AI agents as a new workload category. Databricks leads in the AI/ML segment while competing vigorously in the analytics segment."),

    ("Snowflake Feature Comparison", "Competitive", "Snowflake",
     "Head-to-head feature comparison with Snowflake as of Q1 2026: Databricks leads in ML/AI (native GPU, distributed training, MLflow), data engineering (Structured Streaming, Delta Live Tables, Lakeflow), and open source (Delta Lake, MLflow, Unity Catalog open APIs). Snowflake leads in ease-of-use for SQL analysts (simpler interface, broader BI tool integration). Both platforms are competitive in: query performance for SQL analytics, data governance and access control, and marketplace/data sharing. Snowflake's recent AI features (Cortex, Snowpark ML) are less mature than Databricks' equivalent offerings. The gap is closing in serverless SQL but widening in GenAI agent capabilities."),

    # Technical Docs
    ("MLflow Best Practices", "Technical", "MLOps",
     "Best practices for MLflow on the lakehouse: (1) Use Unity Catalog as the model registry for centralized governance — set mlflow.set_registry_uri('databricks-uc'). (2) Log all experiments with structured parameters and metrics for reproducibility. (3) Use model aliases ('champion', 'challenger') instead of stage transitions for deployment management. (4) Enable inference tables on serving endpoints to automatically capture predictions for monitoring. (5) Implement model signatures to validate input/output schemas. (6) Use MLflow Evaluate for automated model quality assessment before promotion. (7) Set up Lakehouse Monitoring on inference tables for drift detection. (8) Version your training code alongside model artifacts. (9) Use feature tables from Unity Catalog for consistent feature engineering."),

    ("Unity Catalog Governance Guide", "Technical", "Governance",
     "Unity Catalog provides three-level namespace governance: catalog.schema.asset. Best practices: (1) Create separate catalogs for dev, staging, and production environments. (2) Use information_schema for auditing data access and lineage. (3) Implement row-level security with row filters and column masks for sensitive data. (4) Use managed tables for data that should be fully governed; external tables for data that needs to remain in customer-controlled storage. (5) Set up metastore-level IP access lists for network security. (6) Enable system tables for usage monitoring, billing analysis, and audit logs. (7) Use tags for data classification (PII, confidential, public). (8) Configure lineage tracking to trace data flow across notebooks, jobs, and dashboards."),

    ("Vector Search Architecture", "Technical", "GenAI",
     "Databricks Vector Search enables similarity search over embeddings stored in Delta tables. Architecture options: (1) Delta Sync Index — automatically syncs with a source Delta table, updating embeddings as data changes. Recommended for most use cases. (2) Direct Vector Access Index — for manual embedding management with push-based updates. Key design decisions: choose embedding dimension based on model (e.g., 1024 for GTE-Large), configure similarity metric (cosine for text, L2 for images), and set appropriate num_results for retrieval. For RAG applications, combine vector search with metadata filtering to improve relevance. Chunk documents to 512-1024 tokens for optimal retrieval granularity. Use endpoint auto-scaling to handle query spikes."),

    ("Agent Framework Getting Started", "Technical", "Agents",
     "Building AI agents on Databricks: (1) Start in AI Playground to prototype tool-calling agents with no code. (2) Graduate to code with the OpenAI SDK using Databricks Foundation Model endpoints as the backend. (3) Define tools as Python functions that query Delta tables, call Vector Search, execute SQL, or call external APIs. (4) Use MLflow to log, version, and trace agent execution. (5) Deploy agents to Model Serving endpoints with automatic scaling. (6) Evaluate agent quality with MLflow Evaluate using LLM-as-judge metrics (faithfulness, relevance, groundedness). (7) Add guardrails through AI Gateway for content filtering and rate limiting. (8) Monitor agent behavior with MLflow Tracing for full observability of tool calls and reasoning chains."),

    ("Databricks Apps Deployment Guide", "Technical", "Apps",
     "Databricks Apps enables full-stack application deployment directly on the platform. Architecture: FastAPI (Python) backend serving API endpoints, React frontend built with Vite, packaged together in a Databricks Asset Bundle. Key steps: (1) Define app.yaml with runtime configuration and resource requirements. (2) Backend connects to Databricks resources (tables, models, endpoints) using the SDK with automatic OAuth. (3) Frontend communicates with backend via REST API. (4) Deploy with 'databricks bundle deploy' for CI/CD integration. (5) Apps inherit workspace permissions for secure data access. Use cases: interactive ML dashboards, agent chat interfaces, data quality monitoring tools, and custom analytics applications."),

    ("Structured Streaming Patterns", "Technical", "Streaming",
     "Common streaming patterns on the lakehouse: (1) Bronze-Silver-Gold medallion with Auto Loader for incremental file ingestion into bronze, stream-to-stream transformations for silver, and stream-to-table aggregations for gold. (2) Change Data Capture with Debezium or Fivetran feeding into Delta tables with merge operations. (3) Real-time ML inference using foreachBatch to score streaming data against deployed models. (4) Event-driven alerting with streaming queries that detect anomalies and trigger notifications. Key configurations: use trigger(availableNow=True) for micro-batch, checkpoint locations in cloud storage, and watermark settings for late data handling. Delta Live Tables provides a declarative alternative with built-in quality constraints."),

    ("Foundation Model API Reference", "Technical", "LLMs",
     "Databricks Foundation Model APIs provide access to state-of-the-art LLMs via OpenAI-compatible endpoints. Available models include Meta Llama 3.3 70B, DBRX, Mixtral, and embedding models like GTE-Large. Usage: (1) Set up an OpenAI client with workspace URL as base_url and PAT as api_key. (2) Call chat.completions.create() with model name matching the serving endpoint. (3) For embeddings, use the embeddings.create() endpoint. (4) Token limits vary by model — Llama 3.3 supports 128K context. (5) Rate limits are per-endpoint and configurable via AI Gateway. (6) Cost is based on token usage with pay-per-token pricing. Best practices: use system prompts for consistent behavior, implement retry logic for rate limits, and log all calls with MLflow for tracing and cost tracking."),

    # FAQs
    ("Pricing and Licensing FAQ", "FAQ", "Pricing",
     "Frequently asked questions about pricing: Q: How is Databricks priced? A: Databricks uses a consumption-based model measured in DBUs (Databricks Units). Different workload types have different DBU rates — SQL compute, Jobs compute, and All-Purpose compute each have distinct pricing. Serverless workloads are priced per-DBU with automatic scaling. Q: What's included in the platform fee? A: Unity Catalog governance, MLflow, Delta Lake, and the workspace environment are included at no additional cost. Model serving, Vector Search, and Foundation Models are priced separately based on usage. Q: How does this compare to Snowflake pricing? A: For equivalent workloads, Databricks typically offers 30-50% savings on mixed analytics and ML workloads due to the unified architecture eliminating data movement costs."),

    ("Data Migration FAQ", "FAQ", "Migration",
     "Frequently asked questions about data migration: Q: How long does a typical migration take? A: Small-to-medium migrations (under 50TB) typically complete in 4-8 weeks. Enterprise migrations can take 3-6 months depending on complexity. Q: Can we run both platforms in parallel? A: Yes, we recommend a phased migration approach with both platforms running simultaneously. Federation queries allow the lakehouse to read from your existing warehouse without data movement. Q: What about our existing SQL scripts? A: The SQL Migration Toolkit automatically translates most proprietary SQL dialects. Manual intervention is typically needed for 10-15% of complex queries. Q: Will our BI tools still work? A: Yes, all major BI tools (Tableau, Power BI, Looker, Qlik) connect natively to Databricks SQL warehouses."),

    ("Security and Compliance FAQ", "FAQ", "Security",
     "Frequently asked questions about security: Q: Is Databricks SOC 2 certified? A: Yes, Databricks maintains SOC 2 Type II certification, along with ISO 27001, ISO 27017, ISO 27018, HIPAA, FedRAMP, and PCI-DSS. Q: Can we use our own encryption keys? A: Yes, customer-managed keys (CMK) are supported for all cloud providers. Q: How is data isolated between customers? A: Each customer gets a dedicated control plane with data stored in their own cloud account. Q: Does Databricks support private connectivity? A: Yes, Private Link/Private Endpoints are available on AWS, Azure, and GCP, ensuring data never traverses the public internet. Q: How does authentication work? A: SSO via SAML 2.0 or OIDC, with SCIM for automated user/group provisioning from your identity provider."),

    ("Getting Started FAQ", "FAQ", "Onboarding",
     "Frequently asked questions for new users: Q: What programming languages are supported? A: Python, SQL, R, and Scala in notebooks. SQL is the primary language for analytics workloads. Q: Do I need to know Spark? A: Not necessarily. SQL users can work entirely with SQL warehouses. Python users can leverage the Pandas API on Spark for familiar syntax. Q: How do I connect my data? A: Use Auto Loader for file-based ingestion, Lakeflow Connect for SaaS connectors (Salesforce, SAP, etc.), or partner ETL tools. Q: Where can I learn more? A: Databricks Academy offers free self-paced courses, and the documentation at docs.databricks.com is comprehensive. Q: Can I try it for free? A: Yes, a 14-day free trial is available at databricks.com with $400 in credits."),

    # More products
    ("Genie Data Intelligence", "Product", "AI/BI",
     "Genie is our AI-powered data intelligence engine that enables natural language conversations with your data. Users can ask questions in plain English and get instant answers backed by governed data assets. Genie understands business context through curated semantic models, ensuring accurate translations from natural language to SQL. Key features include multi-turn conversations, automatic visualization suggestions, proactive insights, and the ability to save and share analyses. Genie Spaces allow administrators to create domain-specific data assistants with curated table sets and business context. All queries respect Unity Catalog permissions, ensuring users only see data they're authorized to access."),

    ("Lakeflow Pipelines", "Product", "Data Engineering",
     "Lakeflow is our next-generation data pipeline platform that simplifies the creation and management of production ETL workflows. It includes Lakeflow Connect for ingesting data from 100+ SaaS and database sources with change data capture, Lakeflow Pipelines (powered by Delta Live Tables) for declarative data transformation with built-in quality controls, and Lakeflow Jobs for orchestrating complex multi-step workflows. Lakeflow automatically handles schema evolution, data deduplication, and error recovery. Pipeline observability provides real-time metrics on data freshness, quality scores, and processing latency. Cost optimization features recommend right-sized compute and identify idle resources."),

    ("Mosaic AI Model Serving", "Product", "AI Infrastructure",
     "Mosaic AI Model Serving provides production-grade inference infrastructure for all model types. Deploy custom ML models, foundation models, and AI agents to auto-scaling endpoints with GPU support. Features include A/B testing for comparing model versions, traffic splitting for gradual rollouts, and automatic scaling from zero to thousands of concurrent requests. Inference tables capture all predictions for monitoring and debugging. AI Gateway provides unified routing to internal and external model providers with rate limiting, content filtering, and usage tracking. Supported frameworks include MLflow PyFunc, transformers, vLLM, and TensorRT for optimized inference performance."),

    # More competitive
    ("Competitive Positioning vs Azure ML", "Competitive", "Azure ML",
     "Key differentiators against Azure ML: (1) Unified platform — Azure ML is disconnected from data engineering and analytics, requiring separate Azure Data Factory and Synapse services. Databricks provides one platform for the entire data and AI lifecycle. (2) Open source — MLflow is the industry standard for ML lifecycle management with 18M+ monthly downloads. Azure ML's proprietary experiment tracking creates vendor lock-in. (3) Collaborative notebooks — Real-time collaboration with versioning, compared to Azure ML's limited notebook experience. (4) Cost transparency — Databricks' DBU-based pricing is predictable, while Azure ML has complex pricing across compute, storage, and managed endpoints. (5) Community and ecosystem — Larger community, more integrations, and broader framework support."),

    ("Competitive Positioning vs Google Vertex AI", "Competitive", "Google",
     "Key differentiators against Google Vertex AI: (1) Multi-cloud — Databricks runs on AWS, Azure, and GCP while Vertex AI is GCP-only. (2) Data lakehouse integration — Vertex AI requires separate BigQuery for analytics and Cloud Storage for data lakes. Databricks unifies both. (3) Open formats — Delta Lake's open format prevents lock-in, while BigQuery uses proprietary Capacitor format. (4) ML experiment management — MLflow provides superior experiment tracking and model versioning compared to Vertex AI's experiment tracking. (5) Governance — Unity Catalog provides fine-grained governance across all assets while Vertex AI relies on disparate IAM policies."),

    # More technical
    ("Feature Engineering Best Practices", "Technical", "Feature Store",
     "Best practices for feature engineering on the lakehouse: (1) Use Unity Catalog Feature Tables to create reusable, governed feature sets. Define features with proper documentation and tags. (2) Implement point-in-time lookups for training data to prevent data leakage. (3) Use online tables powered by Lakebase for low-latency feature serving in real-time inference. (4) Compute features incrementally using streaming pipelines for freshness. (5) Store feature metadata including descriptions, owners, and SLAs in Unity Catalog. (6) Version features alongside model versions for reproducibility. (7) Monitor feature distributions for drift using Lakehouse Monitoring. (8) Share features across teams through catalog grants, reducing duplication."),

    ("LLMOps on Databricks", "Technical", "LLMOps",
     "Production practices for LLM applications: (1) Use AI Gateway as the single entry point for all LLM calls — it provides unified logging, rate limiting, and failover across providers. (2) Implement evaluation pipelines with MLflow Evaluate before deploying updates. Key metrics: faithfulness (does the answer match retrieved context?), relevance (is the response on-topic?), and groundedness (is the response based on provided information vs hallucination?). (3) Set up MLflow Tracing for full observability of agent tool calls, retrieval steps, and LLM responses. (4) Use A/B testing on serving endpoints to compare agent versions. (5) Monitor token usage and costs through system tables. (6) Implement guardrails for content safety including PII detection, toxicity filtering, and topic boundaries."),

    # Additional playbooks
    ("ROI Calculator Guide", "Sales Playbook", "ROI",
     "Guide to building a compelling ROI case: (1) Infrastructure savings: Calculate current spend on data warehouse licenses, compute clusters, ETL tools, ML platforms, and BI tools. Databricks typically reduces total infrastructure costs by 30-50% through consolidation. (2) Productivity gains: Measure time spent on data pipeline maintenance, model deployment, and cross-tool integration. Lakehouse architecture reduces maintenance overhead by 40-60%. (3) Speed to value: Track time from data ingestion to production insight. Unified platform reduces this from weeks to days. (4) Risk reduction: Quantify cost of data quality issues, compliance violations, and security incidents that governance features prevent. (5) Revenue acceleration: AI/ML models typically generate 5-15% improvement in targeted business metrics (conversion rates, churn reduction, pricing optimization)."),

    ("Customer Success Stories", "Sales Playbook", "References",
     "Key customer success stories by industry: FINANCIAL SERVICES — A top-10 global bank migrated from on-premises Hadoop to the lakehouse, reducing their analytics infrastructure costs by 45% while enabling real-time fraud detection that prevented $200M in annual losses. HEALTHCARE — A Fortune 100 healthcare company built an ML platform on Databricks to predict patient readmission risk, reducing 30-day readmissions by 18% across their hospital network. RETAIL — A leading e-commerce company uses the lakehouse for real-time personalization, serving 50M+ daily product recommendations with sub-100ms latency, increasing click-through rates by 35%. MANUFACTURING — An automotive OEM deployed predictive maintenance models on production line sensor data, reducing unplanned downtime by 40% and saving $60M annually."),

    ("ServiceNow Platform Integration Guide", "Technical", "ServiceNow",
     "Integrating Databricks with ServiceNow for enterprise workflows: (1) Use Lakeflow Connect or JDBC to ingest ServiceNow data (incidents, changes, knowledge articles, CMDB) into Delta tables. (2) Build ML models for ticket classification, priority prediction, and knowledge article recommendation. (3) Use AI agents to automate incident response — agents query the CMDB for affected services, check monitoring dashboards, and suggest remediation steps. (4) Deploy chatbot interfaces through Databricks Apps for internal help desk automation. (5) Push ML predictions back to ServiceNow via REST API for workflow automation. (6) Use Unity Catalog to govern the full data pipeline from ServiceNow source to ML model outputs."),

    # More FAQs
    ("AI and GenAI FAQ", "FAQ", "AI",
     "Frequently asked questions about AI capabilities: Q: What LLMs are available? A: Databricks offers Foundation Model APIs including Meta Llama 3.x, DBRX, Mixtral, and partner models through AI Gateway. Embedding models include GTE-Large and BGE-Large. Q: Can I fine-tune models? A: Yes, the platform supports fine-tuning of open-source LLMs using your own data with built-in training infrastructure. Q: What about data privacy with LLMs? A: Foundation Model APIs run within your Databricks workspace — your data never leaves the platform and is not used for model training. External models accessed through AI Gateway can be configured with data loss prevention policies. Q: How do I build RAG applications? A: Use Vector Search to create a similarity search index on your documents, then combine retrieval with Foundation Model APIs in an agent framework."),

    ("Workspace Administration FAQ", "FAQ", "Admin",
     "Frequently asked questions about workspace administration: Q: How many users can a workspace support? A: There is no hard limit on users per workspace. Enterprise deployments commonly have 1,000+ users. Q: How do I manage costs? A: Use budgets and alerts to track spending, cluster policies to control compute sizes, and system tables to analyze usage patterns. Serverless compute auto-scales and scales to zero when idle. Q: Can I automate workspace setup? A: Yes, use the Terraform provider for infrastructure-as-code, the Databricks CLI for scripting, and the REST API for programmatic management. Q: How do backups work? A: Delta tables support time travel for point-in-time recovery. Workspace objects are versioned in the control plane. External locations should use cloud-native backup solutions."),

    # Additional knowledge entries to reach ~50
    ("Delta Lake Deep Dive", "Technical", "Storage",
     "Delta Lake is the open-source storage layer that powers the lakehouse. Key features: (1) ACID transactions ensure data consistency even with concurrent writers. (2) Time travel allows querying historical versions of data for auditing and rollback. (3) Z-ordering and liquid clustering optimize read performance for common query patterns. (4) Change Data Feed enables efficient incremental processing by exposing row-level changes. (5) Vacuum operations manage storage by removing old file versions. (6) Schema evolution handles column additions and type changes gracefully. (7) Delta sharing enables secure data sharing across organizations without data copying. Performance tip: use liquid clustering instead of traditional partitioning for modern workloads — it automatically optimizes data layout."),

    ("Databricks SQL Best Practices", "Technical", "SQL",
     "Best practices for SQL analytics on the lakehouse: (1) Use serverless SQL warehouses for instant startup and automatic scaling. (2) Create materialized views for frequently-run dashboard queries to achieve sub-second response times. (3) Leverage AI functions (ai_query, ai_classify, ai_extract) to add LLM-powered intelligence directly in SQL queries. (4) Use query profile to identify bottlenecks — look for skewed joins and excessive shuffles. (5) Apply liquid clustering on large tables to optimize read performance without managing partitions. (6) Set up query monitoring alerts for long-running or resource-heavy queries. (7) Use parameters in queries for dynamic filtering in dashboards. (8) Implement row-level security with row access policies for multi-tenant data."),

    ("Customer Health Scoring", "Sales Playbook", "Customer Success",
     "Framework for assessing customer health and predicting expansion or churn risk: Green indicators — growing DBU consumption month-over-month, multiple active teams, executive sponsor engaged, new use cases being explored. Yellow indicators — flat or declining consumption, single team usage, upcoming contract renewal without expansion discussion, support ticket volume increasing. Red indicators — significant consumption drop, key champion departing, competitive evaluation initiated, multiple P1 support incidents. Action playbook: For yellow accounts, schedule QBR with value assessment. For red accounts, escalate to account team, deploy customer success resources, and prepare competitive defense materials. Track health scores monthly in CRM and align with SA territory reviews."),

    ("Model Evaluation Comprehensive Guide", "Technical", "ML Evaluation",
     "Comprehensive guide to model evaluation on Databricks: For classification models: use AUC-ROC for ranking quality, precision-recall for imbalanced classes, and confusion matrix for error analysis. For regression: RMSE, MAE, and R-squared. For LLM/RAG applications: faithfulness (does the response match source documents?), relevance (is the answer on-topic?), groundedness (avoids hallucination), and toxicity (safety). MLflow Evaluate automates evaluation with built-in metrics and custom judges. Best practice: create evaluation datasets with at least 50-100 examples covering edge cases. Run evaluations in CI/CD before promoting models. Use MLflow Compare to track quality across versions. Set minimum quality thresholds as deployment gates."),

    ("Real-Time Data Architecture", "Technical", "Architecture",
     "Reference architecture for real-time data processing on the lakehouse: Ingestion layer uses Auto Loader for cloud storage files, Kafka connector for streaming events, and Lakeflow Connect for SaaS sources. Processing layer implements the medallion architecture: bronze (raw data), silver (cleaned and enriched), gold (business-ready aggregates). Each layer uses Delta Lake for ACID guarantees and time travel. Serving layer provides data through SQL warehouses for dashboards, Model Serving for ML predictions, and Feature Store for real-time features. Orchestration uses Databricks Workflows for scheduling and monitoring. Governance with Unity Catalog spans all layers providing lineage, access control, and audit logging. This architecture supports latencies from sub-second (streaming) to hourly (batch)."),

    ("Executive Briefing Template", "Sales Playbook", "Executive",
     "Structure for executive briefings: (1) Opening (5 min) — acknowledge their time, state the objective, preview the agenda. (2) Industry context (10 min) — share relevant trends and challenges specific to their vertical; reference peer companies (anonymized). (3) Platform overview (15 min) — high-level architecture focused on their priorities; demonstrate 2-3 capabilities most relevant to their stated challenges. (4) Customer proof points (10 min) — share 2 reference stories from similar companies with quantified outcomes. (5) Joint success vision (10 min) — outline a potential roadmap for their organization; propose specific first steps. (6) Q&A and next steps (10 min) — capture questions and commitments; agree on follow-up actions. Key rules: never go deeper than the audience wants, always tie technology back to business outcomes, and end with a clear call to action."),

    ("Multi-Cloud Strategy Guide", "Technical", "Cloud",
     "Databricks multi-cloud capabilities and best practices: Databricks runs natively on AWS, Azure, and GCP with consistent APIs and features across all clouds. Multi-cloud strategies: (1) Follow-the-data — deploy Databricks in the same cloud where source data resides to minimize egress costs and latency. (2) Best-of-breed — use each cloud's strengths (e.g., AWS for ML infrastructure, Azure for enterprise integration). (3) Risk mitigation — avoid single-cloud lock-in by standardizing on open formats (Delta Lake, MLflow). Unity Catalog provides cross-cloud governance, and Delta Sharing enables data exchange between clouds without copying. Databricks Asset Bundles (DABs) enable infrastructure-as-code deployments that are portable across clouds with environment-specific configurations."),

    ("Lakehouse Monitoring Setup Guide", "Technical", "Monitoring",
     "Setting up comprehensive monitoring on the lakehouse: (1) Data quality monitoring — use Lakehouse Monitoring to track table-level statistics (row counts, null rates, distribution changes) and set up anomaly alerts. (2) Model monitoring — enable inference tables on serving endpoints and create monitors for prediction distribution, latency, and error rates. (3) Pipeline monitoring — use Delta Live Tables expectations for data quality rules, and Workflow notifications for job failures. (4) Cost monitoring — query system.billing.usage tables for consumption analysis; set up budget alerts in the account console. (5) Security monitoring — audit logs in system.access.audit track all data access and configuration changes. (6) Custom dashboards — create AI/BI dashboards combining all monitoring signals for a single pane of glass."),

    ("GenAI Use Case Prioritization", "Sales Playbook", "GenAI Strategy",
     "Framework for helping customers prioritize GenAI use cases: Tier 1 (Quick wins, 2-4 weeks): Internal knowledge base chatbots, document summarization, code assistance, data quality tagging with ai_classify. Tier 2 (Medium effort, 1-3 months): Customer support automation, sales email generation, contract analysis, anomaly explanation. Tier 3 (Strategic, 3-6+ months): Multi-agent systems for complex workflows, fine-tuned domain models, real-time decision agents, autonomous data pipeline management. Evaluation criteria: (1) Business impact — revenue/cost impact and executive visibility. (2) Data readiness — is training/context data available and clean? (3) Risk tolerance — how critical is accuracy? Can humans review outputs? (4) Technical feasibility — does the current architecture support the use case? Start with Tier 1 to build confidence and demonstrate platform capabilities."),
]

df_kb = pd.DataFrame(knowledge_docs, columns=["title", "category", "subcategory", "content"])
df_kb.insert(0, "doc_id", uid_array(len(df_kb)))
df_kb["last_updated"] = random_dates(date(2025, 6, 1), date(2026, 3, 1), len(df_kb))

spark.createDataFrame(df_kb).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.gtm_knowledge_base")
print(f"gtm_knowledge_base: {len(df_kb)} rows written")

# COMMAND ----------

# DBTITLE 1,Summary and Validation
# MAGIC %md
# MAGIC ## 0.13 — Summary & Validation

# COMMAND ----------

# DBTITLE 1,Validate All Generated Tables
print("=" * 60)
print("  DATA GENERATION COMPLETE")
print("=" * 60)

tables = [
    "gtm_accounts", "gtm_contacts", "gtm_opportunities",
    "gtm_activities", "gtm_campaigns", "gtm_campaign_members",
    "gtm_lead_scores", "gtm_knowledge_base"
]

total = 0
for t in tables:
    count = spark.sql(f"SELECT COUNT(*) FROM {catalog}.{schema}.{t}").first()[0]
    total += count
    print(f"  {t:<30} {count:>6} rows")

print(f"  {'\u2500' * 40}")
print(f"  {'TOTAL':<30} {total:>6} rows")
print(f"\n  Conversion rate: {spark.sql(f'SELECT AVG(converted) FROM {catalog}.{schema}.gtm_lead_scores').first()[0]:.1%}")

# COMMAND ----------

# DBTITLE 1,Cleanup Instructions
# MAGIC %md
# MAGIC ## Cleanup (only run if you want to remove all training data)
# MAGIC
# MAGIC **Optional.** Only run the cell below to reset the workspace for a fresh run. It drops the entire training schema and all tables within it.

# COMMAND ----------

# DBTITLE 1,Drop Training Schema (Optional)
# Uncomment the line below to drop the entire training schema and all data
# spark.sql(f"DROP SCHEMA IF EXISTS {catalog}.{schema} CASCADE")
