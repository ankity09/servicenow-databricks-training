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
# MAGIC You are a data science team at ServiceNow supporting the Go-To-Market (GTM) organization. Your GTM team wants to
# MAGIC improve lead conversion rates by building an **AI-powered lead scoring system** for ServiceNow's product portfolio (ITSM, CSM, ITOM, HRSD, SecOps, Now Assist). The sales team
# MAGIC currently relies on gut instinct and static rules — you will replace that with ML models trained
# MAGIC on historical CRM data and augment the workflow with Generative AI.
# MAGIC
# MAGIC This notebook sets up the **Unity Catalog** environment (**Unity Catalog** is Databricks' centralized governance layer -- it manages access control, data lineage, and audit logging across all data assets) and generates realistic Salesforce-style
# MAGIC GTM data that we will use throughout the remaining modules:
# MAGIC
# MAGIC | Table | Description | Approx Rows |
# MAGIC |-------|-------------|-------------|
# MAGIC | `gtm_accounts` | Company accounts | 40,000 |
# MAGIC | `gtm_contacts` | Individual contacts / leads | 200,000 |
# MAGIC | `gtm_opportunities` | Sales opportunities | 100,000 |
# MAGIC | `gtm_activities` | Emails, calls, meetings, demos | 1,000,000 |
# MAGIC | `gtm_campaigns` | Marketing campaigns | 2,000 |
# MAGIC | `gtm_campaign_members` | Campaign engagement records | 400,000 |
# MAGIC | `gtm_lead_scores` | Composite lead scores + conversion label | 200,000 |
# MAGIC | `gtm_knowledge_base` | Product docs, playbooks, competitive intel | 50 |
# MAGIC
# MAGIC **Compute:** Serverless (serverless compute auto-provisions and auto-scales infrastructure -- no cluster configuration needed)
# MAGIC
# MAGIC **Estimated Runtime:** ~10 minutes

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
    "Delta", "Echo", "Falcon", "Granite", "Ivy", "Jade", "Keystone",
    "Crescent", "Ember", "Flux", "Glyph", "Harbor", "Ionic", "Jasper", "Kova",
    "Lattice", "Mono", "Neon", "Opal", "Pluto", "Quasar", "Ridge", "Sigma",
    "Terra", "Ultra", "Vortex", "Wren", "Xeno", "Yonder", "Zinc", "Aether",
    "Blaze", "Circuit", "Drift", "Epoch", "Flint", "Gyro",
    "Arrow", "Birch", "Cobalt", "Dune", "Ember", "Fern", "Grove", "Halo",
    "Ion", "Jubilee", "Kelvin", "Lark", "Maven", "Nimbus", "Onyx", "Pine",
    "Quill", "Raven", "Slate", "Thorn", "Umbra", "Vale", "Willow", "Xenon",
    "Yarrow", "Zeal", "Arch", "Brio", "Cedar", "Dash", "Elm", "Frost",
    "Gleam", "Husk", "Indigo", "Jolt", "Kite", "Lynx", "Myth", "Notch",
    "Orion", "Peak", "Quest", "Rush", "Spark", "Trace", "Ursa", "Vine",
    "Warp", "Axis", "Brisk", "Cliff", "Dawn", "Etch", "Flame", "Grit",
    "Helm", "Iris", "Jet", "Knox", "Lever", "Mist", "Nave", "Oak",
    "Pyre", "Rift", "Storm", "Tide", "Verge", "Wake", "Yield", "Zenon",
    "Alloy", "Basalt", "Chroma", "Draco", "Envoy", "Fable", "Glint", "Haze",
    "Ignis", "Jarvis", "Krypton", "Locus", "Magnet", "Nebula", "Opus", "Paragon",
    "Relay", "Sable", "Talon", "Vertex", "Axiom", "Borealis", "Cipher", "Dusk",
    "Equinox", "Fjord", "Geo", "Hyper", "Inertia", "Joule", "Karma", "Lunar",
    "Mantis", "North", "Omega", "Proton", "Quartz", "Rowan", "Synth", "Tempest",
    "Apex2", "Nova2", "Solar", "Aegis", "Bolt", "Coda", "Denali", "Evo",
    "Forge2", "Gale", "Haven", "Ivory", "Jubilo", "Keystone2", "Lithium", "Metro",
    "Noble", "Osprey", "Presto", "Rigel", "Strata", "Turbo", "Unified", "Vista",
    "Apex3", "Nova3", "Cosmo", "Duet", "Elysium", "Fenix", "Granite2", "Herald",
    "Impact", "Juris", "Kondor", "Lira", "Maxim", "Nyx", "Oriel", "Palladin",
    "Quintus", "Regal", "Sirius", "Torus"
]

SUFFIXES = [
    "Digital Solutions", "Industries", "Technologies", "Systems", "Analytics",
    "Consulting", "Global", "Partners", "Corp", "Labs", "Networks",
    "Innovations", "Group", "Platforms", "Services", "Dynamics", "Ventures",
    "Logic", "Data", "Cloud", "Software", "AI", "Security", "Health",
    "Financial", "Media", "Energy", "Robotics", "Biotech", "Aerospace",
    "Logistics", "Commerce", "Intelligence", "Works", "Hub", "Bridge",
    "Matrix", "Realm", "Core", "Link",
    "Capital", "Scientific", "Electric", "Collective", "Machine", "Research",
    "Foundry", "Studio", "Exchange", "Signal", "Labs Inc", "Tech", "Micro",
    "Vision", "Craft", "Systems Inc", "Digital", "Quantum", "Neural", "Cyber",
    "Devices", "Instruments", "Solutions", "Associates", "Holdings", "Enterprise",
    "Incorporated", "International", "Direct", "One", "Prime", "Nexus",
    "Kinetics", "Optics", "Sensors", "Controls", "Fusion", "Synergy",
    "Alliance", "Catalyst", "Precision", "Elements", "Resolve", "Apex",
    "United", "Advanced", "Applied", "Integrated", "Connected", "Distributed",
    "Autonomous", "Responsive", "Predictive", "Cognitive", "Adaptive", "Scalable",
    "Interactive", "Strategic", "Creative", "Industrial", "Professional", "Technical",
    "Frontier", "Horizon", "Summit", "Pioneer", "Navigator", "Pathfinder",
    "Compass", "Lighthouse", "Anchor", "Keystone", "Cornerstone", "Foundation",
    "Architects", "Builders", "Makers", "Forge", "Factory", "Assembly",
    "Depot", "Central", "Source", "Supply", "Grid", "Node",
    "Circuit", "Pulse", "Beam", "Spectrum", "Wave", "Field",
    "Stream", "Flow", "Motion", "Vector", "Force", "Power",
    "Insight", "Foresight", "Outlook", "Perspective", "Scope", "Lens",
    "Blueprint", "Design", "Pattern", "Method", "Protocol", "Framework",
    "Radius", "Vertex", "Apex Inc", "Summit Inc", "Peak", "Pinnacle",
    "Edge", "Point", "Base", "Root", "Stem", "Branch",
    "Atlas", "Meridian", "Equator", "Orbit", "Trajectory", "Momentum",
    "Velocity", "Magnitude", "Amplitude", "Resonance", "Harmony", "Cadence",
    "Tempo", "Rhythm", "Spark", "Ignite", "Catalyst Inc", "Propel",
    "Ascend", "Elevate", "Transcend", "Evolve", "Transform", "Innovate",
    "Ware", "Net", "Logic Inc", "Infotech", "Dataworks", "Technica",
    "Operations", "Mobility", "Automation", "Compute", "Engines", "Nexus Inc",
    "Dynamics Inc", "Ventures Inc"
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

PRODUCT_LINES = ["ITSM Pro", "CSM Enterprise", "ITOM Visibility", "HRSD", "Security Operations", "App Engine"]

# COMMAND ----------

# DBTITLE 1,Accounts Table Description
# MAGIC %md
# MAGIC ## 0.5 — Generate Accounts (40,000 rows)
# MAGIC
# MAGIC Accounts are the root entity in our data model -- every contact, opportunity, and activity traces back to one.
# MAGIC Fields like `industry`, `employee_count`, and `account_tier` will become key ML features for predicting deal outcomes.
# MAGIC
# MAGIC All tables are saved in **Delta Lake** format (Delta Lake is an open-source storage format that adds reliability features like ACID transactions and time travel to data lakes).

# COMMAND ----------

# DBTITLE 1,Generate 40,000 Account Records
n_accounts = 40000

# Generate unique company names (prefixes x suffixes > 40000 needed)
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
# MAGIC ## 0.6 — Generate Contacts (200,000 rows)
# MAGIC
# MAGIC Each contact has a `lead_source` and `seniority_level` -- these become important ML features later.
# MAGIC For example, a "Demo Request" from a "VP" converts at a much higher rate than an "Organic" visit from an "Individual Contributor."

# COMMAND ----------

# DBTITLE 1,Generate 200,000 Contact Records
n_contacts = 200000
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
# MAGIC ## 0.7 — Generate Opportunities (100,000 rows)

# COMMAND ----------

# DBTITLE 1,Generate 100,000 Opportunity Records
n_opps = 100000
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
# MAGIC ## 0.8 — Generate Activities (1,000,000 rows)

# COMMAND ----------

# DBTITLE 1,Generate 1,000,000 Activity Records
n_activities = 1000000
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
# MAGIC ## 0.9 — Generate Campaigns (2,000 rows)

# COMMAND ----------

# DBTITLE 1,Generate 2,000 Campaign Records
n_campaigns = 2000
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
# MAGIC ## 0.10 — Generate Campaign Members (400,000 rows)

# COMMAND ----------

# DBTITLE 1,Generate 400,000 Campaign Member Records
n_members = 400000
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
# MAGIC ## 0.11 — Generate Lead Scores (200,000 rows — one per contact)
# MAGIC
# MAGIC The `converted` column is the **target variable** for our ML lead scoring model.
# MAGIC Approximately 30% of contacts convert, with conversion correlated to engagement and fit scores.

# COMMAND ----------

# DBTITLE 1,Generate 200,000 Lead Score Records
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
    # Products (8 docs)
    ("Now Platform Overview", "Product", "Core Platform",
     "The Now Platform is ServiceNow's cloud-based platform for digital workflows that connects people, functions, and systems across the enterprise. Built on a single data model and single architecture, the Now Platform provides intelligent automation, low-code development, and AI-powered experiences for every department. The platform's Configuration Management Database (CMDB) serves as the single source of truth for all IT and business assets, tracking relationships and dependencies across the infrastructure. Key differentiators include multi-instance architecture with dedicated resources per customer, the Now Platform App Engine for rapid custom application development, and Integration Hub for connecting to over 900 third-party systems. Performance Analytics provides real-time KPI tracking and benchmarking across all modules. The platform processes over 80 billion workflow transactions annually across 8,100+ enterprise customers. ServiceNow consistently leads the Gartner Magic Quadrant for ITSM and has expanded into CSM, HRSD, and Security Operations, positioning the Now Platform as the enterprise operating system for digital transformation."),

    ("IT Service Management (ITSM)", "Product", "ITSM",
     "IT Service Management is ServiceNow's flagship product and the foundation of most enterprise deployments. ITSM provides ITIL-aligned processes for incident management, problem management, change management, and service request management. The incident management module enables rapid ticket creation, intelligent routing using Predictive Intelligence, and SLA tracking with automated escalation rules. Problem management links related incidents to identify root causes and prevent recurrence. Change management includes risk assessment, CAB workflow automation, and change collision detection. The service catalog offers a consumer-grade portal where employees can request IT services, hardware, and software through a self-service storefront. Virtual Agent provides 24/7 conversational AI for common requests like password resets and software access. ITSM Pro adds Predictive Intelligence for automated ticket categorization, assignment, and priority prediction, achieving 95%+ accuracy after training on historical data. Customers typically see 30-50% reduction in mean time to resolution (MTTR) and 40-60% increase in self-service adoption within the first year."),

    ("Customer Service Management (CSM)", "Product", "CSM",
     "Customer Service Management enables omnichannel customer service by connecting front-office engagement with back-office fulfillment on a single platform. CSM provides case management with intelligent routing, self-service portals with AI-powered search, and proactive customer service through monitoring and alerting. The omnichannel capability supports phone, email, chat, social media, and messaging integration through a unified agent workspace. AI-powered routing analyzes case content, customer history, and agent skills to assign cases to the best-qualified agent, reducing handle times by 25-35%. The Customer Service Portal provides branded self-service experiences with knowledge base search, community forums, and chatbot support. Field Service Management extends CSM with work order management, dynamic scheduling, and mobile technician experiences. CSM Enterprise includes Playbooks for guided resolution workflows that standardize complex processes across teams. Integration with ITSM enables seamless escalation from customer issues to internal IT incidents. Key metrics: 35% reduction in customer service costs, 45% improvement in first-contact resolution, and 20-point increase in CSAT scores."),

    ("IT Operations Management (ITOM)", "Product", "ITOM",
     "IT Operations Management provides visibility and automation for IT infrastructure, enabling organizations to predict and prevent outages before they impact business services. ITOM Discovery automatically scans and maps IT infrastructure, identifying servers, applications, databases, load balancers, and cloud resources across on-premises and multi-cloud environments. Service Mapping builds dynamic dependency maps that show how infrastructure components support business services, enabling impact analysis during incidents or changes. Event Management aggregates and correlates alerts from monitoring tools like Splunk, Datadog, New Relic, and Dynatrace, reducing alert noise by 90% through intelligent grouping and root cause identification. Cloud Management provides visibility and governance across AWS, Azure, and GCP, including cost optimization recommendations and compliance monitoring. ITOM Health uses AIOps capabilities including anomaly detection, alert correlation, and automated remediation through Integration Hub. Customers deploying ITOM alongside ITSM typically see 60% reduction in MTTR, 50% decrease in P1 incidents through proactive detection, and 70% reduction in alert noise."),

    ("HR Service Delivery (HRSD)", "Product", "HRSD",
     "HR Service Delivery automates and streamlines HR processes across the employee lifecycle, from onboarding to offboarding, on the Now Platform. The Employee Center provides a single portal for all workplace services, combining HR, IT, Facilities, and Legal resources in one consumer-grade experience. Case and Knowledge Management enables HR teams to track, manage, and resolve employee inquiries with automated routing and SLA tracking, reducing email-based requests by 70%. Employee Onboarding and Transitions orchestrates cross-departmental workflows for new hires, role changes, and departures, ensuring nothing falls through the cracks. Employee Document Management provides secure, compliant storage and lifecycle management for employee records with configurable retention policies. The Employee Journey Management module maps and optimizes key employee moments that matter, from first day to promotion to parental leave. Now Assist for HRSD brings generative AI to employee interactions, providing instant answers from HR knowledge bases and auto-generating case summaries for HR agents. Customers report 40% reduction in HR case resolution time, 50% decrease in HR operational costs, and significant improvement in employee satisfaction scores."),

    ("Security Operations (SecOps)", "Product", "Security",
     "Security Operations connects security tools, automates response workflows, and prioritizes threats based on business impact by integrating with the Now Platform's CMDB and service maps. Vulnerability Response ingests vulnerability scan data from Qualys, Tenable, Rapid7, and other scanners, automatically prioritizing remediation based on asset criticality, exploit availability, and business service impact. Security Incident Response provides structured workflows for investigating and responding to security events, with automated enrichment from threat intelligence feeds including MITRE ATT&CK mapping. Threat Intelligence Management aggregates and correlates indicators of compromise (IOCs) from multiple sources. Configuration Compliance checks infrastructure against security benchmarks like CIS and DISA STIG, automating remediation workflows for non-compliant assets. The Security Operations dashboard provides CISO-level visibility into organizational risk posture with real-time metrics on vulnerability exposure, incident trends, and compliance status. Integration with SOAR platforms enables automated playbook execution for common security scenarios. Organizations using SecOps report 65% faster vulnerability remediation, 50% reduction in security incident response time, and 80% improvement in compliance audit readiness."),

    ("Now Assist AI", "Product", "AI",
     "Now Assist brings generative AI to every ServiceNow workflow, empowering employees, agents, and developers to work faster and smarter. Built on the Now Platform's domain-specific large language models trained on ServiceNow's proprietary workflow data, Now Assist delivers contextually relevant AI experiences across all modules. For agents, Now Assist provides case and incident summarization that condenses complex ticket histories into concise summaries, saving 3-5 minutes per interaction. Resolution note generation automatically creates structured documentation from conversation threads. For employees, Now Assist powers intelligent search across the knowledge base and virtual agent conversations that understand intent and context. For developers, Now Assist generates code, flow configurations, and test cases within App Engine Studio, accelerating development by 30-40%. Now Assist for ITSM includes AI-powered suggested resolutions that analyze similar past incidents to recommend fixes. The platform supports responsible AI with built-in guardrails, data privacy controls, and transparency features that show the reasoning behind AI recommendations. Now Assist is available as an add-on to ITSM Pro, CSM Enterprise, HRSD, and Creator Workflow plans."),

    ("App Engine", "Product", "Development",
     "App Engine is ServiceNow's low-code development platform that enables both professional developers and citizen developers to build custom applications on the Now Platform. App Engine Studio provides a guided, visual development environment for creating applications with drag-and-drop interfaces, automated testing, and one-click deployment. Flow Designer enables process automation through a visual workflow builder with pre-built actions for common operations like approvals, notifications, and record manipulation. Integration Hub connects custom apps to external systems through pre-built spokes for Salesforce, SAP, AWS, Azure, Slack, Microsoft Teams, and hundreds more. The platform supports progressive complexity: citizen developers start with App Engine Studio templates, while professional developers can extend with JavaScript, REST APIs, and custom scripted components. Mobile App Builder creates native mobile experiences without additional coding. App Engine licensing includes Guided App Creator for business users, which provides AI-assisted application scaffolding from natural language descriptions. The ServiceNow Store offers 2,000+ pre-built applications and integrations. Organizations using App Engine report 70% faster application delivery compared to traditional development and 50% reduction in shadow IT."),

    # Sales Playbooks (8 docs)
    ("Enterprise Sales Methodology", "Sales Playbook", "Methodology",
     "ServiceNow enterprise sales follows a value-based approach centered on understanding the customer's digital transformation maturity and workflow automation gaps. The process begins with a Discovery Workshop to assess current ITSM maturity using ServiceNow's proprietary maturity model across five dimensions: process, technology, people, governance, and metrics. Champions are typically VP IT Operations, VP IT Service Management, or CIO/CTO depending on the organization's structure. Economic buyers are almost always CIO or CFO given the platform-level investment. Average enterprise deal size ranges from $800K to $3M ACV for initial lands, with expansion deals averaging $1.2M-$5M. Sales cycles run 6-12 months for new logos and 3-6 months for expansion. Key selling motions include Platform Value Assessment comparing current tooling costs and complexity, Executive Briefing Centers for C-level engagement, and structured proof-of-value engagements over 3-4 weeks. The sales methodology emphasizes quantified business outcomes: MTTR reduction, self-service adoption rates, cost per ticket, and employee satisfaction scores. Every proposal must include a customer success plan with 90-day, 6-month, and 12-month milestones tied to measurable KPIs."),

    ("Objection Handling Guide", "Sales Playbook", "Objections",
     "Common objections and recommended responses for ServiceNow sales: (1) 'We already have BMC Remedy' — Position the Now Platform's modern cloud-native UX and AI capabilities versus Remedy's legacy on-premises architecture. Highlight Now Assist GenAI features that Remedy cannot match, the 3x faster upgrade cycles with SaaS, and the total cost of maintaining Remedy customizations. Reference analyst reports showing ServiceNow as the clear ITSM leader. (2) 'Too expensive' — Build a TCO analysis including hidden costs of maintaining legacy ITSM: custom development, upgrade projects costing $500K-$2M each, hardware refresh cycles, and multiple point tools for ITOM, CSM, and HRSD that ServiceNow consolidates. Show ROI within 12-18 months. (3) 'Our team is happy with Jira Service Management' — Jira SM lacks enterprise ITSM depth: no native CMDB, limited ITOM capabilities, no HRSD or CSM modules, and no AI-powered workflow automation. Position ServiceNow for enterprise scale while acknowledging Jira's DevOps strengths. (4) 'We're mid-contract with our current vendor' — Offer a parallel pilot program, help with migration planning, and structure deal timing around contract expiration. Provide a no-cost Platform Value Assessment to build the business case before renewal."),

    ("Competitive Positioning vs BMC Remedy", "Sales Playbook", "Competitive",
     "Key differentiators versus BMC Remedy/Helix: (1) Architecture — ServiceNow is cloud-native SaaS with automatic upgrades twice per year versus BMC's legacy on-premises roots that require costly upgrade projects. Even BMC Helix SaaS lacks the depth and maturity of ServiceNow's cloud platform. (2) Single platform — ServiceNow provides ITSM, ITOM, CSM, HRSD, SecOps, and App Engine on one platform with a single data model. BMC requires separate products (Remedy, Helix ITSM, Discovery, TrueSight) with complex integrations. (3) AI/ML capabilities — Now Assist brings generative AI natively to every workflow. Predictive Intelligence provides ML-based ticket routing and classification. BMC's AI capabilities are limited and bolted on. (4) Innovation velocity — ServiceNow's two annual releases (Washington, Xanadu) deliver hundreds of new features. BMC's release cadence is slower with fewer innovations. (5) Ecosystem — ServiceNow Store has 2,000+ apps and integrations versus BMC's limited marketplace. (6) Talent availability — Over 500,000 ServiceNow-certified professionals versus a shrinking BMC talent pool. Win rate against BMC is 70%+ in competitive evaluations. Focus on total cost of ownership, modern user experience, and AI-powered automation."),

    ("Proof of Value Framework", "Sales Playbook", "POV",
     "ServiceNow Proof of Value follows a structured 3-phase approach over 3-4 weeks designed to demonstrate measurable business outcomes. Phase 1 — Assessment (Week 1): Conduct ITSM maturity assessment using the ServiceNow Value Calculator. Map current processes for incident, change, and request management. Identify top 5 pain points and quantify their business impact (e.g., MTTR of 4 hours costing $50K per P1 incident). Baseline current metrics including ticket volume, resolution times, self-service adoption rate, and cost per ticket. Phase 2 — Configure and Demonstrate (Weeks 2-3): Stand up a ServiceNow Personal Developer Instance with ITSM Pro. Configure core processes: incident management with Predictive Intelligence, service catalog with Virtual Agent, change management with risk assessment, and Performance Analytics dashboards. Load representative sample data and demonstrate AI-powered routing accuracy. Phase 3 — Results and Roadmap (Week 4): Present side-by-side comparison of current state versus ServiceNow capabilities. Project 12-month ROI based on demonstrated improvements. Deliver a phased implementation roadmap. Always secure executive sponsor commitment before starting the POV to ensure decision-making authority is engaged throughout."),

    ("Land and Expand Strategy", "Sales Playbook", "Growth",
     "ServiceNow's land and expand strategy focuses on establishing ITSM as the initial foothold and systematically expanding across the enterprise. Landing motion: Start with ITSM Standard or Pro for a single IT organization, typically 50-200 IT agents. Initial ACV target: $200K-$800K. Demonstrate quick wins in the first 90 days — reduced MTTR, increased self-service adoption, improved agent productivity. Expansion path — Phase 1 (Year 1): Add ITOM Discovery and Service Mapping to build the CMDB foundation. Typical expansion: $300K-$500K. Phase 2 (Year 1-2): Expand to CSM for customer-facing service operations or HRSD for employee workflows. Typical expansion: $500K-$1M per module. Phase 3 (Year 2-3): Add Security Operations, GRC, and custom App Engine applications. Typical expansion: $300K-$800K. Phase 4 (Year 3+): Enterprise-wide Now Platform with Now Assist AI across all modules. Key expansion triggers include: new executive sponsor with broader mandate, adjacent team seeing success and requesting access, compliance requirements driving GRC adoption, digital transformation initiatives requiring custom apps, and AI/GenAI mandates from the C-suite. Track adoption metrics monthly: active users, self-service percentage, module utilization, and NPS scores."),

    ("ROI Calculator Guide", "Sales Playbook", "ROI",
     "Building the ServiceNow ROI case requires quantifying improvements across four value pillars: (1) Ticket resolution efficiency — Measure current MTTR and project 30-50% reduction with Predictive Intelligence auto-routing, knowledge suggestions, and Virtual Agent deflection. For a company processing 10,000 incidents/month at $25 average cost per ticket, a 40% efficiency gain saves $1.2M annually. (2) Self-service adoption — Baseline current self-service rate (typically 15-25%) and project improvement to 50-65% with ServiceNow's service catalog and Virtual Agent. Each ticket deflected from agent handling saves $15-22. (3) Mean time to restore service (MTRS) — With ITOM Event Management correlation and Service Mapping, P1 incident diagnosis time drops 50-70%. For organizations experiencing 10 P1s per month with $50K average business impact per hour, reducing MTRS from 4 hours to 1.5 hours saves $3M annually. (4) Platform consolidation — Calculate current spend on multiple point tools (incident management, change management, asset management, knowledge base, CMDB) and show consolidated licensing savings of 20-35%. Include hidden costs: integration maintenance, upgrade projects, training across disparate tools, and vendor management overhead. Always present ROI as a 3-year business case with conservative, likely, and optimistic scenarios."),

    ("Customer Success Stories", "Sales Playbook", "References",
     "Key ServiceNow customer success stories by industry: FINANCIAL SERVICES — A major global bank consolidated 15 legacy ITSM tools onto the Now Platform, reducing incident resolution time by 45% and achieving $8M annual savings in IT operations costs. Expanded from ITSM to SecOps for vulnerability management across 200,000 endpoints. HEALTHCARE — A top-5 hospital network deployed ITSM and HRSD, automating nurse onboarding workflows that reduced time-to-productivity from 6 weeks to 2 weeks. Virtual Agent handles 60% of IT requests without human intervention. RETAIL — A Fortune 100 retailer implemented CSM to unify customer service across 2,000 stores, reducing average handle time by 35% and improving CSAT from 72 to 89. Integration Hub connects ServiceNow to Salesforce, SAP, and their e-commerce platform. MANUFACTURING — A global automotive OEM deployed ITOM with Discovery and Service Mapping across 50 manufacturing plants, reducing unplanned downtime by 40% through proactive event correlation. CMDB accuracy improved from 45% to 92%. TECHNOLOGY — A SaaS company with 5,000 employees implemented HRSD Employee Center, deflecting 55% of HR inquiries to self-service and reducing HR case volume by 40%."),

    ("Executive Briefing Template", "Sales Playbook", "Executive",
     "Structure for CIO/CTO briefings at the ServiceNow Executive Briefing Center: (1) Opening (5 min) — acknowledge their time, establish agenda, confirm priority topics identified during pre-brief discovery call. (2) Digital transformation vision (10 min) — share industry-specific trends: platform consolidation momentum, AI-powered automation adoption, employee experience as a board-level priority. Reference Gartner and Forrester research positioning ServiceNow. (3) Platform demonstration (20 min) — focused on their top 2-3 use cases. For CIO: show ITSM + ITOM integration with AIOps-driven incident prediction. For CHRO: demonstrate HRSD Employee Center with Now Assist. For CTO: showcase App Engine rapid development and Integration Hub. (4) Customer proof points (10 min) — share 2 reference stories from their industry with quantified outcomes (MTTR reduction, cost savings, adoption rates). Offer reference calls with named customers. (5) Joint success vision (10 min) — present a phased roadmap tailored to their stated priorities. Include 90-day quick win milestones. (6) Q&A and next steps (5 min) — capture commitments, schedule technical deep-dives, propose POV timeline. Key rules: never demo features irrelevant to their priorities, always tie platform capabilities to their stated business outcomes, and leave behind a customized value assessment document."),

    # Competitive Intel (4 docs)
    ("Competitive Analysis: ITSM Market 2025", "Competitive", "Market Overview",
     "The ITSM market is consolidating as enterprises move from point tools to platform-based approaches for digital workflow automation. ServiceNow leads the Gartner Magic Quadrant for ITSM Tools for the ninth consecutive year and holds the highest position in both Ability to Execute and Completeness of Vision. The competitive landscape includes: BMC (Remedy/Helix) — legacy market leader losing share due to slow cloud migration and aging architecture. Still strong in large government and highly customized on-premises deployments. Atlassian (Jira Service Management) — strong in SMB and developer-centric organizations. Competitive on price but lacks enterprise depth in ITSM, ITOM, CMDB, and AI capabilities. Growing rapidly in mid-market. Zendesk — focused on customer support with growing ITSM features. Competitive for SMB but lacks enterprise IT operations capabilities. Freshservice (Freshworks) — cloud-native, modern UX, competitive pricing for SMB. Limited enterprise features and scale. Ivanti — focuses on IT asset management and endpoint security. Competitive in ITAM but lacks platform breadth. Market trends: AI-powered automation is the top evaluation criterion, platform consolidation continues to drive ServiceNow adoption, and employee experience workflows are expanding the total addressable market beyond IT."),

    ("Jira Service Management Comparison", "Competitive", "Jira SM",
     "Head-to-head comparison with Atlassian Jira Service Management: ServiceNow wins on enterprise depth and scale — native CMDB with Discovery and Service Mapping versus Jira's basic asset management. ServiceNow provides ITIL-aligned change management with risk assessment and CAB automation versus Jira's simplified change tracking. Predictive Intelligence for ML-based routing has no equivalent in Jira SM. ServiceNow's platform extends to CSM, HRSD, SecOps, and GRC — Jira SM is IT-only. Virtual Agent with NLU provides enterprise-grade conversational AI versus Jira's basic chatbot. Performance Analytics offers ITSM-specific KPI dashboards and benchmarking that Jira lacks. Jira SM is competitive in: pricing for SMB (free tier up to 3 agents), developer ecosystem integration (deep Jira Software and Confluence ties), and agile-native organizations already invested in Atlassian. Jira SM wins when: the organization is under 500 employees, primarily developer-focused, already has heavy Atlassian investment, and doesn't need CMDB, ITOM, or enterprise ITSM depth. ServiceNow wins when: the organization needs enterprise ITSM with ITIL alignment, requires CMDB and service mapping, wants AI-powered automation, needs multi-departmental workflows, or has complex compliance requirements. Conversion from Jira SM to ServiceNow is common as organizations scale past 1,000 employees."),

    ("Zendesk Feature Comparison", "Competitive", "Zendesk",
     "Comparison with Zendesk for customer service and IT service scenarios: ServiceNow CSM is enterprise-grade with deep ITSM integration, enabling seamless escalation from customer issues to internal IT incidents and back. Zendesk focuses primarily on SMB customer support with a modern, easy-to-deploy interface. ServiceNow advantages: (1) Platform integration — CSM connects natively to ITSM, ITOM, and CMDB for full visibility into how infrastructure issues impact customers. Zendesk requires custom integrations for IT operations visibility. (2) Field Service Management — ServiceNow provides built-in work order management and dynamic scheduling. Zendesk has no native field service capability. (3) AI depth — Now Assist provides case summarization, resolution suggestions, and knowledge article generation. Zendesk AI is focused on chatbot deflection and simpler automation. (4) Enterprise scale — ServiceNow handles complex multi-tier support structures, parent-child account hierarchies, and SLA management across geographies. Zendesk is optimized for simpler support models. Zendesk advantages: faster deployment (weeks vs months), lower initial cost, simpler administration, and a large marketplace of lightweight integrations. Zendesk wins in SMB and companies with straightforward customer support needs. ServiceNow wins when organizations need enterprise customer service with IT operations integration."),

    ("BMC Helix Feature Comparison", "Competitive", "BMC Helix",
     "Detailed comparison with BMC Helix ITSM: ServiceNow's cloud-native advantage is the primary differentiator — the Now Platform was built for the cloud from day one, while BMC Helix is a re-platformed version of Remedy. This architectural difference shows in several ways: (1) Upgrade experience — ServiceNow delivers two seamless upgrades per year with zero downtime. BMC upgrades are major projects requiring 3-6 months of planning and testing, costing $500K-$2M per cycle. (2) Innovation velocity — ServiceNow releases 500+ new features per major release. BMC's innovation pace is significantly slower. (3) User experience — ServiceNow's modern, responsive UI consistently scores higher in user satisfaction surveys versus BMC's interface, even in the Helix version. (4) AI capabilities — Now Assist generative AI and Predictive Intelligence are natively integrated. BMC's AI features are less mature and require additional licensing. (5) Platform breadth — ServiceNow provides ITSM, ITOM, CSM, HRSD, SecOps, and App Engine on one platform. BMC requires separate products with complex integrations. (6) Ecosystem — ServiceNow has 500,000+ certified professionals and 2,000+ Store apps versus BMC's shrinking certification base. Migration path from BMC to ServiceNow is well-established with automated migration tools and a 12-16 week typical timeline."),

    # Technical Docs (10 docs)
    ("Now Platform Architecture", "Technical", "Architecture",
     "The Now Platform is built on a single-instance multi-tenant architecture where each customer receives a dedicated application instance with isolated data and customizations. The platform uses a single data model — all modules (ITSM, CSM, HRSD, SecOps) share the same underlying database and table structure, eliminating data silos and enabling cross-module workflows. The architecture consists of three tiers: (1) Presentation tier — responsive web UI, mobile apps, Service Portal, and Employee Center, all rendering through the same framework. (2) Application tier — business logic, workflow engine, Flow Designer, Predictive Intelligence ML runtime, and Now Assist AI services. The application server processes scripted business rules, UI policies, client scripts, and scheduled jobs. (3) Data tier — MySQL-based multi-instance database with each customer getting a fully isolated instance. The CMDB (Configuration Management Database) is the central data store for all IT and business assets, maintaining relationships through the Common Service Data Model (CSDM). Key integration points include REST API, Table API, SOAP API, MID Server for on-premises discovery, and Integration Hub with 900+ pre-built spokes. The platform supports global multi-data-center deployment with active-active failover."),

    ("Flow Designer Best Practices", "Technical", "Automation",
     "Flow Designer is the primary tool for building automated workflows on the Now Platform without writing code. Best practices for enterprise deployments: (1) Use subflows to create reusable components — common patterns include approval chains, notification sequences, and record creation workflows. Subflows promote consistency and reduce maintenance overhead. (2) Implement error handling in every flow with try-catch patterns using the Error Handler step. Log failures to a dedicated error table for monitoring. (3) Leverage spokes from Integration Hub for external system calls rather than building custom REST steps. Pre-built spokes for Salesforce, SAP, Slack, Microsoft Teams, AWS, and Azure are tested and maintained by ServiceNow. (4) Use flow triggers wisely — record triggers should include conditions to prevent unnecessary executions that impact performance. Schedule triggers for batch operations during off-peak hours. (5) Test flows in sub-production instances before promoting to production using Update Sets or the Automated Test Framework. (6) Monitor flow execution through Flow Designer Analytics dashboards, tracking success rates, average duration, and error frequency. (7) Follow the principle of least privilege for flow execution context — run flows as system only when necessary. (8) Document flows with descriptions on each step for maintainability."),

    ("CMDB and Service Mapping", "Technical", "CMDB",
     "The Configuration Management Database (CMDB) is the foundation of service-aware IT operations on the Now Platform. It stores Configuration Items (CIs) — servers, applications, databases, network devices, cloud resources, and business services — along with their relationships and dependencies. Key CMDB concepts: (1) CI Classes — hierarchical classification system with over 900 out-of-the-box CI types organized in a class hierarchy (e.g., cmdb_ci → cmdb_ci_server → cmdb_ci_linux_server). (2) Relationships — CIs are connected through relationship types like 'Runs on', 'Hosted on', 'Depends on', and 'Used by'. (3) Discovery — automated scanning using MID Server agents that identify and populate CIs through network probes, credential-based access, and cloud API integration. Supports horizontal discovery (network scanning) and deep discovery (application-level mapping). (4) Service Mapping — builds dynamic service maps by tracing traffic patterns and dependencies from business service entry points down through the infrastructure stack. Maps update automatically as infrastructure changes. (5) Common Service Data Model (CSDM) — ServiceNow's recommended framework for organizing CMDB data into business services, technical services, and infrastructure layers. (6) Health metrics — track CMDB accuracy with completeness, compliance, and currency scores. Target 90%+ CMDB accuracy for effective incident and change management."),

    ("Integration Hub Guide", "Technical", "Integration",
     "Integration Hub connects the Now Platform to external systems through a library of pre-built spokes and a framework for custom integrations. Key capabilities: (1) Pre-built spokes — 900+ connectors for enterprise systems including Salesforce, SAP, Workday, AWS, Azure, GCP, Slack, Microsoft Teams, Jira, PagerDuty, and Splunk. Each spoke provides actions usable in Flow Designer without writing code. (2) REST step — for ad-hoc REST API calls to any system. Supports authentication methods including OAuth 2.0, API key, basic auth, and mutual TLS. (3) Scripted REST APIs — expose ServiceNow functionality as REST endpoints for external systems to call. Supports versioning, rate limiting, and CORS configuration. (4) MID Server — on-premises proxy that enables secure communication between ServiceNow and internal systems behind firewalls. Required for Discovery, Service Mapping, and Orchestration. (5) Import Sets and Transform Maps — bulk data import from CSV, JDBC, LDAP, and REST sources with configurable field mapping and transformation scripts. (6) JDBC connectors — direct database connectivity to Oracle, SQL Server, MySQL, PostgreSQL, and more for real-time data access. (7) Event-based integration — inbound REST API for receiving events from monitoring tools, feeding into Event Management for correlation and alerting. Design principle: prefer spoke-based integrations over custom scripted solutions for maintainability and upgrade safety."),

    ("Performance Analytics", "Technical", "Analytics",
     "Performance Analytics provides real-time visibility into IT and business service performance through KPIs, dashboards, scorecards, and benchmarking. Key components: (1) Indicators — measurable metrics tied to Now Platform data. Out-of-the-box indicators cover ITSM (MTTR, first-contact resolution, SLA compliance, backlog age), CSM (CSAT, handle time, resolution rate), and HRSD (case volume, time to resolution, self-service adoption). Custom indicators can be created for any table. (2) Dashboards — interactive visualizations with drill-down capabilities. Widget types include time series, bar charts, pivot tables, geographic maps, and scorecards. Dashboards support role-based access and real-time auto-refresh. (3) Benchmarks — compare your organization's performance against anonymized peer data from the ServiceNow customer base. Available for ITSM, ITOM, and CSM metrics. (4) Analytics Center — AI-powered insights that proactively surface trends, anomalies, and recommendations. Answers natural language questions about service performance. (5) Reporting — scheduled and ad-hoc report generation with export to PDF, Excel, and CSV. Distribution via email or Service Portal embedding. (6) Scorecards — balanced scorecard methodology for tracking strategic objectives across IT and business services. Link operational metrics to business outcomes for executive visibility."),

    ("Predictive Intelligence", "Technical", "AI/ML",
     "Predictive Intelligence brings machine learning capabilities natively into the Now Platform for intelligent automation across all modules. Core ML capabilities: (1) Classification — automatically categorize, prioritize, and assign incoming tickets based on historical patterns. Models train on your organization's data and typically achieve 92-97% accuracy for category prediction and 85-90% for assignment group prediction. (2) Similarity — identify similar past incidents or cases to suggest resolution steps. Uses natural language processing to compare ticket content, finding relevant matches even with different terminology. (3) Clustering — group related records to identify trends and patterns. Used in Problem Management to identify recurring incident clusters that may indicate underlying problems. (4) Regression — predict continuous values like case resolution time or SLA breach probability. Enables proactive workload management and resource allocation. Training and deployment: Models are trained on your instance data using AutoML-optimized algorithms. Training runs can be scheduled to incorporate new data continuously. Model performance is tracked through accuracy metrics and confusion matrices. Predictive Intelligence requires ITSM Pro, CSM Enterprise, or HRSD Pro licensing. Integration with Now Assist extends ML predictions with generative AI explanations and recommendations."),

    ("Virtual Agent Administration", "Technical", "AI",
     "Virtual Agent provides conversational AI experiences for employees and customers on the Now Platform. Administration guide for enterprise deployments: (1) Topic design — topics define conversational flows that guide users to resolution. Use the Topic Designer visual editor to create decision trees with user prompts, entity extraction, and backend actions. Pre-built topics are available for ITSM (password reset, VPN issues, software requests), HRSD (PTO balance, benefits inquiry), and CSM (order status, returns). (2) NLU models — Natural Language Understanding models classify user intents and extract entities (dates, numbers, choices). Train NLU models with at least 50-100 utterance examples per intent for production accuracy. Support for 15+ languages including English, Spanish, French, German, Japanese, and Portuguese. (3) Multi-channel deployment — Virtual Agent can be embedded in Service Portal, Employee Center, Microsoft Teams, Slack, and mobile apps. Each channel shares the same topic definitions but adapts the interface to the platform. (4) Now Assist integration — enables free-form conversational AI beyond structured topics, using generative AI to answer questions from knowledge base content. (5) Analytics — track deflection rate, user satisfaction, topic completion rates, and fallback frequency to continuously improve conversational experiences. Target 40-60% deflection rate for common IT requests."),

    ("ServiceNow API Reference", "Technical", "API",
     "ServiceNow provides comprehensive APIs for integration and automation. Core APIs: (1) Table API — CRUD operations on any ServiceNow table. Endpoints: GET /api/now/table/{tableName}, POST, PUT, PATCH, DELETE. Supports dot-walking for related records, encoded queries for filtering, and pagination with sysparm_limit and sysparm_offset. (2) Aggregate API — statistical queries (COUNT, SUM, AVG, MIN, MAX) against tables without retrieving individual records. Efficient for dashboards and reporting integrations. (3) Import Set API — bulk data loading with transform map support. POST data to staging tables, then transform to target tables with field mapping and coalescing. (4) Attachment API — upload and download file attachments associated with any record. Supports multipart form data up to configurable size limits. (5) CMDB API — specialized endpoints for CI creation, relationship management, and identification/reconciliation using identification rules. (6) Scripted REST APIs — custom REST endpoints with full server-side scripting capabilities. Define resources, query parameters, and response formats. Supports API versioning and rate limiting. Authentication options: Basic Auth, OAuth 2.0 (Authorization Code and Client Credentials flows), and API key via session-based tokens. Rate limiting is configurable per user and per API. All API calls generate audit records for security compliance. ServiceNow also provides GraphQL API for flexible querying and the Batch API for executing multiple API calls in a single request."),

    ("Security Architecture Guide", "Technical", "Security",
     "ServiceNow's security architecture provides defense-in-depth protection for enterprise workflow data. Key security layers: (1) Access control — Role-based access control (RBAC) with over 100 pre-defined roles for ITSM, CSM, HRSD, and platform administration. Access Control Lists (ACLs) provide fine-grained control at the table, row, column, and field level. Contextual security applies conditions based on user attributes, record properties, and relationships. (2) Encryption — AES-256 encryption for data at rest and TLS 1.2+ for data in transit. Column-level encryption for sensitive fields like SSN, credit card numbers, and passwords. Edge Encryption enables customer-managed encryption keys that encrypt data before it reaches ServiceNow. (3) Authentication — SSO via SAML 2.0 and OIDC. Multi-factor authentication (MFA) with support for TOTP, push notifications, and hardware tokens. Adaptive authentication with risk-based step-up for anomalous access patterns. (4) Data classification — automated sensitive data detection and tagging with configurable policies. (5) IP access control — restrict instance access to approved IP ranges and VPN endpoints. (6) Audit and compliance — comprehensive audit log of all record changes, login events, and administrative actions. SOC 1, SOC 2, ISO 27001, HIPAA, FedRAMP High, and PCI-DSS certifications."),

    ("Instance Management", "Technical", "Operations",
     "Managing ServiceNow instances across the development lifecycle requires structured processes for upgrades, cloning, and environment management. Key practices: (1) Instance strategy — maintain at least three instances: Development (dev), Test/QA (test), and Production (prod). Larger organizations add a staging instance for pre-production validation. Each instance has its own URL (e.g., company-dev.service-now.com). (2) Upgrades — ServiceNow releases two major versions per year (named alphabetically: Washington, Xanadu, etc.). Upgrades are scheduled in advance and applied automatically to non-production instances first. Use the Upgrade Center to track compatibility of customizations and plugins. Run the Instance Scan to identify potential upgrade issues. (3) Cloning — production-to-sub-production cloning refreshes lower environments with real data. Clone preserves data but can exclude specific tables. Data masking is available for sensitive information. Schedule regular clones to keep dev/test environments representative. (4) Update Sets — the primary mechanism for moving customizations between instances. Capture configuration changes in a named set, export as XML, and import into target instances. Use Update Set batching for related changes. (5) Automated Test Framework (ATF) — create and run automated tests for customizations, validating behavior before production promotion. Include ATF tests in your CI/CD pipeline. (6) Instance monitoring — track instance health through performance metrics, transaction response times, and scheduled job execution in the System Diagnostics module."),

    # FAQs (6 docs)
    ("Pricing and Licensing FAQ", "FAQ", "Pricing",
     "Frequently asked questions about ServiceNow pricing: Q: How is ServiceNow priced? A: ServiceNow uses a per-user subscription model with different tiers per module. ITSM is available in Standard, Pro (adds Predictive Intelligence, Virtual Agent, Performance Analytics), and Enterprise tiers. CSM, HRSD, and SecOps each have their own tiered licensing. Pricing is based on the number of 'fulfiller' users (agents who resolve requests) and optionally 'requester' users for self-service access. Q: What does ITOM licensing look like? A: ITOM is priced per node (discovered CI) rather than per user, with tiers for Visibility (Discovery + Service Mapping), Health (Event Management + AIOps), and Optimization (Cloud Management). Q: What is Now Assist pricing? A: Now Assist is an add-on to Pro and Enterprise tiers, priced per user with usage-based components for generative AI transactions. Q: Are there volume discounts? A: Yes, enterprise agreements with 3-year terms and volume commitments receive significant discounts. Multi-module bundles also offer better per-user economics. Q: What about implementation costs? A: Implementation costs vary by scope but typically range from 1-3x the annual license cost for initial deployment, handled by ServiceNow Professional Services or certified partners."),

    ("Migration from BMC FAQ", "FAQ", "Migration",
     "Frequently asked questions about migrating from BMC Remedy to ServiceNow: Q: How long does migration take? A: Typical migration timelines are 3-6 months for core ITSM, including data migration, process redesign, and user training. Complex environments with heavy customization may take 6-12 months. Q: What data can be migrated? A: Incident history, change records, CMDB data, knowledge articles, service catalog items, and user records. ServiceNow provides migration toolkits with pre-built transform maps for BMC data structures. Q: Do we need to redesign our processes? A: We recommend a 'fit-to-standard' approach where you adopt ServiceNow best practices rather than replicating BMC customizations. This reduces implementation time and ensures you benefit from platform upgrades. Q: What about our custom Remedy forms and workflows? A: Custom forms are rebuilt using ServiceNow's Form Designer. Workflows are reimplemented in Flow Designer, often with fewer steps due to out-of-the-box automation. Q: Can we run both platforms in parallel? A: Yes, parallel operations are common during migration. Integration between platforms can maintain data sync during the transition period. Q: What is the training requirement? A: ServiceNow provides role-based training: 2-3 days for agents, 5 days for administrators, and 10 days for developers. The modern UI typically requires less training than Remedy."),

    ("Security and Compliance FAQ", "FAQ", "Security",
     "Frequently asked questions about ServiceNow security and compliance: Q: Is ServiceNow SOC 2 certified? A: Yes, ServiceNow maintains SOC 1 Type II, SOC 2 Type II, ISO 27001, ISO 27017, ISO 27018, HIPAA, FedRAMP High, PCI-DSS, and ISMAP (Japan) certifications. Compliance reports are available through the CORE Security Portal. Q: Where is data hosted? A: ServiceNow operates data centers globally across North America, Europe, Asia-Pacific, and Australia. Customers select their data residency region at instance provisioning. Q: How is data encrypted? A: AES-256 encryption at rest, TLS 1.2+ in transit. Edge Encryption allows customer-managed keys for additional control over sensitive data. Q: Does ServiceNow support zero trust architecture? A: Yes, through IP access controls, MFA enforcement, session management, API rate limiting, and role-based access with contextual policies. Q: How are security patches handled? A: Critical security patches are applied proactively by ServiceNow operations. Platform patches are included in the twice-yearly upgrade cycle. Zero-day vulnerabilities are addressed with emergency patches. Q: Can we perform our own security testing? A: Yes, customers may conduct penetration testing against their non-production instances with advance notification."),

    ("AI Capabilities FAQ", "FAQ", "AI",
     "Frequently asked questions about ServiceNow AI capabilities: Q: What AI features does ServiceNow offer? A: Three main AI pillars: (1) Now Assist — generative AI for case summarization, code generation, conversational search, and virtual agent enhancement. (2) Predictive Intelligence — machine learning for ticket classification, assignment, priority prediction, and similarity matching. (3) Virtual Agent — conversational AI with NLU for self-service automation. Q: Does ServiceNow use my data to train AI models? A: Predictive Intelligence models are trained exclusively on your instance data and are not shared across customers. Now Assist uses ServiceNow's domain-specific LLMs that were trained on anonymized workflow data, but your specific data is not used to train the base models. Q: What is the accuracy of Predictive Intelligence? A: Typical accuracy after training: 92-97% for category prediction, 85-90% for assignment group prediction, and 80-85% for priority prediction. Accuracy improves with more training data and consistent categorization practices. Q: Can we bring our own AI models? A: Yes, the Now Platform supports integration with external AI services through Integration Hub, including connections to OpenAI, Azure OpenAI, Google Vertex AI, and custom ML endpoints. Q: What data is needed for AI training? A: Predictive Intelligence requires minimum 10,000 historical records with consistent categorization for best results."),

    ("Getting Started FAQ", "FAQ", "Onboarding",
     "Frequently asked questions about getting started with ServiceNow: Q: How do I get started? A: Start with ITSM as the foundation module. A typical ITSM implementation takes 8-12 weeks with a certified implementation partner. Begin with incident management and service catalog, then add change management and problem management. Q: Do I need a ServiceNow partner for implementation? A: While not required, it's strongly recommended. Certified partners (Accenture, Deloitte, KPMG, DXC Technology, Infosys) bring best practices and accelerators that reduce implementation time. Q: How do I get a development instance? A: Sign up for a Personal Developer Instance (PDI) at developer.servicenow.com — it's free and includes access to all platform features for learning and testing. Q: What training is available? A: ServiceNow offers extensive training through Now Learning (nowlearning.servicenow.com): Fundamentals for administrators, Implementation for developers, and specialization tracks for ITSM, CSM, HRSD, and ITOM. Certification exams are available for all roles. Q: How long until we see ROI? A: Most organizations see measurable improvements within 90 days of go-live: 20-30% reduction in ticket handling time, 30-40% increase in self-service requests, and improved SLA compliance. Full ROI is typically realized within 12-18 months."),

    ("Administration FAQ", "FAQ", "Admin",
     "Frequently asked questions about ServiceNow instance administration: Q: How many users can an instance support? A: There is no hard limit on users per instance. Enterprise instances commonly support 100,000+ requesters and 5,000+ fulfiller agents. Performance is managed through ServiceNow's auto-scaling infrastructure. Q: How do I manage customizations safely? A: Follow ServiceNow's best practice hierarchy: (1) Configure before customize — use out-of-the-box settings and properties. (2) Use platform features — Flow Designer, UI Builder, and App Engine before writing code. (3) Script only when necessary — and always use scoped applications to isolate custom code. Q: How do I handle upgrades with customizations? A: Use the Upgrade Center to identify customization conflicts before each release. Skipped records (customized platform files) need manual review. Automated Test Framework (ATF) validates that customizations work after upgrades. Q: Can I automate instance administration? A: Yes, the ServiceNow CLI and REST API enable scripted administration. Update Sets and the App Repository automate promotion between instances. The Deployment Pipeline feature in App Engine Management Center provides CI/CD capabilities. Q: How do I monitor instance performance? A: System Diagnostics provides real-time metrics on transaction response times, memory usage, and scheduled job performance. Slow transaction logs identify performance bottlenecks."),

    # Industry Playbooks (6 docs)
    ("Financial Services Industry Guide", "Industry", "Financial Services",
     "ServiceNow in Financial Services addresses regulatory compliance, operational resilience, and customer experience transformation. Key use cases: (1) Regulatory compliance workflows — automate DORA (Digital Operational Resilience Act) compliance with integrated risk assessment, ICT incident management, and third-party risk management. GRC module provides continuous compliance monitoring with automated evidence collection. (2) Operational resilience — ITOM with Service Mapping builds real-time dependency maps of critical banking services, enabling impact analysis for infrastructure changes and incidents. Event Management correlates alerts across trading platforms, core banking systems, and payment networks. (3) Fraud case management — Security Operations workflows for investigating and resolving fraud incidents with automated enrichment from fraud detection systems. (4) Risk management — Integrated Risk Management provides risk assessment workflows, control testing, and audit management. (5) Customer onboarding — CSM workflows automate KYC processes, document collection, and account activation across channels. ServiceNow's financial services customers include 8 of the top 10 global banks. Typical deployment starts with ITSM + ITOM for IT operational resilience, then expands to GRC for regulatory compliance and CSM for customer service transformation."),

    ("Healthcare Industry Guide", "Industry", "Healthcare",
     "ServiceNow in Healthcare drives clinical workflow automation, patient experience improvement, and regulatory compliance. Key use cases: (1) Clinical workflow automation — streamline clinical device management, biomedical engineering work orders, and pharmacy inventory workflows on the Now Platform. Integration Hub connects to Epic, Cerner, and other EHR systems for bi-directional data flow. (2) Patient experience — CSM-based patient service portals for appointment scheduling, referral tracking, and billing inquiries with Virtual Agent for common questions. (3) HIPAA compliance — built-in controls for PHI handling including encryption, access controls, audit logging, and data classification. Security Operations automates HIPAA breach notification workflows. (4) Medical device management — CMDB tracks medical devices as CIs with maintenance schedules, warranty tracking, and FDA compliance status. Discovery automates inventory of network-connected devices. (5) Employee experience — HRSD for healthcare-specific onboarding workflows including credentialing verification, compliance training tracking, and license management. Nurse and clinician onboarding reduced from 6 weeks to 2 weeks. (6) IT operations — ITOM ensures availability of critical clinical systems through proactive monitoring and AIOps-driven incident prevention. ServiceNow maintains HIPAA BAA coverage and processes data in HIPAA-compliant data centers."),

    ("Manufacturing Industry Guide", "Industry", "Manufacturing",
     "ServiceNow in Manufacturing enables connected operations, OT/IT convergence, and supply chain workflow automation. Key use cases: (1) Connected operations — ITOM extends beyond traditional IT to monitor operational technology (OT) systems on the manufacturing floor. Service Mapping creates dependency maps between production systems, MES, SCADA, and enterprise IT. Event Management correlates alerts from manufacturing IoT sensors with IT infrastructure events. (2) Supply chain workflows — custom App Engine applications for supplier management, procurement approvals, and logistics tracking with Integration Hub connecting to SAP, Oracle, and specialized MES systems. (3) Quality management — workflow automation for non-conformance tracking, corrective action requests (CARs), and quality audit management. Integration with quality management systems (QMS) enables closed-loop quality processes. (4) Maintenance management — planned and unplanned maintenance workflows with integration to predictive maintenance ML models. CMDB tracks manufacturing assets with maintenance history, warranty status, and criticality ratings. (5) Employee safety — HRSD workflows for safety incident reporting, investigation tracking, and compliance management. Mobile-optimized for shop floor workers. Manufacturing customers report 40% reduction in unplanned downtime, 30% improvement in maintenance efficiency, and 50% faster quality issue resolution."),

    ("Technology Industry Guide", "Industry", "Technology",
     "ServiceNow in Technology companies supports DevOps integration, SRE workflows, and customer success automation. Key use cases: (1) DevOps integration — bi-directional integration between ServiceNow and DevOps tools (GitHub, GitLab, Jenkins, Azure DevOps) for change management automation. Changes linked to CI/CD pipelines enable automated risk assessment and approval workflows. (2) SRE and incident management — ITSM integrated with monitoring tools (PagerDuty, Datadog, Splunk, New Relic) for automated incident creation, intelligent routing, and post-incident review workflows. On-call scheduling and escalation management built into the platform. (3) Customer success automation — CSM workflows for tracking customer health scores, product adoption metrics, and renewal management. Proactive alerting when customer engagement drops below thresholds. (4) Product-led growth support — self-service portals for product trials, developer documentation, and community forums built on Service Portal. (5) Internal IT at scale — technology companies with 10,000+ employees use ITSM and HRSD to manage the complexity of supporting a highly technical workforce. Virtual Agent handles developer-specific requests like environment provisioning and access management. (6) Security compliance — SecOps automates SOC 2, ISO 27001, and customer security questionnaire responses through GRC integration."),

    ("Government Industry Guide", "Industry", "Government",
     "ServiceNow in Government enables citizen service delivery, GRC compliance, and zero trust security implementation. Key capabilities: (1) FedRAMP authorization — ServiceNow holds FedRAMP High authorization (Impact Level 4 and 5 for DoD), enabling deployment for federal, state, and local government agencies handling controlled unclassified information. (2) Citizen services portal — Service Portal provides public-facing digital service delivery for permits, licenses, benefits applications, and service requests. Virtual Agent provides 24/7 citizen support. (3) GRC workflows — Governance, Risk, and Compliance module automates continuous authority to operate (cATO), NIST 800-53 control assessment, FISMA reporting, and audit management. Policy and Compliance Management tracks regulatory changes and automates impact analysis. (4) Zero trust security — Security Operations supports zero trust architecture implementation with continuous monitoring, identity-aware access controls, and automated incident response aligned with CISA's Zero Trust Maturity Model. (5) IT modernization — ITSM and ITOM support federal IT modernization mandates including cloud migration tracking, technology business management (TBM), and FITARA compliance. (6) Cross-agency collaboration — the platform enables shared services across departments while maintaining data isolation and role-based access controls required by government security policies."),

    ("Telecommunications Industry Guide", "Industry", "Telecommunications",
     "ServiceNow in Telecommunications addresses network operations, customer service, and 5G service management. Key use cases: (1) Network operations — ITOM provides end-to-end visibility across physical and virtual network infrastructure. Event Management correlates network alarms from OSS/BSS systems, reducing NOC alert fatigue by 80%. Service Mapping traces customer-facing services through the network stack for rapid impact analysis. (2) Customer service for telcos — CSM with industry-specific workflows for service provisioning, outage communication, and billing dispute resolution. Omnichannel support across call center, chat, social media, and retail store channels. (3) 5G service management — ITOM and service catalog workflows for managing 5G network slice provisioning, edge computing deployments, and IoT service activation. CMDB extended with telco-specific CI classes for network elements. (4) Order management — App Engine-based order orchestration workflows that coordinate across network provisioning, CRM, and billing systems. Integration Hub connects to Amdocs, Ericsson, Nokia, and other telco-specific platforms. (5) Field service — dispatching and managing field technicians for network installations, repairs, and upgrades with mobile-optimized work order management. (6) Regulatory compliance — GRC workflows for telecommunications-specific regulations including data retention, lawful intercept, and service availability SLAs."),

    # Additional (8 docs)
    ("Customer Health Scoring Framework", "Sales Playbook", "Customer Success",
     "ServiceNow customer health scoring framework for predicting expansion and renewal risk. Health indicators are organized into four categories: (1) Adoption metrics — active fulfiller users as percentage of licensed seats (green: >80%, yellow: 50-80%, red: <50%), module utilization across licensed products, self-service adoption rate, Virtual Agent deflection rate, and new workflow deployment velocity. (2) Engagement metrics — executive sponsor engagement level, participation in ServiceNow events and webinars, Community forum activity, and response rate to customer success outreach. (3) Technical health — instance upgrade currency (within one release is green), CMDB accuracy score, number of active integrations, customization debt (skipped updates as percentage of total), and performance metrics (transaction response times). (4) Business outcomes — measured improvement in MTTR, SLA compliance trends, ticket volume trends, self-service adoption trajectory, and reported ROI metrics. Scoring model: each indicator is weighted and scored 1-5. Composite score determines health status: Green (4.0+), Yellow (3.0-3.9), Red (<3.0). Action playbook: Green accounts receive proactive expansion discussions and reference requests. Yellow accounts get executive QBR with value reassessment. Red accounts trigger customer success escalation with dedicated resources and competitive defense preparation."),

    ("Knowledge Management Best Practices", "Technical", "Knowledge",
     "Implementing effective knowledge management on the Now Platform using KCS (Knowledge-Centered Service) methodology: (1) Knowledge article lifecycle — create articles during incident resolution using the KCS 'solve-and-evolve' approach. Agents draft articles while resolving incidents, subject matter experts review and approve, and continuous improvement is driven by user feedback and usage analytics. (2) Article structure — use ServiceNow's knowledge article templates with standard sections: symptom, cause, resolution, and related articles. Apply knowledge categories and taxonomy consistently for effective search. (3) Self-service optimization — configure the Knowledge Base widget on Service Portal and Employee Center with AI-powered search that understands natural language queries. Track article view-to-resolution ratios to identify effective content. Target: 20% of incidents resolved through self-service knowledge articles. (4) Now Assist for Knowledge — automatically generates draft knowledge articles from resolved incident data, including structured symptom-cause-resolution formatting. Content is reviewed by knowledge authors before publication. (5) Metrics and governance — track Knowledge Management KPIs: article creation rate, article reuse rate, self-service deflection percentage, article feedback scores, and knowledge gap analysis (incidents without matching articles). (6) Multi-language support — ServiceNow supports article translation workflows with automated translation and human review processes."),

    ("Workflow Automation Patterns", "Technical", "Workflow",
     "Common workflow automation patterns on the Now Platform: (1) Approval workflows — multi-level approval chains with conditional routing based on request type, amount, and department. Use Flow Designer's approval action with parallel and serial approval patterns. Configure delegation rules and time-based escalation for unresponsive approvers. (2) SLA management — define SLA definitions with response and resolution time targets per priority. Configure breach notifications at 50%, 75%, and 90% of SLA duration. Use SLA Workflow to trigger escalation actions automatically. Performance Analytics dashboards track SLA compliance trends across teams and categories. (3) Notification patterns — use Flow Designer notification actions with dynamic recipients based on record fields, groups, and escalation rules. Support for email, SMS, push notification, and Microsoft Teams/Slack channels. Implement notification preferences to prevent alert fatigue. (4) Escalation rules — time-based and condition-based escalation using SLA-driven workflows. Functional escalation routes to specialist groups; hierarchical escalation notifies management. (5) Scheduled workflows — batch processing patterns for periodic tasks like license reclamation, stale ticket cleanup, and compliance reporting. Use scheduled flows with error handling and completion notifications. (6) Event-driven patterns — inbound event processing from monitoring tools through Event Management, triggering incident creation, notification, and automated remediation workflows."),

    ("Developer Guide", "Technical", "Development",
     "Scripting and development guide for the Now Platform: (1) Business Rules — server-side scripts that execute when records are displayed, inserted, updated, or deleted. Use before rules for data validation and field manipulation, after rules for cross-table updates and notifications, and async rules for non-blocking operations. Avoid complex logic in business rules; delegate to Script Includes for reusability. (2) Script Includes — reusable server-side JavaScript libraries. Extend AbstractAjaxProcessor for client-callable scripts. Use the prototype pattern for class-based implementations. Always scope Script Includes to the appropriate application. (3) Client Scripts — browser-side scripts for form interactivity. Types: onLoad (form opens), onChange (field value changes), onSubmit (form saves). Minimize client scripts for performance; prefer UI Policies for simple show/hide and mandatory field logic. (4) UI Policies — declarative rules for form behavior without coding. Set field visibility, mandatory status, and read-only status based on conditions. UI Policies execute before Client Scripts and are preferred for simple form logic. (5) Scoped Applications — isolate custom development in application scopes for upgrade safety and portability. Each scope has its own tables, scripts, and security context. (6) Automated Test Framework — create automated tests for business rules, workflows, and UI behavior. Include ATF suites in Update Set promotion processes."),

    ("Partner Ecosystem Guide", "Sales Playbook", "Partners",
     "ServiceNow's partner ecosystem drives implementation success and market reach. Implementation partners by tier: (1) Elite partners — Accenture, Deloitte, KPMG, DXC Technology, Infosys, and Wipro. These partners maintain 500+ certified consultants each and handle the largest enterprise implementations. Typical engagement: 6-12 months, $2M-$10M implementation value. (2) Premier partners — Cognizant, TCS, HCL, Capgemini, EY, and PwC. Strong in specific verticals or modules with 200+ certified consultants. (3) Specialist partners — focused on specific industries (healthcare, government) or modules (SecOps, ITOM). Smaller teams but deep expertise. (4) Technology partners — ISVs that build applications on the ServiceNow Store. Key technology partners include Dynatrace, Splunk, PagerDuty, Okta, and CrowdStrike for security integrations. (5) ServiceNow Professional Services — ServiceNow's own delivery team handles strategic accounts and complex deployments. Offers Expert Services for ongoing advisory and optimization. Partner selection criteria for customers: industry experience, module specialization, geographic coverage, team availability, and customer references. ServiceNow maintains partner performance scorecards based on customer satisfaction, certification levels, and delivery quality. When positioning against partners, emphasize ServiceNow's direct involvement in ensuring implementation success through Customer Success Management."),

    ("Digital Transformation Framework", "Sales Playbook", "Strategy",
     "ServiceNow's digital transformation framework guides enterprises through four stages of workflow maturity: (1) Digitize — replace manual processes and spreadsheets with structured digital workflows. Entry point is typically ITSM for IT operations and service catalog for employee self-service. Quick wins include automated ticket routing, self-service catalog deployment, and knowledge base creation. Timeline: 3-6 months. Outcomes: 30% reduction in manual effort, baseline metrics established. (2) Optimize — apply analytics and AI to improve process efficiency. Deploy Performance Analytics for KPI tracking and benchmarking. Implement Predictive Intelligence for automated categorization and routing. Optimize CMDB with Discovery and Service Mapping. Timeline: 6-12 months. Outcomes: 40-50% improvement in MTTR, 50%+ self-service adoption. (3) Transform — expand beyond IT to enterprise-wide workflows. Deploy HRSD for employee experience, CSM for customer operations, and SecOps for security. Build custom applications with App Engine for department-specific needs. Timeline: 12-24 months. Outcomes: platform consolidation savings of 30-40%, enterprise-wide workflow visibility. (4) Innovate — leverage AI and automation for proactive, predictive operations. Deploy Now Assist across all modules, implement AIOps through ITOM Health, and build intelligent agents for autonomous workflow execution. Timeline: 24+ months. Outcomes: predictive operations, autonomous remediation, continuous improvement driven by AI insights."),

    ("ITSM Maturity Model", "Technical", "ITSM Maturity",
     "ServiceNow's ITSM Maturity Model provides a framework for assessing and improving IT service management capabilities across five levels: Level 1 — Reactive: IT operates in firefighting mode. Incidents are handled ad-hoc without formal processes. No CMDB, manual ticket creation via email or walk-up. Metrics are not tracked consistently. Typical characteristics: MTTR >8 hours, <10% self-service, no change management process. Level 2 — Proactive: Basic ITSM processes are defined and followed. Incident, change, and request management are formalized. CMDB exists but has <50% accuracy. Self-service catalog is available but adoption is low. Typical characteristics: MTTR 4-8 hours, 10-25% self-service, basic SLA tracking. Level 3 — Service-oriented: ITSM is aligned with business services. Service Mapping links infrastructure to business outcomes. Performance Analytics provides KPI dashboards. Virtual Agent handles common requests. Typical characteristics: MTTR 2-4 hours, 25-50% self-service, CMDB >70% accurate. Level 4 — Business-aligned: IT is a strategic partner to the business. Predictive Intelligence automates ticket handling. ITOM provides proactive monitoring. Cross-departmental workflows (HRSD, CSM) are integrated. Typical characteristics: MTTR 1-2 hours, 50-70% self-service, CMDB >85% accurate. Level 5 — Optimized: AI-driven autonomous operations. Now Assist provides generative AI across all workflows. AIOps predicts and prevents incidents. Continuous improvement driven by analytics. Typical characteristics: MTTR <1 hour, >70% self-service, CMDB >95% accurate."),

    ("GenAI Use Case Prioritization", "Sales Playbook", "GenAI Strategy",
     "Framework for helping ServiceNow customers prioritize generative AI use cases with Now Assist: Tier 1 — Quick wins (2-4 weeks deployment): Incident summarization — Now Assist automatically generates concise summaries of incident records including timeline, actions taken, and resolution, saving agents 3-5 minutes per ticket. Knowledge article generation — auto-draft articles from resolved incidents for knowledge author review. Search enhancement — AI-powered natural language search across knowledge bases and service catalogs. Virtual Agent enhancement — generative AI responses for unstructured questions beyond pre-defined topics. Tier 2 — Medium effort (1-3 months): Code generation in App Engine — Now Assist generates Business Rules, Script Includes, and Flow Designer configurations from natural language descriptions. Chat summarization for CSM — condense lengthy customer conversations for case handoffs. Change risk assessment — AI-powered analysis of change request risk based on historical patterns and CMDB dependencies. Tier 3 — Strategic initiatives (3-6+ months): Autonomous incident resolution — AI-driven runbook execution for known error patterns without human intervention. Predictive resource planning — AI forecasting of ticket volumes, staffing needs, and SLA risk. Multi-modal AI — processing images, screenshots, and documents attached to incidents for automated diagnosis. Evaluation criteria: (1) Data readiness — minimum 10K records for training. (2) Process maturity — ITSM maturity Level 3+ recommended. (3) Business impact — quantified time savings and quality improvement. (4) Risk tolerance — human-in-the-loop requirement for each use case."),
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

hr = '─' * 40
print(f"  {hr}")
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
