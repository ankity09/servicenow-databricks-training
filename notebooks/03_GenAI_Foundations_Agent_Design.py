# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Module 3: GenAI Foundations & Agent System Design
# MAGIC
# MAGIC **Databricks Training for ServiceNow | Afternoon Session - Part 1**
# MAGIC
# MAGIC Welcome to the afternoon session! This morning we explored the Databricks Data Intelligence Platform,
# MAGIC built a Lakehouse, and trained ML models. Now we shift to the frontier: **Generative AI and Agentic Systems**.
# MAGIC
# MAGIC ### What You'll Learn
# MAGIC | Section | Topic | Hands-On |
# MAGIC |---------|-------|----------|
# MAGIC | 1 | From Prompts to Autonomous Agents | Conceptual framework |
# MAGIC | 2 | Foundation Models & Prompt Engineering | Call LLMs via API |
# MAGIC | 3 | Data Engineering for RAG — Vector Search | Build a vector index |
# MAGIC | 4 | Equipping Agents with Tools | Create 3 agent tools |
# MAGIC | 5 | Agent Bricks & MCP | Architecture patterns |
# MAGIC
# MAGIC **Estimated Duration:** 75 minutes
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Configuration

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")
print(f"Catalog: {catalog} | Schema: {schema} | User: {username}")

# COMMAND ----------

# Verify our GTM data is available from the morning session
tables = ["gtm_accounts", "gtm_contacts", "gtm_opportunities", "gtm_activities",
          "gtm_campaigns", "gtm_campaign_members", "gtm_lead_scores", "gtm_knowledge_base"]

print("Checking tables from the morning session...\n")
for t in tables:
    try:
        count = spark.sql(f"SELECT COUNT(*) FROM {catalog}.{schema}.{t}").first()[0]
        print(f"  {t:<30} {count:>6} rows")
    except Exception as e:
        print(f"  {t:<30} NOT FOUND - {str(e)[:60]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 1: From Prompts to Autonomous Agents
# MAGIC
# MAGIC ## The AI Application Spectrum
# MAGIC
# MAGIC Not every problem needs an autonomous agent. Understanding where your use case falls on the
# MAGIC complexity spectrum is the first — and most important — design decision.
# MAGIC
# MAGIC ```
# MAGIC  Complexity & Autonomy ──────────────────────────────────────────────────────────►
# MAGIC
# MAGIC  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
# MAGIC  │              │   │              │   │              │   │              │   │              │
# MAGIC  │   Simple     │   │   Prompt     │   │     RAG      │   │ Tool-Calling │   │ Multi-Agent  │
# MAGIC  │   Prompts    │──►│   Chains     │──►│  Pipelines   │──►│   Agents     │──►│   Systems    │
# MAGIC  │              │   │              │   │              │   │              │   │              │
# MAGIC  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
# MAGIC        │                   │                  │                  │                   │
# MAGIC   "Summarize            "Extract           "Answer            "Research          "Coordinate
# MAGIC    this text"          info, then         questions          accounts,           multiple
# MAGIC                       classify,           from our           search docs,        specialists
# MAGIC                       then draft"         company            analyze data"       to handle
# MAGIC                                           docs"                                  complex
# MAGIC                                                                                  workflows"
# MAGIC  ─────────────────────────────────────────────────────────────────────────────────────────────
# MAGIC  Low latency                                                              Higher latency
# MAGIC  Deterministic                                                            More autonomous
# MAGIC  Easy to test                                                             Harder to test
# MAGIC  Low cost                                                                 Higher cost
# MAGIC ```
# MAGIC
# MAGIC ### When to Use Each Pattern
# MAGIC
# MAGIC | Pattern | Best For | Example | Latency |
# MAGIC |---------|----------|---------|---------|
# MAGIC | **Simple Prompt** | Classification, summarization, extraction | "Classify this support ticket" | < 2s |
# MAGIC | **Prompt Chain** | Multi-step transformations with deterministic flow | "Extract entities → Enrich → Generate email" | 5-10s |
# MAGIC | **RAG** | Q&A over proprietary documents | "What's our pricing for Enterprise tier?" | 3-5s |
# MAGIC | **Tool-Calling Agent** | Dynamic tasks requiring data access | "Analyze our top accounts and draft outreach" | 10-30s |
# MAGIC | **Multi-Agent System** | Complex workflows with specialized roles | "Plan, research, write, and review a proposal" | 30s-5min |
# MAGIC
# MAGIC > **Key Insight:** Start simple. Move up the complexity ladder only when the simpler pattern can't solve your problem.
# MAGIC > An agent that uses RAG + SQL tools covers 80% of enterprise use cases.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 2: Foundation Models & Prompt Engineering
# MAGIC
# MAGIC Databricks provides a **Foundation Model API** — a unified, OpenAI-compatible interface for
# MAGIC accessing LLMs. This means you can use the familiar OpenAI SDK to call models hosted on Databricks,
# MAGIC with no data leaving your environment.
# MAGIC
# MAGIC ### Available Models in This Workshop
# MAGIC | Model | Endpoint | Use Case |
# MAGIC |-------|----------|----------|
# MAGIC | Llama 3.3 70B Instruct | `databricks-meta-llama-3-3-70b-instruct` | General purpose, tool calling |
# MAGIC | GPT 5.4 | `databricks-gpt-5-4` | Advanced reasoning |
# MAGIC | Claude Sonnet 4.6 | `databricks-claude-sonnet-4-6` | Analysis, writing |
# MAGIC | GTE-Large-EN | `databricks-gte-large-en` | Text embeddings |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 — Your First Foundation Model API Call

# COMMAND ----------

# MAGIC %pip install openai databricks-sdk --quiet

# COMMAND ----------

# Fix typing_extensions path precedence on serverless compute
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "typing_extensions>=4.5", "--target", "/tmp/pip_overrides", "--upgrade", "--quiet"])
sys.path.insert(0, "/tmp/pip_overrides")

# Reload typing_extensions from the updated path
import importlib
if "typing_extensions" in sys.modules:
    del sys.modules["typing_extensions"]
import typing_extensions
importlib.reload(typing_extensions)

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

from openai import OpenAI

# Create the client pointing to Databricks Model Serving
client = OpenAI(
    api_key=api_token,
    base_url=f"{workspace_url}/serving-endpoints"
)

# A simple completion — just like calling OpenAI, but on Databricks
response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[
        {"role": "user", "content": "In one paragraph, explain why enterprises should care about GenAI for their Go-To-Market strategy."}
    ],
    max_tokens=300,
    temperature=0.7
)

print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 — Prompt Engineering Techniques
# MAGIC
# MAGIC The way you structure your prompt dramatically impacts output quality. Let's explore the four
# MAGIC most important techniques using our GTM data.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 1: Zero-Shot Classification
# MAGIC The model classifies without any examples — relying entirely on its training knowledge.

# COMMAND ----------

# Zero-shot: Classify a lead based on a description
lead_description = """
Sarah Chen, VP of Engineering at a Fortune 500 financial services company.
She attended our webinar on real-time data pipelines, downloaded the Lakehouse whitepaper,
and requested a custom demo. Her team currently uses Snowflake but is evaluating alternatives
for their ML workloads. Budget is approved for Q2.
"""

response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[
        {
            "role": "user",
            "content": f"""Classify the following lead into exactly one category:
HOT (ready to buy), WARM (interested, needs nurturing), or COLD (low intent).

Respond with the classification and a one-sentence justification.

Lead Description:
{lead_description}"""
        }
    ],
    max_tokens=150,
    temperature=0.0  # Low temperature for deterministic classification
)

print("ZERO-SHOT CLASSIFICATION")
print("=" * 50)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 2: Few-Shot Extraction
# MAGIC Providing examples teaches the model your exact desired output format.

# COMMAND ----------

# Few-shot: Extract key information from sales notes
sales_notes = """
Had a great call with Mike Rodriguez (CTO) at Acme Corp today. They're running
on AWS with a mix of Redshift and EMR. Pain points are data silos between their
analytics and ML teams, plus governance headaches with PII data. They have a
$2M annual budget for data infrastructure. Next step is a technical deep-dive
with their platform team next Thursday.
"""

response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[
        {
            "role": "user",
            "content": f"""Extract structured information from sales call notes. Follow these examples exactly.

EXAMPLE 1:
Notes: "Spoke with Jane Lee, Head of Data at GlobalBank. They use Azure Synapse.
Main concern is cost — spending $500K/yr. Want to evaluate Databricks for cost savings.
Follow-up meeting scheduled for Monday."
Extracted:
- Contact: Jane Lee (Head of Data)
- Company: GlobalBank
- Current Stack: Azure Synapse
- Pain Points: High costs ($500K/yr)
- Budget: Not specified
- Next Step: Follow-up meeting Monday
- Urgency: Medium

EXAMPLE 2:
Notes: "Cold outreach to Tom Park, Dir of Analytics at RetailMax. Left voicemail.
No response yet. They were seen at a Snowflake event last month."
Extracted:
- Contact: Tom Park (Dir of Analytics)
- Company: RetailMax
- Current Stack: Likely Snowflake
- Pain Points: Unknown
- Budget: Unknown
- Next Step: Follow up on voicemail
- Urgency: Low

NOW EXTRACT FROM THESE NOTES:
{sales_notes}"""
        }
    ],
    max_tokens=300,
    temperature=0.0
)

print("FEW-SHOT EXTRACTION")
print("=" * 50)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 3: Chain-of-Thought Reasoning
# MAGIC By asking the model to "think step by step," we get more accurate and explainable results.

# COMMAND ----------

# Chain-of-thought: Analyze a deal and recommend next steps
deal_info = """
Deal: Enterprise Data Platform — TechGlobal Inc.
Stage: Negotiation
Amount: $1.2M ARR
Probability: 60%
Days in Stage: 45
Champion: VP of Data Engineering (strong advocate)
Economic Buyer: CIO (met once, seemed cautious about migration risk)
Competition: Snowflake (incumbent, 3-year contract ending in 6 months)
Last Activity: Technical POC completed successfully 2 weeks ago
Stalled Reason: Legal review of data residency requirements
"""

response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[
        {
            "role": "user",
            "content": f"""Analyze this deal and recommend the top 3 next steps to move it forward.

Think step by step:
1. First, assess the deal health based on the metrics
2. Then, identify the key risks and blockers
3. Then, evaluate the competitive dynamics
4. Finally, recommend specific, actionable next steps

Deal Information:
{deal_info}"""
        }
    ],
    max_tokens=600,
    temperature=0.3
)

print("CHAIN-OF-THOUGHT ANALYSIS")
print("=" * 50)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 4: System Prompts for Persona Setting
# MAGIC System prompts set the behavior, tone, and constraints for the model across an entire conversation.

# COMMAND ----------

# System prompt: Create a specialized GTM analyst persona
response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[
        {
            "role": "system",
            "content": """You are a senior GTM (Go-To-Market) Strategy Analyst at a leading enterprise
software company. You specialize in data-driven sales strategy.

Your communication style:
- Concise and executive-ready (use bullet points)
- Always cite data or metrics when available
- Frame recommendations in terms of revenue impact
- Use MEDDPICC sales methodology terminology when relevant
- If you don't have data to support a claim, say so explicitly

You never make things up. If asked about data you don't have, you recommend what data
to collect and why it matters."""
        },
        {
            "role": "user",
            "content": "Our win rate dropped from 35% to 28% last quarter. We think it's competitive pressure from Snowflake. What should we investigate?"
        }
    ],
    max_tokens=500,
    temperature=0.4
)

print("SYSTEM PROMPT — GTM ANALYST PERSONA")
print("=" * 50)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 — AI Playground
# MAGIC
# MAGIC The **Databricks AI Playground** provides a no-code interface for experimenting with Foundation Models.
# MAGIC
# MAGIC **What you can do in AI Playground:**
# MAGIC - Test any model endpoint with different prompts and parameters
# MAGIC - Compare outputs from multiple models side-by-side (e.g., Llama vs GPT vs Claude)
# MAGIC - Tune temperature, max tokens, top-p, and other generation parameters
# MAGIC - Test tool-calling / function-calling capabilities interactively
# MAGIC - Export working prompts directly to notebook code
# MAGIC
# MAGIC > **Try it:** Navigate to **Machine Learning → AI Playground** in the left sidebar.
# MAGIC > Paste in the chain-of-thought prompt from above and try it with different models!
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 3: Data Engineering for RAG — Vector Search
# MAGIC
# MAGIC **Retrieval-Augmented Generation (RAG)** is the most common enterprise GenAI pattern. Instead of
# MAGIC fine-tuning a model on your data, you *retrieve* relevant documents at query time and include them
# MAGIC in the prompt as context.
# MAGIC
# MAGIC ```
# MAGIC  ┌─────────────────────────────────────────────────────────────────────────┐
# MAGIC  │                        RAG Architecture                                 │
# MAGIC  │                                                                         │
# MAGIC  │   User Query ──► Embed Query ──► Vector Search ──► Top-K Documents      │
# MAGIC  │                                                          │               │
# MAGIC  │                                                          ▼               │
# MAGIC  │                                                   ┌─────────────┐        │
# MAGIC  │   User Query ─────────────────────────────────────►│   LLM       │       │
# MAGIC  │                                                   │  (with       │       │
# MAGIC  │                              Retrieved Docs ──────►│  context)    │       │
# MAGIC  │                                                   └──────┬──────┘        │
# MAGIC  │                                                          │               │
# MAGIC  │                                                     Answer               │
# MAGIC  └─────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Why Vector Search on Databricks?
# MAGIC - **Delta Sync**: Automatically keeps your vector index in sync with your Delta table
# MAGIC - **Governance**: Vectors inherit Unity Catalog permissions
# MAGIC - **Scale**: Handles millions of vectors with low-latency retrieval
# MAGIC - **Managed**: No separate vector database to operate

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 — Explore the Knowledge Base

# COMMAND ----------

# Let's look at our GTM knowledge base — this is what we'll make searchable via RAG
knowledge_df = spark.sql(f"SELECT * FROM {catalog}.{schema}.gtm_knowledge_base")
print(f"Total documents: {knowledge_df.count()}")
print(f"\nSchema:")
knowledge_df.printSchema()

# COMMAND ----------

# Show the distribution of documents by category
display(
    spark.sql(f"""
        SELECT category, subcategory, COUNT(*) as doc_count
        FROM {catalog}.{schema}.gtm_knowledge_base
        GROUP BY category, subcategory
        ORDER BY category, subcategory
    """)
)

# COMMAND ----------

# Preview a few documents to understand the content
display(
    spark.sql(f"""
        SELECT doc_id, title, category, subcategory,
               LENGTH(content) as content_length,
               LEFT(content, 200) as content_preview
        FROM {catalog}.{schema}.gtm_knowledge_base
        LIMIT 10
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 — Enable Change Data Feed
# MAGIC
# MAGIC Delta Sync Vector Search indexes require **Change Data Feed (CDF)** to be enabled on the source table.
# MAGIC CDF tracks row-level changes (inserts, updates, deletes) so the vector index can stay in sync automatically.

# COMMAND ----------

# Enable Change Data Feed on the source table
spark.sql(f"""
    ALTER TABLE {catalog}.{schema}.gtm_knowledge_base
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print(f"Change Data Feed enabled on {catalog}.{schema}.gtm_knowledge_base")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 — Generate Embeddings with AI Functions
# MAGIC
# MAGIC Databricks **AI Functions** let you call AI models directly from SQL. We'll use `ai_query()` to
# MAGIC generate embeddings for every document in our knowledge base using the `databricks-gte-large-en` model.
# MAGIC
# MAGIC This creates a new table with an `embedding` column — a dense vector representation of each document's content.

# COMMAND ----------

# Create the embeddings table using ai_query — this runs the embedding model on every row
spark.sql(f"""
    CREATE OR REPLACE TABLE {catalog}.{schema}.gtm_knowledge_embeddings AS
    SELECT *, ai_query(
        'databricks-gte-large-en',
        content,
        'ARRAY<FLOAT>'
    ) as embedding
    FROM {catalog}.{schema}.gtm_knowledge_base
""")

print(f"Embeddings table created: {catalog}.{schema}.gtm_knowledge_embeddings")

# COMMAND ----------

# Verify the embeddings
embeddings_df = spark.sql(f"""
    SELECT doc_id, title, category,
           SIZE(embedding) as embedding_dimensions
    FROM {catalog}.{schema}.gtm_knowledge_embeddings
    LIMIT 5
""")
display(embeddings_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 — Create a Vector Search Index
# MAGIC
# MAGIC Now we'll create a **Delta Sync Vector Search Index**. This index:
# MAGIC - Automatically syncs when the source Delta table changes
# MAGIC - Supports filtered similarity search
# MAGIC - Is governed by Unity Catalog ACLs
# MAGIC

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Configuration
vs_endpoint_name = "mas-3876475e-endpoint"
vs_index_name = f"{catalog}.{schema}.gtm_knowledge_vs_index"
source_table = f"{catalog}.{schema}.gtm_knowledge_base"

print(f"Vector Search Endpoint: {vs_endpoint_name}")
print(f"Index Name:             {vs_index_name}")
print(f"Source Table:           {source_table}")

# COMMAND ----------

# Check the endpoint status
try:
    endpoint = w.vector_search_endpoints.get_endpoint(vs_endpoint_name)
    print(f"Endpoint '{vs_endpoint_name}' is {endpoint.endpoint_status.state.value}")
except Exception as e:
    print(f"Error checking endpoint: {e}")

# COMMAND ----------

# Create the Delta Sync vector search index
# This will automatically compute embeddings using the specified model

try:
    # First, try to delete any existing index with the same name
    try:
        w.vector_search_indexes.delete_index(vs_index_name)
        print(f"Deleted existing index: {vs_index_name}")
        import time
        time.sleep(5)
    except Exception:
        pass  # Index doesn't exist yet — that's fine

    # Create a new Delta Sync index
    index = w.vector_search_indexes.create_index(
        name=vs_index_name,
        endpoint_name=vs_endpoint_name,
        primary_key="doc_id",
        index_type="DELTA_SYNC",
        delta_sync_index_spec={
            "source_table": source_table,
            "pipeline_type": "TRIGGERED",
            "embedding_source_columns": [
                {
                    "name": "content",
                    "embedding_model_endpoint_name": "databricks-gte-large-en"
                }
            ]
        }
    )
    print(f"Vector Search index created: {vs_index_name}")
    print(f"The index will sync automatically. Initial sync may take a few minutes.")

except Exception as e:
    print(f"Note: {e}")
    print("\nIf the index already exists, we can proceed to querying it.")

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# Check the index sync status
import time

def check_index_status(index_name, max_wait_seconds=300):
    """Poll the index status until it's ready or timeout."""
    start = time.time()
    while time.time() - start < max_wait_seconds:
        try:
            idx = w.vector_search_indexes.get_index(index_name)
            status = idx.status
            print(f"  Index status: {status.ready} | Message: {status.message if hasattr(status, 'message') else 'N/A'}")
            if status.ready:
                print("\n  Index is READY for queries!")
                return True
        except Exception as e:
            print(f"  Checking... ({e})")
        time.sleep(15)
    print(f"\n  Timeout after {max_wait_seconds}s — index may still be syncing.")
    print("  You can proceed; queries will work once the sync completes.")
    return False

print(f"Checking index: {vs_index_name}")
check_index_status(vs_index_name, max_wait_seconds=180)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 — Query the Vector Search Index
# MAGIC
# MAGIC Let's test our index with a similarity search. The index automatically handles:
# MAGIC 1. Embedding your query text using the same model
# MAGIC 2. Finding the nearest vectors in the index
# MAGIC 3. Returning the matching documents with similarity scores

# COMMAND ----------

# Test similarity search
try:
    results = w.vector_search_indexes.query_index(
        index_name=vs_index_name,
        columns=["doc_id", "title", "category", "subcategory", "content"],
        query_text="How should we handle pricing objections from enterprise customers?",
        num_results=3
    )

    print("VECTOR SEARCH RESULTS")
    print("=" * 70)
    print(f"Query: 'How should we handle pricing objections from enterprise customers?'\n")

    for i, row in enumerate(results.result.data_array):
        doc_id, title, category, subcategory, content = row[0], row[1], row[2], row[3], row[4]
        print(f"Result {i+1}:")
        print(f"  Title:       {title}")
        print(f"  Category:    {category} / {subcategory}")
        print(f"  Content:     {str(content)[:200]}...")
        print()

except Exception as e:
    print(f"Query failed (index may still be syncing): {e}")
    print("\nTry again in a minute or two. The index needs to finish its initial sync.")

# COMMAND ----------

# Let's try another search to show the versatility
try:
    results = w.vector_search_indexes.query_index(
        index_name=vs_index_name,
        columns=["doc_id", "title", "category", "content"],
        query_text="What are the technical requirements for data migration to Databricks?",
        num_results=3
    )

    print("VECTOR SEARCH RESULTS")
    print("=" * 70)
    print(f"Query: 'What are the technical requirements for data migration to Databricks?'\n")

    for i, row in enumerate(results.result.data_array):
        print(f"Result {i+1}: {row[1]} [{row[2]}]")
        print(f"  {str(row[3])[:200]}...\n")

except Exception as e:
    print(f"Query failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 4: Equipping Agents with Tools
# MAGIC
# MAGIC An agent is only as useful as the tools it can access. We'll build three tools that cover the
# MAGIC primary data access patterns in enterprise AI:
# MAGIC
# MAGIC | Tool | Pattern | Data Source |
# MAGIC |------|---------|-------------|
# MAGIC | `query_accounts_tool` | Structured data retrieval | Delta tables (SQL) |
# MAGIC | `search_knowledge_base` | Unstructured retrieval | Vector Search (semantic) |
# MAGIC | `analyze_pipeline` | Data analysis | Delta tables (aggregation) |
# MAGIC
# MAGIC These tools will become the "hands" of our agent in Notebook 04.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 — Tool 1: Structured Data Retrieval (SQL)

# COMMAND ----------

def query_accounts_tool(industry: str = None, min_revenue: float = None,
                         account_tier: str = None, region: str = None,
                         limit: int = 10) -> str:
    """
    Query GTM accounts from Delta tables based on filters.

    Args:
        industry: Filter by industry (e.g., 'Technology', 'Financial Services')
        min_revenue: Minimum annual revenue in dollars
        account_tier: Filter by tier ('Enterprise', 'Mid-Market', 'SMB')
        region: Filter by region (e.g., 'North America', 'EMEA')
        limit: Maximum number of results to return (default 10)

    Returns:
        Formatted string with matching account details
    """
    conditions = []
    if industry:
        conditions.append(f"industry = '{industry}'")
    if min_revenue:
        conditions.append(f"annual_revenue >= {min_revenue}")
    if account_tier:
        conditions.append(f"account_tier = '{account_tier}'")
    if region:
        conditions.append(f"region = '{region}'")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT account_id, company_name, industry, employee_count,
               annual_revenue, region, country, account_tier
        FROM {catalog}.{schema}.gtm_accounts
        WHERE {where_clause}
        ORDER BY annual_revenue DESC
        LIMIT {limit}
    """

    try:
        results = spark.sql(query).collect()
        if not results:
            return "No accounts found matching the specified criteria."

        output_lines = [f"Found {len(results)} accounts:\n"]
        for row in results:
            output_lines.append(
                f"- {row['company_name']} | {row['industry']} | "
                f"Revenue: ${row['annual_revenue']:,.0f} | "
                f"Employees: {row['employee_count']:,} | "
                f"{row['region']}, {row['country']} | Tier: {row['account_tier']}"
            )
        return "\n".join(output_lines)
    except Exception as e:
        return f"Error querying accounts: {str(e)}"

# COMMAND ----------

# Test Tool 1
print("TEST: Enterprise Technology accounts with revenue > $500M")
print("=" * 60)
print(query_accounts_tool(industry="Technology", account_tier="Enterprise"))

# COMMAND ----------

print("\nTEST: All accounts in EMEA")
print("=" * 60)
print(query_accounts_tool(region="EMEA", limit=5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 — Tool 2: Unstructured Retrieval (Vector Search)

# COMMAND ----------

def search_knowledge_base(query: str, num_results: int = 3, category: str = None) -> str:
    """
    Search the GTM knowledge base using Vector Search for semantically relevant documents.

    Args:
        query: Natural language search query
        num_results: Number of results to return (default 3)
        category: Optional filter by document category

    Returns:
        Formatted string with relevant document excerpts
    """
    try:
        search_params = {
            "index_name": f"{catalog}.{schema}.gtm_knowledge_vs_index",
            "columns": ["doc_id", "title", "category", "subcategory", "content"],
            "query_text": query,
            "num_results": num_results
        }

        # Add category filter if specified
        if category:
            search_params["filters_json"] = f'{{"category": ["{category}"]}}'

        results = w.vector_search_indexes.query_index(**search_params)

        if not results.result.data_array:
            return "No relevant documents found for your query."

        output_lines = [f"Found {len(results.result.data_array)} relevant documents:\n"]
        for i, row in enumerate(results.result.data_array):
            doc_id, title, cat, subcat, content = row[0], row[1], row[2], row[3], row[4]
            # Truncate content for readability in tool output
            content_preview = str(content)[:500]
            output_lines.append(
                f"--- Document {i+1}: {title} ---\n"
                f"Category: {cat} / {subcat}\n"
                f"{content_preview}\n"
            )
        return "\n".join(output_lines)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# COMMAND ----------

# Test Tool 2 — semantic search
print("TEST: Searching for 'competitive positioning against Snowflake'")
print("=" * 60)
print(search_knowledge_base("competitive positioning against Snowflake"))

# COMMAND ----------

print("\nTEST: Searching for 'pricing and packaging'")
print("=" * 60)
print(search_knowledge_base("pricing and packaging"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 — Tool 3: Data Analysis (Pipeline Analytics)

# COMMAND ----------

def analyze_pipeline(stage: str = None, include_details: bool = False) -> str:
    """
    Analyze the sales pipeline and return summary insights.

    Args:
        stage: Optional filter by deal stage (e.g., 'Negotiation', 'Proposal', 'Discovery')
        include_details: If True, include individual deal details

    Returns:
        Formatted string with pipeline analysis including totals, averages, and breakdowns
    """
    try:
        # Stage-level summary
        stage_filter = f"WHERE stage = '{stage}'" if stage else ""

        summary_query = f"""
            SELECT
                stage,
                COUNT(*) as deal_count,
                ROUND(SUM(amount), 0) as total_amount,
                ROUND(AVG(amount), 0) as avg_deal_size,
                ROUND(AVG(probability), 1) as avg_probability,
                ROUND(SUM(amount * probability / 100), 0) as weighted_pipeline
            FROM {catalog}.{schema}.gtm_opportunities
            {stage_filter}
            GROUP BY stage
            ORDER BY avg_probability DESC
        """

        stage_results = spark.sql(summary_query).collect()

        if not stage_results:
            return "No pipeline data found for the specified criteria."

        # Overall totals
        totals_query = f"""
            SELECT
                COUNT(*) as total_deals,
                ROUND(SUM(amount), 0) as total_pipeline,
                ROUND(SUM(amount * probability / 100), 0) as weighted_pipeline,
                ROUND(AVG(amount), 0) as avg_deal_size
            FROM {catalog}.{schema}.gtm_opportunities
            {stage_filter}
        """
        totals = spark.sql(totals_query).first()

        output_lines = [
            "PIPELINE ANALYSIS",
            "=" * 50,
            f"Total Deals:       {totals['total_deals']}",
            f"Total Pipeline:    ${totals['total_pipeline']:,.0f}",
            f"Weighted Pipeline: ${totals['weighted_pipeline']:,.0f}",
            f"Avg Deal Size:     ${totals['avg_deal_size']:,.0f}",
            "",
            "BREAKDOWN BY STAGE:",
            "-" * 50
        ]

        for row in stage_results:
            output_lines.append(
                f"  {row['stage']:<20} | "
                f"{row['deal_count']:>3} deals | "
                f"${row['total_amount']:>12,.0f} | "
                f"Avg: ${row['avg_deal_size']:>10,.0f} | "
                f"Prob: {row['avg_probability']:>5.1f}% | "
                f"Weighted: ${row['weighted_pipeline']:>12,.0f}"
            )

        # If details requested and stage filter applied, show individual deals
        if include_details and stage:
            detail_query = f"""
                SELECT o.opportunity_id, a.company_name, o.amount, o.probability, o.close_date
                FROM {catalog}.{schema}.gtm_opportunities o
                JOIN {catalog}.{schema}.gtm_accounts a ON o.account_id = a.account_id
                WHERE o.stage = '{stage}'
                ORDER BY o.amount DESC
                LIMIT 10
            """
            details = spark.sql(detail_query).collect()

            output_lines.append(f"\nTOP DEALS IN '{stage}' STAGE:")
            output_lines.append("-" * 50)
            for row in details:
                output_lines.append(
                    f"  {row['company_name']:<30} | "
                    f"${row['amount']:>12,.0f} | "
                    f"Prob: {row['probability']}% | "
                    f"Close: {row['close_date']}"
                )

        return "\n".join(output_lines)
    except Exception as e:
        return f"Error analyzing pipeline: {str(e)}"

# COMMAND ----------

# Test Tool 3 — full pipeline overview
print("TEST: Full pipeline analysis")
print(analyze_pipeline())

# COMMAND ----------

# Test Tool 3 — specific stage with details
print("\nTEST: Negotiation stage with deal details")
print(analyze_pipeline(stage="Negotiation", include_details=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 — Quick End-to-End RAG Test
# MAGIC
# MAGIC Let's combine our tools with the LLM to see a complete RAG flow before we build the full agent.

# COMMAND ----------

# A simple RAG flow: retrieve context, then generate an answer
user_question = "What's the best approach for selling to financial services companies?"

# Step 1: Retrieve relevant documents
context = search_knowledge_base(user_question, num_results=3)

# Step 2: Generate answer with context
response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[
        {
            "role": "system",
            "content": """You are a GTM strategy assistant. Answer questions using ONLY the
provided context documents. If the context doesn't contain enough information, say so.
Always cite which document(s) you're drawing from."""
        },
        {
            "role": "user",
            "content": f"""Context Documents:
{context}

Question: {user_question}

Provide a detailed answer based on the context above."""
        }
    ],
    max_tokens=500,
    temperature=0.3
)

print(f"Question: {user_question}\n")
print("=" * 60)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 5: Agent Bricks & Model Context Protocol (MCP)
# MAGIC
# MAGIC ## 5.1 — Agent Bricks: Pre-Built Agent Components
# MAGIC
# MAGIC **Agent Bricks** are pre-built, production-ready agent capabilities that Databricks provides out of the box.
# MAGIC Instead of building every agent capability from scratch, you can compose agents from these building blocks:
# MAGIC
# MAGIC ```
# MAGIC  ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC  │                    Your Custom Agent                                │
# MAGIC  │                                                                     │
# MAGIC  │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐         │
# MAGIC  │   │  Unity Catalog│  │    Genie      │  │  Vector Search│         │
# MAGIC  │   │  Agent Brick  │  │  Agent Brick  │  │  Agent Brick  │         │
# MAGIC  │   │               │  │               │  │               │         │
# MAGIC  │   │  "Query any   │  │  "Natural     │  │  "Semantic    │         │
# MAGIC  │   │   table in    │  │   language     │  │   search over │         │
# MAGIC  │   │   your        │  │   to SQL on    │  │   documents"  │         │
# MAGIC  │   │   catalog"    │  │   any table"   │  │               │         │
# MAGIC  │   └───────────────┘  └───────────────┘  └───────────────┘         │
# MAGIC  │                                                                     │
# MAGIC  │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐         │
# MAGIC  │   │  Code Exec    │  │   Custom      │  │   MCP         │         │
# MAGIC  │   │  Agent Brick  │  │   Function    │  │   Connector   │         │
# MAGIC  │   │               │  │   Agent Brick │  │               │         │
# MAGIC  │   │  "Run Python  │  │               │  │  "Connect to  │         │
# MAGIC  │   │   in sandbox" │  │  "Your own    │  │   external    │         │
# MAGIC  │   │               │  │   tools"      │  │   services"   │         │
# MAGIC  │   └───────────────┘  └───────────────┘  └───────────────┘         │
# MAGIC  │                                                                     │
# MAGIC  └─────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Available Agent Bricks
# MAGIC
# MAGIC | Agent Brick | What It Does | Use Case |
# MAGIC |-------------|--------------|----------|
# MAGIC | **Unity Catalog** | Query structured data from any table | "What's our revenue by region?" |
# MAGIC | **Genie** | Natural language to SQL (Text-to-SQL) | Complex analytics without writing SQL |
# MAGIC | **Vector Search** | Semantic retrieval from document indexes | RAG over knowledge bases |
# MAGIC | **Code Execution** | Run Python code in a secure sandbox | Data analysis, chart generation |
# MAGIC | **Custom Function** | Wrap any Python function as a tool | API calls, business logic |
# MAGIC
# MAGIC ### Why Agent Bricks Matter
# MAGIC 1. **Faster development** — Use tested, production-grade components
# MAGIC 2. **Governed by default** — All data access goes through Unity Catalog permissions
# MAGIC 3. **Observable** — Every tool call is traced via MLflow
# MAGIC 4. **Composable** — Mix and match to build the agent your use case requires
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 — Model Context Protocol (MCP)
# MAGIC
# MAGIC **MCP** is an open standard (created by Anthropic, adopted across the industry) that defines how
# MAGIC AI agents connect to external tools and data sources. Think of it as **USB-C for AI agents** — a
# MAGIC universal protocol that lets any agent talk to any tool.
# MAGIC
# MAGIC ```
# MAGIC  ┌──────────────────────────────────────────────────────────────────────────┐
# MAGIC  │                   Model Context Protocol (MCP)                           │
# MAGIC  │                                                                          │
# MAGIC  │   ┌─────────────┐         ┌──────────────┐        ┌─────────────────┐   │
# MAGIC  │   │   Agent     │  MCP    │    MCP       │  MCP   │   Tool/Data     │   │
# MAGIC  │   │   (Client)  │◄──────►│   Server     │◄──────►│   Source        │   │
# MAGIC  │   │             │         │              │        │                 │   │
# MAGIC  │   │  "I need    │  JSON   │  "Here are   │        │  - Databases    │   │
# MAGIC  │   │   to search │  RPC    │   the tools  │        │  - APIs         │   │
# MAGIC  │   │   for data" │         │   available" │        │  - File systems │   │
# MAGIC  │   │             │         │              │        │  - SaaS apps    │   │
# MAGIC  │   └─────────────┘         └──────────────┘        └─────────────────┘   │
# MAGIC  │                                                                          │
# MAGIC  │   Benefits:                                                              │
# MAGIC  │   - Standardized tool discovery (agents learn what tools are available)  │
# MAGIC  │   - Secure (auth, permissions, rate limiting at the server layer)        │
# MAGIC  │   - Portable (switch agents without rewriting tool integrations)         │
# MAGIC  │   - Observable (every tool call is logged and traceable)                 │
# MAGIC  └──────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### MCP on Databricks
# MAGIC
# MAGIC Databricks integrates MCP to securely connect agents to enterprise data:
# MAGIC
# MAGIC - **Unity Catalog as MCP Server**: Your catalog of tables, functions, and models becomes
# MAGIC   discoverable via MCP. An agent can browse available tools and data without hardcoding.
# MAGIC - **Secure by design**: MCP connections inherit Unity Catalog governance — row/column level
# MAGIC   security, audit logging, and access controls.
# MAGIC - **External MCP servers**: Connect to Slack, Jira, Salesforce, GitHub, etc. via community
# MAGIC   MCP servers — with Databricks handling auth and rate limiting.
# MAGIC
# MAGIC ### Conceptual Example: Agent Using MCP Tools
# MAGIC
# MAGIC ```python
# MAGIC # Conceptual — MCP integration with Databricks agents
# MAGIC from databricks.agents import Agent, MCPToolkit
# MAGIC
# MAGIC # Create a toolkit that discovers tools via MCP
# MAGIC toolkit = MCPToolkit(
# MAGIC     servers=[
# MAGIC         "unity-catalog://ankit_yadav.servicenow_training",  # Delta tables
# MAGIC         "mcp://slack-server",                                # Slack integration
# MAGIC         "mcp://jira-server",                                 # Jira integration
# MAGIC     ]
# MAGIC )
# MAGIC
# MAGIC # The agent automatically discovers available tools from all MCP servers
# MAGIC agent = Agent(
# MAGIC     model="databricks-meta-llama-3-3-70b-instruct",
# MAGIC     tools=toolkit.get_tools(),  # Auto-discovered!
# MAGIC     instructions="You are a GTM assistant..."
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC > **Key Takeaway:** MCP eliminates the need to manually define tool schemas. As your enterprise
# MAGIC > data landscape evolves, agents automatically discover new capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary: What We Built in Module 3
# MAGIC
# MAGIC | Component | Status | What It Does |
# MAGIC |-----------|--------|--------------|
# MAGIC | Foundation Model API | Tested | Call LLMs with OpenAI-compatible SDK |
# MAGIC | Prompt Engineering | 4 techniques | Zero-shot, few-shot, CoT, system prompts |
# MAGIC | Vector Search Index | Created | Semantic search over 50+ GTM documents |
# MAGIC | SQL Query Tool | Built | Query accounts by industry, revenue, tier |
# MAGIC | Knowledge Search Tool | Built | Semantic retrieval from knowledge base |
# MAGIC | Pipeline Analysis Tool | Built | Analyze deals by stage with metrics |
# MAGIC
# MAGIC ### Next Up: Module 4
# MAGIC In the next notebook, we'll wire these tools into a **fully functional agent** that can:
# MAGIC - Autonomously decide which tools to call
# MAGIC - Handle multi-turn conversations
# MAGIC - Be evaluated for quality and deployed to production
# MAGIC
# MAGIC ---
# MAGIC
