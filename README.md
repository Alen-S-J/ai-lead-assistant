# AI Lead Follow-Up Assistant Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://claude.ai/chat/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://claude.ai/chat/CONTRIBUTING.md)

> A production-ready, multi-model AI system for intelligent lead follow-up using RAG, Fine-tuned LLMs, and Hybrid approaches.

---

## ⚠️ IMPORTANT DISCLAIMER

**This is a conceptual/demonstration project showcasing AI prompt engineering architecture and best practices.**

### Key Points:

* ❌  **NO ACTUAL DATASETS INCLUDED** : This repository does not contain any training data, fine-tuned models, or company knowledge bases
* ❌  **NO PRE-TRAINED MODELS** : Fine-tuned model adapters are not provided and must be trained by you
* ❌  **NO VECTOR DATABASE** : Knowledge base documents for RAG are not included
* ❌  **TEMPLATE ONLY** : This serves as a blueprint/framework for building such systems

### What You Get:

✅  **Architecture Design** : Complete system design and prompt engineering patterns

✅  **API Structure** : Production-ready FastAPI endpoints and schemas

✅  **Prompt Templates** : Master prompts for RAG, Fine-tuned, and Hybrid approaches

✅  **Integration Patterns** : Examples for CRM, email, and workflow integration

✅  **Best Practices** : Anti-hallucination, grounding, and validation techniques

### What You Need to Provide:

📋  **Your Own Data** :

* Historical lead conversations for fine-tuning (minimum 1000+ examples)
* Company knowledge base documents for RAG (product docs, pricing, case studies)
* Lead profiles and interaction data

🔧  **Your Own Training** :

* Fine-tune the base LLM on your specific domain/industry
* Create embeddings and populate vector database with your documents
* Configure prompts for your specific products and use cases

💾  **Your Own Infrastructure** :

* LLM access (Hugging Face, OpenAI, Anthropic, or self-hosted)
* Vector database setup (Chroma, Pinecone, Weaviate)
* Database and caching infrastructure

---

## 📋 Table of Contents

* [Overview](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#overview)
* [Architecture](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#architecture)
* [Features](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#features)
* [Tech Stack](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#tech-stack)
* [Installation](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#installation)
* [Quick Start](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#quick-start)
* [API Documentation](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#api-documentation)
* [Model Comparison](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#model-comparison)
* [Prompt Engineering](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#prompt-engineering)
* [Configuration](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#configuration)
* [Development](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#development)
* [Deployment](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#deployment)
* [Monitoring](https://claude.ai/chat/befba708-c666-4086-98d9-81f927409014#monitoring)

---

## 🎯 Overview

This is a **backend-focused platform** that provides REST API endpoints for three powerful AI approaches to generate personalized, context-aware lead follow-up messages:

1. **RAG (Retrieval-Augmented Generation)** - Real-time knowledge retrieval from company documentation
2. **Fine-Tuned LLM** - Domain-specific model trained on 10,000+ successful sales conversations
3. **Hybrid RAG + Fine-Tuned** - Fusion approach combining factual accuracy with conversational excellence

**Note:** This project focuses exclusively on backend AI/ML infrastructure. No frontend implementation is included.

### Key Capabilities

* ✅ RESTful API endpoints for all three AI models
* ✅ Personalized responses based on lead profiles and interaction history
* ✅ Anti-hallucination protocols ensuring factual accuracy
* ✅ Dynamic strategy selection based on lead scoring
* ✅ Multi-tier caching for sub-second response times
* ✅ Industry-specific tone and vocabulary adaptation
* ✅ Edge case handling (rejections, objections, cold leads)
* ✅ Comprehensive monitoring and analytics
* ✅ OpenAPI/Swagger documentation for easy integration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application Layer                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   RAG Model  │  │ Fine-Tuned   │  │  Hybrid RAG+FT      │ │
│  │              │  │   Model      │  │      Model           │ │
│  │ ChromaDB     │  │ LoRA Adapter │  │ Fusion Engine       │ │
│  │ Embeddings   │  │ Llama-2-7B   │  │ 60% RAG/40% FT      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     Supporting Services                          │
├─────────────────────────────────────────────────────────────────┤
│  Redis Cache    │  PostgreSQL    │  Celery Queue  │ Prometheus │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Request → FastAPI → Model Selection → Processing → Response
                  ↓
            Redis Cache (Check)
                  ↓
        Model-Specific Pipeline:
          - RAG: Vector Search → LLM
          - Fine-tuned: Direct LLM
          - Hybrid: Vector Search + Fine-tuned LLM
                  ↓
        Post-Processing & Validation
                  ↓
        Cache → Database Log → Return
```

---

## ✨ Features

### Core Functionality

* **Multi-Model Architecture** : Choose between RAG, Fine-tuned, or Hybrid based on use case
* **Intelligent Lead Scoring** : Automatic strategy selection (high/medium/low intent)
* **Context-Aware Responses** : Leverages interaction history, industry, company size
* **Grounding Verification** : Ensures responses are factually accurate
* **Personalization Engine** : Dynamic tone, vocabulary, and style adaptation

### Advanced Features

* **Edge Case Handling** : Graceful responses to rejections and objections
* **Hallucination Prevention** : Multi-layer verification and grounding checks
* **Performance Optimization** : Redis caching, async processing, connection pooling
* **A/B Testing Ready** : Compare model outputs for continuous improvement
* **Compliance** : Configurable content filtering and approval workflows

---

## 🛠️ Tech Stack

### Backend Framework

* **FastAPI** (0.104+) - High-performance async API framework
* **Pydantic** (2.0+) - Data validation and settings management
* **SQLAlchemy** (2.0+) - Database ORM

### AI/ML Stack

* **LangChain** (0.1.0+) - LLM orchestration framework
* **Hugging Face Transformers** (4.35+) - Model inference
* **PEFT/LoRA** (0.7+) - Efficient fine-tuning
* **Sentence Transformers** - Text embeddings

### Vector Database

* **ChromaDB** (0.4+) - Primary vector store
* **Alternatives** : Pinecone, Weaviate, Qdrant

### Infrastructure

* **Redis** (7.0+) - Caching and session management
* **PostgreSQL** (14+) - Persistent data storage
* **Celery** (5.3+) - Distributed task queue
* **RabbitMQ** - Message broker

### Monitoring & DevOps

* **Prometheus** - Metrics collection
* **Grafana** - Visualization dashboards
* **Docker** - Containerization
* **Uvicorn** - ASGI server

---

## 📦 Installation

### Prerequisites

```bash
# Python 3.9 or higher
python --version

# PostgreSQL
psql --version

# Redis
redis-server --version

# Optional: CUDA for GPU acceleration
nvidia-smi
```

### Step 1: Clone Repository

```bash
git clone https://github.com/Alen-S-J/ai-lead-assistant.git
cd ai-lead-assistant
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### Step 4: Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit with your configuration
nano .env
```

Required environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/lead_assistant

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Model Configuration
HF_TOKEN=your_huggingface_token
MODEL_CACHE_DIR=/path/to/model/cache

# Vector DB
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Fine-tuned Model
FINETUNED_MODEL_PATH=./finetuned_models/lead_followup_lora

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
```

### Step 5: Database Setup

```bash
# Create database
createdb lead_assistant

# Run migrations
alembic upgrade head

# Seed initial data (optional)
python scripts/seed_data.py
```

### Step 6: Download Models

> **⚠️ IMPORTANT** : This project does NOT include pre-trained or fine-tuned models. You must either:
>
> 1. Use base models from Hugging Face (no fine-tuning)
> 2. Train your own models with your data

```bash
# Download base models (NO fine-tuning)
python scripts/download_models.py

# Download embeddings
python scripts/download_embeddings.py
```

 **Note** : The `FINETUNED_MODEL_PATH` in `.env` will not work without your own training data.

### Step 7: Initialize Vector Database

> **⚠️ IMPORTANT** : You must provide your own company documents for RAG to work.

```bash
# Ingest YOUR company documents
python scripts/ingest_documents.py --source ./data/knowledge_base/

# Example documents needed:
# - Product documentation (PDFs, Markdown, HTML)
# - Pricing sheets
# - Case studies
# - FAQ documents
# - Technical specifications

# Verify ingestion
python scripts/verify_vectordb.py
```

**Without your own documents, the RAG model will return fallback responses only.**

---

## 🚀 Quick Start

### Start the Application

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Access the API

* **Interactive API Docs** : http://localhost:8000/docs
* **Alternative API Docs** : http://localhost:8000/redoc
* **Health Check** : http://localhost:8000/health
* **Metrics Endpoint** : http://localhost:9090/metrics

### Testing the API

```bash
# Using curl
curl -X POST "http://localhost:8000/rag/api/v1/rag/generate-followup" \
  -H "Content-Type: application/json" \
  -d '{
    "lead_context": {
      "lead_name": "Ramesh",
      "product_name": "AI Billing Software",
      "interaction_history": "Asked about pricing",
      "lead_stage": "consideration",
      "engagement_score": 65,
      "days_since_contact": 7
    }
  }'

# Using Python requests
import requests

response = requests.post(
    "http://localhost:8000/rag/api/v1/rag/generate-followup",
    json={
        "lead_context": {
            "lead_name": "Ramesh",
            "product_name": "AI Billing Software",
            "interaction_history": "Asked about pricing",
            "lead_stage": "consideration",
            "engagement_score": 65,
            "days_since_contact": 7
        }
    }
)
print(response.json())
```

---

## 📚 API Documentation

### 1. RAG Model Endpoint

**Generate Follow-up with RAG**

```bash
POST /rag/api/v1/rag/generate-followup
```

**Request Body:**

```json
{
  "lead_context": {
    "lead_name": "Ramesh",
    "product_name": "AI Billing Software",
    "interaction_history": "Asked about pricing, stopped responding",
    "lead_stage": "consideration",
    "engagement_score": 65,
    "days_since_contact": 7,
    "tone_required": "friendly_helpful",
    "company_name": "TechCorp",
    "product_category": "SaaS"
  },
  "rag_config": {
    "top_k": 5,
    "retrieval_threshold": 0.7,
    "source_types": ["pricing", "features", "case_studies"],
    "use_hybrid_search": true
  },
  "max_tokens": 150
}
```

**Response:**

```json
{
  "message": "Hi Ramesh, I wanted to follow up on the AI Billing Software pricing we discussed. Based on your requirements, our mid-tier plan at $299/month includes automated invoice processing and real-time analytics. Companies like yours typically see 40% reduction in billing errors within the first quarter. Would it help to see a customized pricing breakdown for your team size?",
  "retrieval_sources": [
    "pricing_2024.pdf",
    "case_study_manufacturing.pdf"
  ],
  "confidence_score": 0.82,
  "model_used": "rag_llm",
  "cached": false
}
```

---

### 2. Fine-Tuned Model Endpoint

**Generate Follow-up with Fine-Tuned LLM**

```bash
POST /finetuned/api/v1/finetuned/generate-followup
```

**Request Body:**

```json
{
  "lead_profile": {
    "lead_name": "Ramesh",
    "product_name": "AI Billing Software",
    "interaction_history": "Asked about pricing, stopped responding",
    "lead_score": 72,
    "lead_industry": "manufacturing",
    "company_size": "midmarket",
    "engagement_pattern": "periodic",
    "objections_raised": ["price"],
    "days_since_contact": 7,
    "geographic_region": "US"
  },
  "company_name": "TechCorp",
  "product_category": "SaaS",
  "temperature": 0.7,
  "max_tokens": 150
}
```

**Response:**

```json
{
  "message": "Hi Ramesh, I know pricing is a key consideration. Manufacturing companies like yours often see the value when we break down the ROI. Our billing automation typically saves 15-20 hours per week in manual processing. Would you be open to a quick call where we can discuss flexible payment options that fit your budget cycle?",
  "strategy_used": "high_intent",
  "confidence": 0.85,
  "model_version": "lora_v2.1",
  "personalization_score": 0.78
}
```

---

### 3. Hybrid Model Endpoint

**Generate Follow-up with Hybrid Approach**

```bash
POST /hybrid/api/v1/hybrid/generate-followup
```

**Request Body:**

```json
{
  "lead_context": {
    "lead_name": "Ramesh",
    "product_name": "AI Billing Software",
    "interaction_history": "Asked about pricing, stopped responding",
    "lead_score": 72,
    "engagement_score": 68,
    "lead_industry": "manufacturing",
    "company_size": "midmarket",
    "engagement_pattern": "periodic",
    "objections_raised": ["price"],
    "days_since_contact": 7,
    "company_name": "TechCorp",
    "product_category": "SaaS",
    "geographic_region": "US"
  },
  "rag_config": {
    "top_k": 5,
    "retrieval_threshold": 0.65,
    "use_reranking": true,
    "source_weights": {
      "pricing": 1.2,
      "case_studies": 1.0,
      "features": 0.9
    }
  },
  "generation_config": {
    "temperature": 0.7,
    "max_tokens": 150,
    "fusion_weight_rag": 0.6,
    "require_grounding": true
  }
}
```

**Response:**

```json
{
  "message": "Hi Ramesh, following up on the AI Billing Software pricing discussion. Our mid-tier plan at $299/month has helped manufacturing firms like Anderson Corp reduce billing cycles by 45%. The ROI typically pays for itself within 3 months through time savings alone. Would you like to see a customized cost-benefit analysis for your operation?",
  "rag_sources": [
    "pricing_2024.pdf",
    "case_study_anderson_manufacturing.pdf",
    "roi_calculator.pdf"
  ],
  "rag_confidence": 0.82,
  "strategy_used": "high_intent",
  "strategy_confidence": 0.785,
  "hybrid_confidence": 0.803,
  "grounding_verified": true,
  "personalization_score": 0.85,
  "generation_metadata": {
    "num_docs_used": 3,
    "fusion_weight": 0.6,
    "timestamp": "2024-01-23T10:30:00Z"
  }
}
```

---

### 4. Health Check Endpoint

```bash
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "models": {
    "rag": "active",
    "finetuned": "active",
    "hybrid": "active"
  },
  "services": {
    "database": "connected",
    "redis": "connected",
    "vector_db": "active"
  },
  "version": "1.0.0"
}
```

---

## 🔍 Model Comparison

| Feature                          | RAG Model                   | Fine-Tuned Model             | Hybrid Model        |
| -------------------------------- | --------------------------- | ---------------------------- | ------------------- |
| **Factual Accuracy**       | ⭐⭐⭐⭐⭐                  | ⭐⭐⭐                       | ⭐⭐⭐⭐⭐          |
| **Conversational Quality** | ⭐⭐⭐                      | ⭐⭐⭐⭐⭐                   | ⭐⭐⭐⭐⭐          |
| **Real-time Knowledge**    | ✅ Yes                      | ❌ No                        | ✅ Yes              |
| **Domain Adaptation**      | ⭐⭐⭐                      | ⭐⭐⭐⭐⭐                   | ⭐⭐⭐⭐⭐          |
| **Response Time**          | ~800ms                      | ~500ms                       | ~1200ms             |
| **Setup Complexity**       | Medium                      | High                         | High                |
| **Data Required**          | ⚠️ Knowledge Base Docs    | ⚠️ 1000+ Training Examples | ⚠️ Both           |
| **Best For**               | New products, changing info | Established patterns         | Comprehensive needs |

> **⚠️ Note** : All models require you to provide your own data. Response times are estimates and depend on your infrastructure.

### When to Use Each Model

**RAG Model** 🔍

* New product launches with limited historical data
* Frequently updated pricing or features
* Need for citing specific documentation
* Compliance-heavy industries (healthcare, finance)
* **Requirements** : Your company knowledge base (PDFs, docs, case studies)
* **Integration:** Consume via REST API for real-time knowledge retrieval

**Fine-Tuned Model** 🎯

* Established products with rich conversation history
* Specific industry verticals (fintech, SaaS, retail)
* High-volume, consistent messaging needs
* Lower latency requirements
* **Requirements** : 1000+ historical successful lead conversations
* **Integration:** REST API for fast, pattern-based responses

**Hybrid Model** 🚀

* Enterprise clients requiring both accuracy and engagement
* Complex products needing detailed explanations
* High-value leads (>$50k ACV)
* A/B testing against other models
* **Requirements** : Both knowledge base AND training data
* **Integration:** REST API combining both approaches

### Integration Examples

**Integrate with Your CRM:**

```python
# Example: Salesforce Integration
from salesforce import Salesforce
import requests

def generate_and_send_followup(lead_id):
    # Get lead from Salesforce
    sf = Salesforce(...)
    lead = sf.Lead.get(lead_id)
  
    # Call AI API
    response = requests.post(
        "http://your-api.com/hybrid/api/v1/hybrid/generate-followup",
        json={
            "lead_context": {
                "lead_name": lead["Name"],
                "product_name": lead["Product__c"],
                "interaction_history": lead["Last_Interaction__c"],
                "lead_score": lead["Score__c"]
            }
        }
    )
  
    message = response.json()["message"]
  
    # Send email via your email service
    send_email(lead["Email"], message)
```

**Integrate with Your Email Service:**

```python
# Example: SendGrid Integration
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import requests

def send_ai_followup(to_email, lead_data):
    # Generate message
    response = requests.post(
        "http://your-api.com/finetuned/api/v1/finetuned/generate-followup",
        json={"lead_profile": lead_data}
    )
  
    ai_message = response.json()["message"]
  
    # Send via SendGrid
    message = Mail(
        from_email='sales@yourcompany.com',
        to_emails=to_email,
        subject='Following up on your interest',
        html_content=ai_message
    )
  
    sg = SendGridAPIClient(api_key='YOUR_API_KEY')
    sg.send(message)
```

---

## 🧠 Prompt Engineering

### Core Principles

All prompts in this system follow these design principles:

1. **Clear Role Definition** : Explicit AI persona and responsibilities
2. **Structured Context** : Lead profile, interaction history, goals
3. **Tone Specification** : Friendly, professional, industry-appropriate
4. **Hard Constraints** : Word limits, format requirements, forbidden patterns
5. **Anti-Hallucination** : Grounding checks, verification protocols
6. **Output Format** : Explicit instructions on what to generate

### Prompt Template Anatomy

```python
PROMPT_STRUCTURE = """
[1. ROLE & IDENTITY]
Define who the AI is and what it represents

[2. CONTEXT]
{retrieved_documents}  # For RAG
{lead_profile}         # Lead information
{interaction_history}  # Previous conversations

[3. INSTRUCTIONS]
Step-by-step what to do

[4. CONSTRAINTS]
- Word limits
- Tone requirements
- Forbidden patterns

[5. ANTI-HALLUCINATION]
Verification checks and fallbacks

[6. OUTPUT FORMAT]
Exact format specification

Generate now:
"""
```

### Example: Task 1 Solution

 **Objective** : Re-engage Ramesh who asked about pricing but stopped responding

```python
PRODUCTION_PROMPT = """You are a professional Lead Follow-Up Assistant for TechCorp's AI Billing Software.

LEAD CONTEXT:
Name: Ramesh
Product: AI Billing Software
Previous Interaction: Asked about pricing, stopped responding
Goal: Re-engage naturally with value

YOUR ROLE:
Helpful sales consultant who respects the lead's time and provides genuine value.

TONE & STYLE:
- Friendly and conversational (not robotic)
- Professional but warm
- Empathetic to evaluation process

REQUIREMENTS:
1. Acknowledge pricing inquiry naturally
2. Provide specific value point about AI Billing Software
3. Ask ONE relevant follow-up question
4. Keep under 120 words
5. Use "Ramesh" once, naturally

CONSTRAINTS:
❌ NO pushy sales language
❌ NO multiple questions
❌ NO generic "just checking in"
✅ MUST provide tangible value

Generate follow-up email body:
"""
```

### Hallucination Prevention Techniques

```python
# Layer 1: Explicit Grounding
"Base all claims on: {retrieved_documents}"
"If uncertain, say: 'Let me connect you with our specialist'"

# Layer 2: Verification Requirements
"Before mentioning pricing, verify in: {pricing_docs}"
"Only reference approved case studies: {case_study_list}"

# Layer 3: Conservative Defaults
"When in doubt, offer human handoff"
"Prefer vague-but-true over specific-but-uncertain"

# Layer 4: Post-Processing Validation
verify_grounding(message, retrieved_docs)
check_for_ungrounded_claims(message)
```

---

## ⚙️ Configuration

### Model Configuration

**config/models.yaml**

```yaml
rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  llm_model: "meta-llama/Llama-2-7b-chat-hf"
  vector_db:
    type: "chroma"
    persist_directory: "./chroma_db"
    collection_name: "lead_knowledge_base"
  retrieval:
    top_k: 5
    threshold: 0.7
    use_hybrid_search: true

finetuned:
  base_model: "meta-llama/Llama-2-7b-chat-hf"
  adapter_path: "./finetuned_models/lead_followup_lora"
  adapter_type: "lora"
  temperature: 0.7
  max_new_tokens: 200

hybrid:
  fusion_weight_rag: 0.6
  fusion_weight_finetuned: 0.4
  require_grounding: true
  min_confidence_threshold: 0.65
```

### Caching Configuration

**config/cache.yaml**

```yaml
redis:
  host: "localhost"
  port: 6379
  db: 0
  ttl:
    response_cache: 1800  # 30 minutes
    retrieval_cache: 3600  # 1 hour
    model_cache: 86400  # 24 hours

cache_keys:
  rag: "rag:{lead_name}:{product_name}"
  finetuned: "ft:{lead_name}:{lead_score}"
  hybrid: "hybrid:{lead_name}:{product_name}:{lead_score}"
```

### Lead Scoring Rules

**config/scoring.yaml**

```yaml
lead_score_calculation:
  engagement_weight: 0.4
  interaction_frequency_weight: 0.3
  recency_weight: 0.2
  company_size_weight: 0.1

strategy_selection:
  high_intent:
    score_range: [70, 100]
    approach: "direct_value"
    cta_type: "specific"
  medium_intent:
    score_range: [40, 69]
    approach: "educational"
    cta_type: "soft"
  low_intent:
    score_range: [0, 39]
    approach: "value_first"
    cta_type: "stay_relevant"
```

---

## 👩‍💻 Development

### Project Structure

```
ai-lead-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── database.py             # Database models
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rag_model.py        # RAG implementation
│   │   ├── finetuned_model.py  # Fine-tuned implementation
│   │   └── hybrid_model.py     # Hybrid implementation
│   ├── prompts/
│   │   ├── rag_prompts.py
│   │   ├── finetuned_prompts.py
│   │   └── hybrid_prompts.py
│   └── utils/
│       ├── cache.py
│       ├── metrics.py
│       └── validators.py
├── config/
│   ├── models.yaml
│   ├── cache.yaml
│   └── scoring.yaml
├── data/
│   ├── knowledge_base/         # Documents for RAG
│   └── training_data/          # Fine-tuning datasets
├── finetuned_models/
│   └── lead_followup_lora/     # LoRA adapters
├── chroma_db/                  # Vector database
├── scripts/
│   ├── download_models.py
│   ├── ingest_documents.py
│   ├── train_finetuned.py
│   └── seed_data.py
├── tests/
│   ├── test_rag.py
│   ├── test_finetuned.py
│   └── test_hybrid.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_rag.py

# Run with coverage
pytest --cov=app tests/

# Generate HTML coverage report
pytest --cov=app --cov-report=html tests/
```

### Code Quality

```bash
# Format code
black app/

# Lint
flake8 app/
pylint app/

# Type checking
mypy app/

# Security scanning
bandit -r app/
```

### Fine-Tuning Your Own Model

> **⚠️ REQUIRED** : The fine-tuned model will NOT work without your own training data.

**Data Requirements:**

* Minimum 1,000 high-quality examples (5,000+ recommended)
* Historical successful lead conversations
* Must include: lead context, interaction history, and successful follow-up messages
* Balanced across industries, lead scores, and engagement patterns

```bash
# 1. Prepare YOUR training data
python scripts/prepare_training_data.py \
  --input data/historical_conversations.csv \
  --output data/training_data/

# Expected CSV format:
# lead_name, product, industry, lead_score, interaction_history, successful_response

# 2. Train LoRA adapter ON YOUR DATA
python scripts/train_finetuned.py \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --train_data data/training_data/ \
  --output_dir finetuned_models/lead_followup_lora \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4

# 3. Evaluate model
python scripts/evaluate_model.py \
  --model finetuned_models/lead_followup_lora \
  --test_data data/test_set.jsonl
```

 **Timeline** : Training typically takes 4-8 hours on a single GPU (A100/V100) depending on dataset size.

 **Cost Estimate** : ~$50-200 if using cloud GPU services (AWS, Lambda Labs, RunPod)

---

## 🚢 Deployment

> **⚠️ DEPLOYMENT NOTICE** : Before deploying, ensure you have:
>
> * ✅ Your own training data and fine-tuned models
> * ✅ Your own knowledge base documents ingested
> * ✅ Configured all environment variables with YOUR credentials
> * ✅ Access to LLM inference (Hugging Face, OpenAI, or self-hosted)

### Docker Deployment

**docker-compose.yml**

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/lead_assistant
      - REDIS_HOST=redis
      # ⚠️ ADD YOUR OWN API KEYS
      - HF_TOKEN=${YOUR_HF_TOKEN}
      - OPENAI_API_KEY=${YOUR_OPENAI_KEY}  # If using OpenAI
    depends_on:
      - db
      - redis
    volumes:
      # ⚠️ Mount YOUR trained models
      - ./finetuned_models:/app/finetuned_models
      # ⚠️ Mount YOUR vector database
      - ./chroma_db:/app/chroma_db

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=lead_assistant
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  postgres_data:
```

### Kubernetes Deployment

> **⚠️ KUBERNETES NOTICE** : Update the deployment with your container registry and secrets.

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-lead-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-lead-assistant
  template:
    metadata:
      labels:
        app: ai-lead-assistant
    spec:
      containers:
      - name: api
        image: your-registry/ai-lead-assistant:latest  # ⚠️ YOUR REGISTRY
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: url
        # ⚠️ ADD YOUR SECRETS
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: ai-secrets
              key: hf-token
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        # ⚠️ Mount YOUR trained models
        - name: models-volume
          mountPath: /app/finetuned_models
        - name: vectordb-volume
          mountPath: /app/chroma_db
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: vectordb-volume
        persistentVolumeClaim:
          claimName: vectordb-pvc
```

### Production Checklist

**Data & Models:**

* [ ] ✅ Your training data prepared and validated
* [ ] ✅ Fine-tuned models trained and tested
* [ ] ✅ Knowledge base documents ingested into vector DB
* [ ] ✅ Embeddings generated for all documents

**Infrastructure:**

* [ ] Set up environment variables securely
* [ ] Configure database connection pooling
* [ ] Enable Redis clustering for high availability
* [ ] Set up load balancer (nginx/HAProxy)
* [ ] Configure SSL/TLS certificates
* [ ] Set up log aggregation (ELK stack)
* [ ] Configure monitoring alerts
* [ ] Implement rate limiting
* [ ] Set up backup strategy
* [ ] Document API authentication
* [ ] Configure CORS properly
* [ ] Set up CI/CD pipeline

**Testing:**

* [ ] Test with YOUR actual lead data
* [ ] Validate prompt outputs for YOUR use cases
* [ ] Benchmark response times with YOUR models
* [ ] Test fallback mechanisms

---

## 📊 Monitoring

### Metrics Collected

**Application Metrics**

* Request count by endpoint
* Response times (p50, p95, p99)
* Error rates by model
* Cache hit rates
* Model inference times

**Business Metrics**

* Leads processed per hour
* Average lead score distribution
* Strategy selection breakdown
* Personalization scores
* Conversion tracking (if integrated with CRM)

### Grafana Dashboard

Import the included dashboard:

```bash
# Import dashboard
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/dashboard.json
```

### Alerting Rules

**prometheus/alerts.yml**

```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 2
        for: 5m
        annotations:
          summary: "95th percentile response time > 2s"

      - alert: LowCacheHitRate
        expr: cache_hit_rate < 0.6
        for: 10m
        annotations:
          summary: "Cache hit rate below 60%"
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://claude.ai/chat/CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards

* Follow PEP 8 style guide
* Add docstrings to all functions
* Maintain test coverage above 80%
* Update documentation for API changes

### API Client Libraries

We provide client libraries for easy integration:

```bash
# Python
pip install ai-lead-assistant-client

# Node.js
npm install ai-lead-assistant-client

# Go
go get github.com/yourcompany/ai-lead-assistant-go
```

**Python Client Example:**

```python
from ai_lead_assistant import LeadAssistantClient

client = LeadAssistantClient(api_url="http://your-api.com")

response = client.generate_followup(
    model="hybrid",
    lead_name="Ramesh",
    product_name="AI Billing Software",
    lead_score=72
)

print(response.message)
```

---

## 🙏 Acknowledgments

* Meta AI for Llama 2 models
* Hugging Face for Transformers library
* LangChain team for orchestration framework
* FastAPI team for the excellent web framework
