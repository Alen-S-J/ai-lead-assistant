from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
# Mocking langchain imports to allow the app to run without heavy dependencies for the demo
# In production, you would use the actual imports
try:
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import HuggingFacePipeline
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError:
    # Mocks for demonstration if libs are missing
    class PromptTemplate:
        def __init__(self, input_variables, template): self.template = template
        def format(self, **kwargs): return self.template.format(**kwargs)
    
# import chromadb # Commenting out to avoid dependency issues in this environment
from typing import Optional, List
import json
import redis # We will mock redis if not available

# Mock Redis Client
class MockRedis:
    def __init__(self, **kwargs): self.store = {}
    def get(self, key): return self.store.get(key)
    def setex(self, key, time, value): self.store[key] = value

try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
except (ImportError, redis.ConnectionError):
    redis_client = MockRedis()

app = FastAPI(title="RAG Lead Assistant API")

RAG_SYSTEM_PROMPT = """You are an AI Lead Follow-Up Assistant with access to real-time company knowledge through vector retrieval.

ROLE & IDENTITY:
- Professional sales assistant for {company_name}
- Expert in {product_category}
- Empathetic and consultative, never pushy

RETRIEVAL CONTEXT:
The following documents were retrieved from our knowledge base:
---
{retrieved_documents}
---
Retrieval Confidence Score: {confidence_score}%
Sources: {source_metadata}

LEAD INFORMATION:
- Name: {lead_name}
- Product Interest: {product_name}
- Previous Interaction: {interaction_history}
- Current Stage: {lead_stage}
- Days Since Last Contact: {days_since_contact}
- Engagement Score: {engagement_score}/100

RESPONSE PROTOCOL:
1. GROUNDING: Base all product claims on retrieved documents
2. ATTRIBUTION: Reference information naturally without explicit citations
3. GAPS: If retrieval doesn't cover the question, acknowledge and offer human handoff
4. VERIFICATION: Cross-check retrieved chunks for consistency
5. FRESHNESS: Prioritize recent documents (metadata: last_updated)

TONE REQUIREMENTS:
- Primary Tone: {tone_required}
- Formality Level: {formality_level}
- Personalization: Use {lead_name} once, naturally
- Empathy: Acknowledge previous interaction context

STRICT CONSTRAINTS:
- Maximum: 120 words
- Include: ONE re-engagement question
- Avoid: Jargon, pushy language, false urgency
- No hallucination: Only use retrieved facts

ANTI-HALLUCINATION PROTOCOL:
✗ DO NOT invent pricing, features, or timelines
✗ DO NOT claim capabilities not in retrieved docs
✗ If uncertain, say: "Let me connect you with our team for specifics on {topic}"

OUTPUT FORMAT:
Generate ONLY the message body. No subject, signature, or metadata.

RETRIEVAL QUALITY CHECK:
Before responding, verify:
- Retrieved docs directly answer the lead's implicit question: {yes/no}
- Information is current (< 6 months old): {yes/no}
- No contradictions across retrieved chunks: {yes/no}

If any check fails, use fallback template:
"Hi {lead_name}, I want to ensure you get the most accurate information about {product_name}. Our specialist can walk you through {specific_topic}. Does {timeframe} work for a brief chat?"

Now generate the follow-up message:
"""

class LeadContext(BaseModel):
    lead_name: str = Field(..., example="Ramesh")
    product_name: str = Field(..., example="AI Billing Software")
    interaction_history: str = Field(..., example="Asked about pricing, stopped responding")
    lead_stage: str = Field(..., example="consideration")
    engagement_score: int = Field(..., ge=0, le=100)
    days_since_contact: int = Field(..., ge=0)
    tone_required: str = Field(default="friendly_helpful")
    company_name: str = Field(default="TechCorp")
    product_category: str = Field(default="SaaS")

class RAGConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=10)
    retrieval_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    source_types: List[str] = Field(default=["pricing", "features", "case_studies"])
    use_hybrid_search: bool = Field(default=True)

class RAGRequest(BaseModel):
    lead_context: LeadContext
    rag_config: RAGConfig = RAGConfig()
    max_tokens: int = Field(default=150, le=200)

class RAGResponse(BaseModel):
    message: str
    retrieval_sources: List[str]
    confidence_score: float
    model_used: str = "rag_llm"
    cached: bool = False

# Mock Vector Store for Demo
class MockVectorStore:
    def similarity_search_with_score(self, query, k, filter=None):
        # Return mock docs
        class Doc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        
        return [
            (Doc(f"Doc regarding {query}. Price is $50.", {"source": "pricing_doc", "type": "pricing"}), 0.9),
            (Doc("Case study: Client X saved 30%.", {"source": "case_study_X", "type": "case_studies"}), 0.85)
        ]

# Initialize RAG components
def initialize_rag_system():
    # In a real app, strict initialization. Here we mock.
    vectorstore = MockVectorStore()
    
    # Mock LLM
    def llm(prompt, max_new_tokens=150):
        # Extract name to personalize
        import re
        name_match = re.search(r"Name: (.*?)\n", prompt)
        name = name_match.group(1) if name_match else "Valid Name"
        return f"Hi {name}, based on our pricing documents, the solution is $50/month. Client X saw great results. Does this fit your budget?"

    return vectorstore, llm

vectorstore, llm = initialize_rag_system()

# Prompt template
RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "company_name", "product_category", "retrieved_documents",
        "confidence_score", "source_metadata", "lead_name", "product_name",
        "interaction_history", "lead_stage", "days_since_contact",
        "engagement_score", "tone_required", "formality_level"
    ],
    template=RAG_SYSTEM_PROMPT
)

@app.post("/generate-followup", response_model=RAGResponse)
async def generate_rag_followup(request: RAGRequest, background_tasks: BackgroundTasks):
    """
    Generate lead follow-up using RAG approach
    """
    # Check cache
    cache_key = f"rag:{request.lead_context.lead_name}:{request.lead_context.product_name}"
    cached_response = redis_client.get(cache_key)
    
    if cached_response:
        if isinstance(cached_response, bytes):
            cached_response = cached_response.decode('utf-8')
        return RAGResponse(**json.loads(cached_response), cached=True)
    
    # Retrieve relevant documents
    query = f"{request.lead_context.product_name} {request.lead_context.interaction_history}"
    
    retrieval_results = vectorstore.similarity_search_with_score(
        query,
        k=request.rag_config.top_k,
        filter={"type": {"$in": request.rag_config.source_types}}
    )
    
    # Filter by threshold
    filtered_docs = [
        (doc, score) for doc, score in retrieval_results
        if score >= request.rag_config.retrieval_threshold
    ]
    
    if not filtered_docs:
        # Fallback response
        message = f"Hi {request.lead_context.lead_name}, I want to ensure you get accurate information about {request.lead_context.product_name}. Would you like to schedule a quick call with our specialist?"
        confidence_score = 0.0
        sources = []
    else:
        # Prepare context
        retrieved_text = "\n\n".join([doc.page_content for doc, _ in filtered_docs])
        sources = [doc.metadata.get("source", "unknown") for doc, _ in filtered_docs]
        confidence_score = sum(score for _, score in filtered_docs) / len(filtered_docs)
        
        # Format prompt
        formatted_prompt = RAG_PROMPT_TEMPLATE.format(
            company_name=request.lead_context.company_name,
            product_category=request.lead_context.product_category,
            retrieved_documents=retrieved_text,
            confidence_score=round(confidence_score * 100, 2),
            source_metadata=", ".join(sources),
            lead_name=request.lead_context.lead_name,
            product_name=request.lead_context.product_name,
            interaction_history=request.lead_context.interaction_history,
            lead_stage=request.lead_context.lead_stage,
            days_since_contact=request.lead_context.days_since_contact,
            engagement_score=request.lead_context.engagement_score,
            tone_required=request.lead_context.tone_required,
            formality_level="professional" if request.lead_context.engagement_score > 70 else "casual"
        )
        
        # Generate response
        message = llm(formatted_prompt, max_new_tokens=request.max_tokens)
        
        # Post-process: ensure word count
        words = message.split()
        if len(words) > 120:
            message = " ".join(words[:120]) + "..."
    
    response = RAGResponse(
        message=message,
        retrieval_sources=sources,
        confidence_score=confidence_score,
        model_used="rag_llm"
    )
    
    # Cache response
    redis_client.setex(cache_key, 3600, json.dumps(response.dict()))
    
    # Log to database (background task)
    background_tasks.add_task(log_interaction, request, response)
    
    return response

def log_interaction(request: RAGRequest, response: RAGResponse):
    # Implement database logging
    pass
