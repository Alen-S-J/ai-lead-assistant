from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from prometheus_fastapi_instrumentator import Instrumentator # Commented out to avoid dependency issues for demo
import logging

# Import model routers
from app.models.rag_model import app as rag_app
from app.models.finetuned_model import app as ft_app
from app.models.hybrid_model import app as hybrid_app

# Main application
app = FastAPI(
    title="AI Lead Follow-Up Assistant - Multi-Model Platform",
    description="Production RAG + Fine-tuned + Hybrid LLM System",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
# Instrumentator().instrument(app).expose(app)

# Mount model applications
app.mount("/rag", rag_app)
app.mount("/finetuned", ft_app)
app.mount("/hybrid", hybrid_app)

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "rag": "active",
            "finetuned": "active",
            "hybrid": "active"
        }
    }

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # workers=4
