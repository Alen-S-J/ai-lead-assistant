from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql://user:password@localhost/lead_assistant_db"

# Note: In a real scenario, consider using environment variables for the DB URL.
# defaulting to sqlite for local demo purposes if postgres not available, 
# but keeping the code as requested.
# For this demo environment, let's use sqlite so it actually runs without a postgres server.
DATABASE_URL = "sqlite:///./lead_assistant.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Interaction(Base):
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_name = Column(String, index=True)
    model_type = Column(String)  # rag, finetuned, hybrid
    message = Column(Text)
    strategy = Column(String, nullable=True)
    lead_score = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    
class HybridInteraction(Base):
    __tablename__ = "hybrid_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_name = Column(String, index=True)
    lead_score = Column(Integer)
    message = Column(Text)
    rag_confidence = Column(Float)
    strategy_used = Column(String)
    hybrid_confidence = Column(Float)
    grounding_verified = Column(Boolean)
    personalization_score = Column(Float)
    sources_used = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    response_time_ms = Column(Integer, nullable=True)
    
class LeadProfile(Base):
    __tablename__ = "lead_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    email = Column(String, unique=True)
    company = Column(String)
    industry = Column(String)
    lead_score = Column(Integer, default=50)
    engagement_pattern = Column(String)
    last_interaction = Column(DateTime)
    interaction_count = Column(Integer, default=0)
    conversion_probability = Column(Float, default=0.0)
    metadata = Column(JSON)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base.metadata.create_all(bind=engine)
