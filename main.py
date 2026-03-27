"""
AgroLens AI — FastAPI Backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.routes import predict, chat, report


app = FastAPI(
    title="AgroLens AI API",
    description="Backend for Crop Disease Detection & AI Chatbot",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, tags=["Prediction"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(report.router, tags=["Report"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "AgroLens AI API",
        "model_type": settings.MODEL_TYPE,
        "llm_available": bool(settings.GEMINI_API_KEY),
    }
