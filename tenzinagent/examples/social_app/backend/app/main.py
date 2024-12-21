"""
FastAPI backend with async database access and caching
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from redis import asyncio as aioredis

from app.core.config import settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.api.v1.api import api_router
from app.cache import Cache

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis cache
redis = aioredis.from_url(
    settings.REDIS_URL,
    encoding="utf-8",
    decode_responses=True
)
cache = Cache(redis)

# Add cache to app state
app.state.cache = cache

# Include routers
app.include_router(
    api_router,
    prefix="/api/v1"
)

@app.on_event("startup")
async def startup_event():
    # Initialize connections
    await cache.init()
    
@app.on_event("shutdown")
async def shutdown_event():
    # Clean up connections
    await cache.close()
    
@app.get("/api/healthcheck")
async def healthcheck():
    """Health check endpoint"""
    return {"status": "healthy"}