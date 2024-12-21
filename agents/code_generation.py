"""
Code generation agent for creating application code.
"""
import logging
from pathlib import Path
from typing import Dict, List, Any

from ..llm_integration import LLMProvider
from ..knowledge_base import KnowledgeBase
from ..core import ProjectSpec

logger = logging.getLogger(__name__)

class CodeGenerationAgent:
    """Generates application code using LLM and knowledge base."""
    
    def __init__(self, llm: LLMProvider, kb: KnowledgeBase):
        self.llm = llm
        self.kb = kb
        
    def generate_next_page(self, name: str, components: List[str]) -> str:
        """Generate a Next.js page."""
        # Get page template
        template = self.kb.get_templates("frontend", "next.js")["page"]
        
        # Generate imports
        imports = [
            'import React from "react"',
            'import Head from "next/head"'
        ]
        
        # Add component imports
        for component in components:
            imports.append(f'import {{ {component} }} from "@/components/{component}"')
            
        # Generate page content
        content = self.llm.generate(
            prompt=f"Generate Next.js page {name} using components: {components}",
            provider="openai"
        )
        
        return "\n".join(imports) + "\n\n" + content
        
    def generate_component(self, name: str, design: Dict) -> str:
        """Generate a React component."""
        # Get component template
        template = self.kb.get_templates("frontend", "next.js")["component"]
        
        # Get design tokens
        colors = design["colors"]
        spacing = design.get("spacing", {})
        
        # Generate component
        content = self.llm.generate(
            prompt=f"Generate React component {name} with design: {design}",
            provider="openai"
        )
        
        return content
        
    def generate_next_config(self) -> str:
        """Generate Next.js configuration."""
        return """
        /** @type {import('next').NextConfig} */
        const nextConfig = {
            reactStrictMode: true,
            swcMinify: true,
        }
        
        module.exports = nextConfig
        """
        
    def generate_tailwind_config(self, design: Dict) -> str:
        """Generate Tailwind CSS configuration."""
        colors = json.dumps(design["colors"], indent=2)
        return f"""
        /** @type {{import('tailwindcss').Config}} */
        module.exports = {{
            content: [
                './pages/**/*.{{js,ts,jsx,tsx}}',
                './components/**/*.{{js,ts,jsx,tsx}}',
            ],
            theme: {{
                extend: {{
                    colors: {colors}
                }}
            }},
            plugins: []
        }}
        """
        
    def generate_fastapi_main(self) -> str:
        """Generate FastAPI main application file."""
        return """
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI()
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        async def root():
            return {"message": "Hello World"}
        """
        
    def generate_db_config(self, database: str) -> str:
        """Generate database configuration."""
        if database == "postgresql":
            return """
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker
            
            DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"
            
            engine = create_async_engine(DATABASE_URL, echo=True)
            async_session = sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            
            async def get_session() -> AsyncSession:
                async with async_session() as session:
                    yield session
            """
            
    def generate_models(self, spec: ProjectSpec) -> Dict[str, str]:
        """Generate database models."""
        models = {}
        
        # User model
        models["user"] = """
        from sqlalchemy import Column, Integer, String
        from sqlalchemy.ext.declarative import declarative_base
        
        Base = declarative_base()
        
        class User(Base):
            __tablename__ = "users"
            
            id = Column(Integer, primary_key=True, index=True)
            email = Column(String, unique=True, index=True)
            name = Column(String)
            password = Column(String)
        """
        
        return models
        
    def generate_routes(self, spec: ProjectSpec) -> Dict[str, str]:
        """Generate API routes."""
        routes = {}
        
        # Auth routes
        routes["auth"] = """
        from fastapi import APIRouter, Depends, HTTPException
        from sqlalchemy.ext.asyncio import AsyncSession
        
        from ..db.database import get_session
        from ..models.user import User
        
        router = APIRouter()
        
        @router.post("/register")
        async def register(user: UserCreate, db: AsyncSession = Depends(get_session)):
            # Implementation
            pass
            
        @router.post("/login")
        async def login(user: UserLogin, db: AsyncSession = Depends(get_session)):
            # Implementation
            pass
        """
        
        return routes
        
    def generate_dockerfile(self, spec: ProjectSpec) -> str:
        """Generate Dockerfile."""
        return """
        # Frontend
        FROM node:18-alpine AS frontend
        WORKDIR /app
        COPY frontend/package*.json ./
        RUN npm install
        COPY frontend .
        RUN npm run build
        
        # Backend
        FROM python:3.11-slim
        WORKDIR /app
        COPY backend/requirements.txt .
        RUN pip install -r requirements.txt
        COPY backend .
        
        CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
        """
        
    def generate_docker_compose(self, spec: ProjectSpec) -> str:
        """Generate Docker Compose configuration."""
        return """
        version: '3.8'
        
        services:
          frontend:
            build:
              context: ./frontend
              dockerfile: Dockerfile
            ports:
              - "3000:3000"
            environment:
              - NODE_ENV=production
              
          backend:
            build:
              context: ./backend
              dockerfile: Dockerfile
            ports:
              - "8000:8000"
            depends_on:
              - postgres
              
          postgres:
            image: postgres:14-alpine
            environment:
              - POSTGRES_USER=user
              - POSTGRES_PASSWORD=password
              - POSTGRES_DB=app
            volumes:
              - postgres_data:/var/lib/postgresql/data
              
        volumes:
          postgres_data:
        """
        
    def generate_kubernetes(self, spec: ProjectSpec) -> Dict[str, str]:
        """Generate Kubernetes manifests."""
        manifests = {}
        
        # Frontend deployment
        manifests["frontend"] = """
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: frontend
        spec:
          replicas: 3
          selector:
            matchLabels:
              app: frontend
          template:
            metadata:
              labels:
                app: frontend
            spec:
              containers:
              - name: frontend
                image: frontend:latest
                ports:
                - containerPort: 3000
        """
        
        # Backend deployment
        manifests["backend"] = """
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: backend
        spec:
          replicas: 3
          selector:
            matchLabels:
              app: backend
          template:
            metadata:
              labels:
                app: backend
            spec:
              containers:
              - name: backend
                image: backend:latest
                ports:
                - containerPort: 8000
        """
        
        return manifests