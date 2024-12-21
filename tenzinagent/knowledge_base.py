"""
Knowledge base system using FAISS for vector search and retrieval of:
1. Code templates
2. Design patterns
3. Best practices
4. UI/UX patterns
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Knowledge base with vector search capabilities."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.index = None
        self.documents = []
        self.templates = {}
        self.patterns = {}
        self.design_systems = {}
        
        # Load knowledge
        self._load_knowledge()
        
    def _load_knowledge(self):
        """Load all knowledge from files."""
        logger.info("Loading knowledge base...")
        
        # Load frameworks
        frameworks_path = self.base_path / "frameworks"
        if frameworks_path.exists():
            for file_path in frameworks_path.glob("*.json"):
                with open(file_path) as f:
                    self.templates[file_path.stem] = json.load(f)
                    
        # Load patterns
        patterns_path = self.base_path / "patterns"
        if patterns_path.exists():
            for file_path in patterns_path.glob("*.json"):
                with open(file_path) as f:
                    self.patterns[file_path.stem] = json.load(f)
                    
        # Load design systems
        design_path = self.base_path / "design"
        if design_path.exists():
            for file_path in design_path.glob("*.json"):
                with open(file_path) as f:
                    self.design_systems[file_path.stem] = json.load(f)
                    
        # Build search index
        self._build_index()
        
    def _build_index(self):
        """Build FAISS index from documents."""
        logger.info("Building search index...")
        
        # Collect all text content
        texts = []
        for template in self.templates.values():
            texts.extend(self._extract_text(template))
        for pattern in self.patterns.values():
            texts.extend(self._extract_text(pattern))
            
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Build index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.documents = texts
        
    def _extract_text(self, obj: Any) -> List[str]:
        """Extract text content from nested object."""
        texts = []
        if isinstance(obj, dict):
            for value in obj.values():
                texts.extend(self._extract_text(value))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(self._extract_text(item))
        elif isinstance(obj, str):
            texts.append(obj)
        return texts
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search knowledge base."""
        if not self.index:
            return []
            
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            k
        )
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "content": self.documents[idx],
                "score": float(distances[0][i])
            })
            
        return results
        
    def get_templates(self, type: str, language: str) -> Dict:
        """Get templates for specific type and language."""
        templates = {}
        
        # Get framework templates
        if type in self.templates:
            framework_templates = self.templates[type]
            if language in framework_templates:
                templates.update(framework_templates[language])
                
        return templates
        
    def get_patterns(self, type: str) -> Dict:
        """Get patterns for specific type."""
        patterns = {}
        
        # Get relevant patterns
        if type in self.patterns:
            patterns.update(self.patterns[type])
            
        return patterns
        
    def get_design_system(self, name: str = "modern") -> Dict:
        """Get design system by name."""
        if name in self.design_systems:
            return self.design_systems[name]
        return self.design_systems.get("modern", {})