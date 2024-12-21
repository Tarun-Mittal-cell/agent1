"""
Advanced knowledge base system with:
1. FAISS vector storage
2. Semantic search
3. Code pattern matching
4. Multi-language support
5. Auto-updating capabilities
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class CodePattern:
    """Represents a reusable code pattern."""
    language: str
    pattern_type: str  # design_pattern, algorithm, boilerplate, etc.
    code: str
    description: str
    tags: List[str]
    metadata: Dict[str, Any]

@dataclass
class LanguageConfig:
    """Language-specific configuration."""
    name: str
    file_extensions: List[str]
    linters: List[str]
    formatters: List[str]
    test_frameworks: List[str]
    build_tools: List[str]
    package_managers: List[str]
    docker_base_images: List[str]

class KnowledgeBase:
    """
    Advanced knowledge base for code patterns, best practices, and tooling.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(self, 
                 base_path: Path,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 device: str = "cuda"):
        self.base_path = base_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize embedding model
        self.encoder = SentenceTransformer(embedding_model).to(self.device)
        
        # Initialize FAISS indices
        self.pattern_index = None
        self.pattern_docs = []
        
        # Load configurations
        self.language_configs = self._load_language_configs()
        
        # Load knowledge base
        self._load_knowledge_base()
        
    def _load_language_configs(self) -> Dict[str, LanguageConfig]:
        """Load language-specific configurations."""
        config_path = self.base_path / "languages" / "configs.json"
        if not config_path.exists():
            logger.warning(f"No language configs found at {config_path}")
            return {}
            
        with open(config_path) as f:
            configs = json.load(f)
            
        return {
            lang: LanguageConfig(**cfg)
            for lang, cfg in configs.items()
        }
        
    def _load_knowledge_base(self):
        """Load and index all knowledge base documents."""
        logger.info("Loading knowledge base...")
        
        # Load code patterns
        patterns_path = self.base_path / "patterns"
        if not patterns_path.exists():
            logger.warning(f"No patterns found at {patterns_path}")
            return
            
        # Load patterns in parallel
        with ThreadPoolExecutor() as executor:
            pattern_files = list(patterns_path.rglob("*.json"))
            patterns = list(executor.map(self._load_pattern_file, pattern_files))
            
        # Flatten patterns list
        patterns = [p for pattern_list in patterns if pattern_list for p in pattern_list]
        
        if not patterns:
            logger.warning("No patterns loaded")
            return
            
        # Create FAISS index
        self._create_pattern_index(patterns)
        logger.info(f"Loaded {len(patterns)} patterns")
        
    def _load_pattern_file(self, file_path: Path) -> List[CodePattern]:
        """Load patterns from a single file."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            return [CodePattern(**pattern) for pattern in data]
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []
            
    def _create_pattern_index(self, patterns: List[CodePattern]):
        """Create FAISS index for patterns."""
        # Get embeddings for all patterns
        texts = [
            f"{p.description}\n{p.code}\n{' '.join(p.tags)}"
            for p in patterns
        ]
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.pattern_index = faiss.IndexFlatL2(dimension)
        self.pattern_index.add(embeddings.astype(np.float32))
        self.pattern_docs = patterns
        
    def search_patterns(self,
                       query: str,
                       language: Optional[str] = None,
                       pattern_type: Optional[str] = None,
                       k: int = 5) -> List[CodePattern]:
        """
        Search for relevant code patterns using semantic search.
        Optionally filter by language and pattern type.
        """
        if not self.pattern_index:
            return []
            
        # Get query embedding
        query_emb = self.encoder.encode([query], convert_to_tensor=True)
        query_emb = query_emb.cpu().numpy().astype(np.float32)
        
        # Search
        D, I = self.pattern_index.search(query_emb, k * 2)  # Get extra results for filtering
        
        # Filter results
        results = []
        for idx in I[0]:
            pattern = self.pattern_docs[idx]
            if language and pattern.language != language:
                continue
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            results.append(pattern)
            if len(results) >= k:
                break
                
        return results
        
    def get_language_config(self, language: str) -> Optional[LanguageConfig]:
        """Get configuration for a specific language."""
        return self.language_configs.get(language.lower())
        
    def get_tool_commands(self,
                         language: str,
                         tool_type: str) -> List[str]:
        """Get commands for language-specific tools."""
        config = self.get_language_config(language)
        if not config:
            return []
            
        if tool_type == "lint":
            return config.linters
        elif tool_type == "format":
            return config.formatters
        elif tool_type == "test":
            return config.test_frameworks
        elif tool_type == "build":
            return config.build_tools
        return []
        
    def add_pattern(self, pattern: CodePattern):
        """Add a new pattern to the knowledge base."""
        # Create embedding
        text = f"{pattern.description}\n{pattern.code}\n{' '.join(pattern.tags)}"
        embedding = self.encoder.encode([text], convert_to_tensor=True)
        embedding = embedding.cpu().numpy().astype(np.float32)
        
        # Add to index
        self.pattern_index.add(embedding)
        self.pattern_docs.append(pattern)
        
        # Save to disk
        pattern_path = (self.base_path / "patterns" / 
                       pattern.language / f"{pattern.pattern_type}.json")
        pattern_path.parent.mkdir(parents=True, exist_ok=True)
        
        patterns = []
        if pattern_path.exists():
            with open(pattern_path) as f:
                patterns = json.load(f)
                
        patterns.append(pattern.__dict__)
        
        with open(pattern_path, "w") as f:
            json.dump(patterns, f, indent=2)
            
    def save(self):
        """Save the entire knowledge base to disk."""
        # Save FAISS index
        index_path = self.base_path / "faiss_index.bin"
        faiss.write_index(self.pattern_index, str(index_path))
        
        # Save patterns
        patterns_path = self.base_path / "patterns.json"
        with open(patterns_path, "w") as f:
            json.dump([p.__dict__ for p in self.pattern_docs], f, indent=2)
            
    def load(self):
        """Load knowledge base from disk."""
        # Load FAISS index
        index_path = self.base_path / "faiss_index.bin"
        if index_path.exists():
            self.pattern_index = faiss.read_index(str(index_path))
            
        # Load patterns
        patterns_path = self.base_path / "patterns.json"
        if patterns_path.exists():
            with open(patterns_path) as f:
                patterns = json.load(f)
            self.pattern_docs = [CodePattern(**p) for p in patterns]