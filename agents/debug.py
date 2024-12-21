"""
Debug agent for code analysis and optimization.
"""
import logging
from typing import Dict, List, Any

from ..llm_integration import LLMProvider

logger = logging.getLogger(__name__)

class DebugAgent:
    """Analyzes and optimizes code."""
    
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        
    def debug_code(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code for issues."""
        logger.info(f"Analyzing {language} code...")
        
        # Get analysis from LLM
        analysis = self.llm.generate(
            prompt=f"Analyze this {language} code for issues:\n\n{code}",
            provider="openai"
        )
        
        return {
            "issues": analysis
        }
        
    def optimize_code(self, code: str) -> str:
        """Optimize code for performance."""
        logger.info("Optimizing code...")
        
        # Get optimized version from LLM
        optimized = self.llm.generate(
            prompt=f"Optimize this code for performance:\n\n{code}",
            provider="openai"
        )
        
        return optimized